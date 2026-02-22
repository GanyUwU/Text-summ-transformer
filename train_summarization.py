"""
Fine-tuning Script for Summarization

Loads pretrained weights and fine-tunes on CNN/DailyMail.

Features:
- Loads pretrained weights from pretraining stage
- Lower learning rate (3e-5) for fine-tuning
- Beam search decoding for evaluation
- ROUGE score computation
- Early stopping based on validation metric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import warnings
from tqdm import tqdm
from pathlib import Path
import heapq

from datasets import load_dataset
from model import build_transformer
from pretrain_config import get_finetune_config
from tokenizer_utils import get_tokenizer
from diagnostics import (
    get_layerwise_param_groups, reinit_collapsed_heads,
    compute_coverage_loss, entropy_regularization,
    log_attention_entropy, log_pgen, check_gradient_health,
    print_diagnostic_summary, smooth_nll_loss, pgen_balance_loss
)

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Install rouge-score for ROUGE evaluation: pip install rouge-score")


class SummarizationDataset(Dataset):
    """Dataset for CNN/DailyMail summarization using SentencePiece tokenizer."""
    
    def __init__(self, data, tokenizer, src_seq_len, tgt_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        article = item['article']
        summary = item['highlights']
        
        # Tokenize
        src_tokens = self.tokenizer.encode(article)[:self.src_seq_len - 2]
        tgt_tokens = self.tokenizer.encode(summary)[:self.tgt_seq_len - 2]
        
        # Encoder input: [BOS] + article + [EOS] + [PAD]
        enc_input = [self.bos_id] + src_tokens + [self.eos_id]
        enc_padding = max(0, self.src_seq_len - len(enc_input))
        enc_input = enc_input[:self.src_seq_len] + [self.pad_id] * enc_padding
        
        # Decoder input: [BOS] + summary
        dec_input = [self.bos_id] + tgt_tokens
        dec_padding = max(0, self.tgt_seq_len - len(dec_input))
        dec_input = dec_input[:self.tgt_seq_len] + [self.pad_id] * dec_padding
        
        # Label: summary + [EOS]
        label = tgt_tokens + [self.eos_id]
        label_padding = max(0, self.tgt_seq_len - len(label))
        label = label[:self.tgt_seq_len] + [self.pad_id] * label_padding
        
        # Tensors
        encoder_input = torch.tensor(enc_input, dtype=torch.long)
        decoder_input = torch.tensor(dec_input, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        # Masks
        encoder_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)
        decoder_mask = (decoder_input != self.pad_id).unsqueeze(0) & self._causal_mask(self.tgt_seq_len)
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label,
            'src_text': article,
            'tgt_text': summary,
        }
    
    def _causal_mask(self, size):
        return torch.tril(torch.ones((1, size, size), dtype=torch.bool))


def beam_search_decode(model, encoder_output, encoder_mask, tokenizer, max_len, 
                      beam_size=4, length_penalty=1.0, device='cuda'):
    """
    Beam search decoding for better output quality.
    
    Args:
        model: Transformer model
        encoder_output: Encoded source
        encoder_mask: Source mask
        tokenizer: Tokenizer
        max_len: Maximum output length
        beam_size: Number of beams
        length_penalty: Length normalization factor
        device: Device
    
    Returns:
        Best decoded sequence
    """
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    
    # Initialize beams: (score, sequence)
    beams = [(0.0, [bos_id])]
    completed = []
    
    for step in range(max_len):
        all_candidates = []
        
        for score, seq in beams:
            if seq[-1] == eos_id:
                # Normalize score by length
                length_norm = ((5 + len(seq)) / 6) ** length_penalty
                completed.append((score / length_norm, seq))
                continue
            
            # Build decoder input
            decoder_input = torch.tensor([seq], dtype=torch.long).to(device)
            decoder_mask = torch.tril(
                torch.ones((1, 1, len(seq), len(seq)), dtype=torch.bool)
            ).to(device)
            
            # Forward pass
            with torch.no_grad():
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )
                logits = model.project(decoder_output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Get top-k tokens
            top_log_probs, top_tokens = torch.topk(log_probs[0], beam_size)
            
            for i in range(beam_size):
                token = top_tokens[i].item()
                new_score = score + top_log_probs[i].item()
                new_seq = seq + [token]
                all_candidates.append((new_score, new_seq))
        
        # Select top beams
        if not all_candidates:
            break
        
        # Sort by score (descending)
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_size]
        
        # Early stop if all beams completed
        if all(seq[-1] == eos_id for _, seq in beams):
            for score, seq in beams:
                length_norm = ((5 + len(seq)) / 6) ** length_penalty
                completed.append((score / length_norm, seq))
            break
    
    # Add any remaining beams to completed
    for score, seq in beams:
        if seq[-1] != eos_id:
            length_norm = ((5 + len(seq)) / 6) ** length_penalty
            completed.append((score / length_norm, seq))
    
    if not completed:
        return [bos_id]
    
    # Return best sequence
    best = max(completed, key=lambda x: x[0])
    return best[1]


def greedy_decode(model, encoder_output, encoder_mask, encoder_input_ids, tokenizer, max_len, device):
    """
    Greedy decoding with support for Copy Mechanism.
    """
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id
    
    decoder_input = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    
    use_copy = model.copy_mechanism is not None
    
    for _ in range(max_len):
        decoder_mask = torch.tril(
            torch.ones((1, 1, decoder_input.size(1), decoder_input.size(1)), dtype=torch.bool)
        ).to(device)
        
        with torch.no_grad():
            if use_copy:
                # Get blended distribution
                decoder_output, cross_attn = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask,
                    return_cross_attn=True
                )
                vocab_logits = model.project(decoder_output)
                
                # Get decoder embeddings for p_gen
                tgt_embed = model.tgt_embed(decoder_input)
                tgt_embed = model.tgt_pos(tgt_embed)
                
                # Context vector
                context_vector = torch.bmm(cross_attn, encoder_output)
                
                # Blended distribution
                final_dist, p_gen = model.copy_mechanism(
                    decoder_output, context_vector, tgt_embed,
                    vocab_logits, cross_attn, encoder_input_ids
                )
                
                # Argmax on blended distribution
                next_token = torch.argmax(final_dist[:, -1, :], dim=-1).item()
            else:
                # Standard decoding
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )
                logits = model.project(decoder_output[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).item()
        
        decoder_input = torch.cat([
            decoder_input,
            torch.tensor([[next_token]], dtype=torch.long).to(device)
        ], dim=1)
        
        if next_token == eos_id:
            break
            
    return decoder_input[0].tolist()


def compute_rouge(predictions, references):
    """Compute ROUGE scores."""
    if not ROUGE_AVAILABLE:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores['rouge1'].append(result['rouge1'].fmeasure)
        scores['rouge2'].append(result['rouge2'].fmeasure)
        scores['rougeL'].append(result['rougeL'].fmeasure)
    
    return {k: np.mean(v) for k, v in scores.items()}


def run_validation(model, val_loader, tokenizer, config, device, num_examples=5):
    """Run validation with beam search and ROUGE."""
    model.eval()
    
    predictions = []
    references = []
    
    print("\n" + "-"*60)
    print("VALIDATION")
    print("-"*60)
    #some with loop
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_examples:
                break
            
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Encode
            enc_out = model.encode(encoder_input, encoder_mask)
            
            # Decode - ALWAYS use greedy decode with copy support for now
            # (Beam search with copy mechanism is complex and currently unimplemented)
            out_ids = greedy_decode(
                model, enc_out, encoder_mask, encoder_input,
                tokenizer, config['tgt_seq_len'], device
            )
            
            # Decode to text
            decoded = tokenizer.decode(out_ids)
            predictions.append(decoded)
            
            # Label
            lbl_ids = batch['label'][0].tolist()
            if tokenizer.eos_id in lbl_ids:
                lbl_ids = lbl_ids[:lbl_ids.index(tokenizer.eos_id)]
            ref_text = tokenizer.decode(lbl_ids)
            references.append(ref_text)
            
            if i < num_examples:
                print(f"\nExample {i+1}:")
                print(f"  REF: {ref_text[:100]}...")
                print(f"  GEN: {decoded[:100]}...")
    
    # Compute ROUGE
    rouge_scores = compute_rouge(predictions, references)
    
    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print("-"*60)
    
    return rouge_scores


def finetune():
    """Main fine-tuning loop."""
    config = get_finetune_config()
    
    print("\n" + "="*70)
    print("          FINE-TUNING ON CNN/DAILYMAIL")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary: {vocab_size}")
    
    # Load dataset
    print("\nLoading CNN/DailyMail...")
    train_data = load_dataset(
        config['datasource'],
        config['dataset_version'],
        split=f"train[:{config['train_samples']}]"
    )
    val_data = load_dataset(
        config['datasource'],
        config['dataset_version'],
        split=f"validation[:{config['val_samples']}]"
    )
    
    train_dataset = SummarizationDataset(
        train_data, tokenizer, config['src_seq_len'], config['tgt_seq_len']
    )
    val_dataset = SummarizationDataset(
        val_data, tokenizer, config['src_seq_len'], config['tgt_seq_len']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Build model
    model = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=config['src_seq_len'],
        tgt_seq_len=config['tgt_seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff'],
        share_weights=config.get('share_weights', True),
        use_copy=config.get('use_copy', True),
    ).to(device)
    
    use_copy = config.get('use_copy', True) and model.copy_mechanism is not None
    
    # Load pretrained weights
    pretrain_path = config.get('pretrain_weights')
    if pretrain_path and Path(pretrain_path).exists():
        print(f"\n✓ Loading pretrained weights: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filter: load with intelligent slicing for positional embeddings
        loaded, skipped = [], []
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded.append(k)
                elif 'pos.pe' in k and model_dict[k].shape[-1] == v.shape[-1]:
                    # Intelligent slicing for Positional Embeddings (Different seq_len)
                    # v shape: (1, seq_old, d_model), model_dict shape: (1, seq_new, d_model)
                    min_seq = min(model_dict[k].shape[1], v.shape[1])
                    model_dict[k][:, :min_seq, :] = v[:, :min_seq, :]
                    loaded.append(f"{k} (sliced {min_seq} steps)")
                else:
                    skipped.append(k)
            else:
                skipped.append(k)
        
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(loaded)}/{len(pretrained_dict)} weight tensors")
        if skipped:
            print(f"  Skipped {len(skipped)} (shape mismatch or new layers):")
            for s in skipped[:10]:
                print(f"    - {s}")
            if len(skipped) > 10:
                print(f"    ... and {len(skipped)-10} more")
        print(f"  Pretrain loss was: {checkpoint.get('loss', 'N/A')}")
    else:
        print("\n⚠ No pretrained weights found. Training from scratch.")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Reinitialize collapsed decoder heads (gives them a fresh start)
    if config.get('reinit_decoder_heads', True):
        reinit_collapsed_heads(model)
    
    # Optimizer with layer-wise LR (decoder gets lower LR to prevent collapse)
    decoder_lr_scale = config.get('decoder_lr_scale', 0.33)
    print(f"\n📊 Layer-wise Learning Rates (decoder scale: {decoder_lr_scale}):")
    param_groups = get_layerwise_param_groups(model, config['lr'], decoder_lr_scale)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    total_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation']
    warmup_steps = config['warmup_steps']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP
    scaler = GradScaler() if config['use_amp'] else None
    
    # Loss - different depending on copy mechanism
    if use_copy:
        # Copy mechanism outputs probabilities, so use NLLLoss on log-probs
        loss_fn = nn.NLLLoss(
            ignore_index=tokenizer.pad_id,
            # Note: label_smoothing not available in NLLLoss, applied manually if needed
        )
    else:
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=config['label_smoothing']
        )
    
    # TensorBoard
    writer = SummaryWriter(config['experiment_name'])
    
    # Training
    # Diagnostic config
    entropy_reg_weight = config.get('entropy_reg_weight', 1e-3)
    coverage_loss_weight = config.get('coverage_loss_weight', 0.1)
    diagnostic_every = config.get('diagnostic_every', 100)
    
    print("\n" + "-"*50)
    print("TRAINING (with diagnostics)")
    print("-"*50)
    print(f"Epochs: {config['num_epochs']}")
    print(f"LR: {config['lr']} (encoder) / {config['lr'] * decoder_lr_scale} (decoder)")
    print(f"Entropy reg: {entropy_reg_weight}, Coverage loss: {coverage_loss_weight}")
    print(f"Diagnostics every: {diagnostic_every} steps")
    print(f"Early stopping: patience {config['patience']}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    best_rouge = 0.0
    patience_counter = 0
    global_step = 0
    accumulation_steps = config['gradient_accumulation']
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress):
            enc_input = batch['encoder_input'].to(device)
            dec_input = batch['decoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            dec_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass with diagnostic hooks
            amp_enabled = config['use_amp']
            with autocast(enabled=amp_enabled):
                if use_copy:
                    final_dist, p_gen = model.forward_with_copy(
                        enc_input, enc_mask, dec_input, dec_mask
                    )
                    log_probs = torch.log(final_dist + 1e-12)
                    
                    # Label smoothing for copy path
                    if config.get('label_smoothing', 0) > 0:
                        loss = smooth_nll_loss(
                            log_probs.view(-1, vocab_size), label.view(-1),
                            smoothing=config['label_smoothing'],
                            ignore_index=tokenizer.pad_id
                        )
                    else:
                        loss = loss_fn(log_probs.view(-1, vocab_size), label.view(-1))
                else:
                    enc_out = model.encode(enc_input, enc_mask)
                    dec_out = model.decode(enc_out, enc_mask, dec_input, dec_mask)
                    logits = model.project(dec_out)
                    loss = loss_fn(logits.view(-1, vocab_size), label.view(-1))
                
                # ── Auxiliary losses ──
                # Coverage loss (from cross-attention in last decoder layer)
                if use_copy and coverage_loss_weight > 0:
                    last_layer = model.decoder.layers[-1]
                    cross_attn = last_layer.cross_attention_block.attention_scores
                    if cross_attn is not None:
                        # cross_attn: (B, H, T_out, T_in) → avg over heads
                        avg_cross = cross_attn.mean(dim=1)  # (B, T_out, T_in)
                        cov_loss = compute_coverage_loss(avg_cross)
                        loss = loss + coverage_loss_weight * cov_loss
                
                # Entropy regularization (prevent decoder self-attention collapse)
                if entropy_reg_weight > 0:
                    decoder_layer2 = model.decoder.layers[2]
                    self_attn = decoder_layer2.self_attention_block.attention_scores
                    if self_attn is not None:
                        ent_loss = entropy_regularization(self_attn, target_entropy=2.0)
                        loss = loss + entropy_reg_weight * ent_loss
                
                # p_gen balance (prevent copy mechanism domination)
                if use_copy and p_gen is not None:
                    pgen_aux = 0.01 * pgen_balance_loss(p_gen)
                    loss = loss + pgen_aux
                
                loss = loss / accumulation_steps
            
            # Backward
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item() * accumulation_steps
            
            # Update
            if (batch_idx + 1) % accumulation_steps == 0:
                if config['use_amp']:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                
                if config['use_amp']:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                writer.add_scalar('finetune/loss', loss.item() * accumulation_steps, global_step)
                
                # ── Periodic diagnostics ──
                if global_step % diagnostic_every == 0:
                    # Log attention entropy for decoder layers
                    for li, layer in enumerate(model.decoder.layers):
                        sa_attn = layer.self_attention_block.attention_scores
                        if sa_attn is not None:
                            log_attention_entropy(writer, sa_attn, f"dec_self_L{li}", global_step)
                        ca_attn = layer.cross_attention_block.attention_scores
                        if ca_attn is not None:
                            log_attention_entropy(writer, ca_attn, f"dec_cross_L{li}", global_step)
                    
                    # Log p_gen if using copy
                    if use_copy and p_gen is not None:
                        log_pgen(writer, p_gen, global_step)
                    
                    # Gradient health check
                    grad_report = check_gradient_health(model, writer, global_step)
                    if grad_report['warnings']:
                        for w in grad_report['warnings'][:3]:
                            tqdm.write(f"  {w}")
                
                # ── Intra-epoch Validation ──
                if global_step % config['save_every'] == 0:
                    # Validation with beam search + ROUGE
                    rouge_scores = run_validation(
                        model, val_loader, tokenizer, config, device,
                        num_examples=config['num_validation_examples']
                    )
                    
                    writer.add_scalar('finetune/rouge1', rouge_scores['rouge1'], global_step)
                    writer.add_scalar('finetune/rouge2', rouge_scores['rouge2'], global_step)
                    writer.add_scalar('finetune/rougeL', rouge_scores['rougeL'], global_step)
                    
                    # Save best checkpoint
                    current_rouge = rouge_scores['rouge1']
                    if current_rouge > best_rouge:
                        best_rouge = current_rouge
                        patience_counter = 0
                        best_path = Path(config['model_folder']) / f"{config['model_basename']}best.pt"
                        torch.save({
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'best_rouge1': best_rouge,
                        }, best_path)
                        tqdm.write(f"  ⭐ New best ROUGE-1: {best_rouge:.4f} (Saved to {best_path})")
                    else:
                        patience_counter += 1
                        if patience_counter >= config['patience']:
                            tqdm.write("  ⚠️ Early stopping triggered!")
                            return
                    
                    # Periodic checkpoint
                    ckpt_path = Path(config['model_folder']) / f"{config['model_basename']}step_{global_step}.pt"
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item() * accumulation_steps,
                    }, ckpt_path)
            
            progress.set_postfix({'loss': f'{loss.item() * accumulation_steps:.3f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Early stopping
        current_rouge = rouge_scores['rouge1']
        if current_rouge > best_rouge:
            best_rouge = current_rouge
            patience_counter = 0
            best_path = Path(config['model_folder']) / f"{config['model_basename']}best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'rouge1': current_rouge,
            }, best_path)
            print(f"✓ New best model! ROUGE-1: {best_rouge:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config['patience']}")
            
            if patience_counter >= config['patience']:
                print("\n⚠ Early stopping triggered!")
                break
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"\nBest ROUGE-1: {best_rouge:.4f}")
    print(f"Models saved in: {config['model_folder']}/")
    
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    finetune()