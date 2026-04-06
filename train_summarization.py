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
import math

from datasets import load_dataset, exceptions as ds_exceptions
import random
from model import build_transformer
from pretrain_config import get_finetune_config
from tokenizer_utils import get_tokenizer
from checkpoint_utils import load_checkpoint
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
    
    def __init__(self, data, tokenizer, src_seq_len, tgt_seq_len, lead_mask_prob=0.0):
        self.data = data
        self.tokenizer = tokenizer
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.lead_mask_prob = lead_mask_prob
        
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        
        # Pre-generate causal mask
        self.causal_mask = torch.tril(torch.ones((1, tgt_seq_len, tgt_seq_len), dtype=torch.bool))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Support fallback keys for multi-dataset (CNN/DM, XSum, SAMSum)
        article = item.get('article', item.get('document', item.get('dialogue', '')))
        summary = item.get('highlights', item.get('summary', ''))
        
        if self.lead_mask_prob > 0 and random.random() < self.lead_mask_prob:
            # Mask the lead: find first 1-2 sentence endings
            sentences = article.split('. ')
            if len(sentences) > 2:
                # Drop first 1-2 sentences randomly
                num_to_drop = random.randint(1, 2)
                article = '. '.join(sentences[num_to_drop:]).strip()
            elif len(sentences) > 1:
                article = sentences[1].strip()

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
        
        # Masks — return 3D shapes. DataLoader stacking will make them 4D: (B, 1, Tq, Tk)
        # Encoder mask: (1, 1, src_len) -> batches to (B, 1, 1, src_len)
        encoder_mask = (encoder_input != self.pad_id).view(1, 1, -1)
        # Decoder mask: (1, tgt_len, tgt_len) -> batches to (B, 1, tgt_len, tgt_len)
        padding_mask = (decoder_input != self.pad_id).view(1, 1, -1)  # (1,1,T)
        
        # Slicing pre-generated mask is faster than creating it every time
        decoder_mask = padding_mask & self.causal_mask[:, :self.tgt_seq_len, :self.tgt_seq_len]
        
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



def greedy_decode(model, encoder_output, encoder_mask, encoder_input_ids, tokenizer, max_len, device, 
                 no_repeat_ngram=3, repetition_penalty=1.2, min_len=10):
    """
    Greedy decoding with support for Copy Mechanism and N-Gram blocking.
    """
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    
    decoder_input = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    generated_tokens = []
    
    use_copy = model.copy_mechanism is not None
    
    # Pre-generate causal mask for current device
    full_causal_mask = torch.tril(torch.ones((1, 1, max_len + 1, max_len + 1), dtype=torch.bool), device=device)
    
    for i in range(max_len):
        # Current length is decoder_input.size(1)
        cur_len = decoder_input.size(1)
        decoder_mask = full_causal_mask[:, :, :cur_len, :cur_len]
        
        with torch.no_grad():
            if use_copy:
                decoder_output, cross_attn = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask,
                    return_cross_attn=True
                )
                vocab_logits = model.project(decoder_output)
                
                # REFACTOR: Avoid pos_encoding recalculation every step if possible, 
                # but for simplicity we keep it matching the model forward structure.
                tgt_embed = model.tgt_pos(model.tgt_embed(decoder_input))
                context_vector = torch.bmm(cross_attn, encoder_output)
                
                final_dist, p_gen = model.copy_mechanism(
                    decoder_output, context_vector, tgt_embed,
                    vocab_logits, cross_attn, encoder_input_ids
                )
                probs = final_dist[:, -1, :]
            else:
                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )
                logits = model.project(decoder_output[:, -1, :])
                probs = torch.softmax(logits, dim=-1)
        
        # Repetition Penalty
        if repetition_penalty != 1.0:
            for token in set(generated_tokens):
                probs[:, token] /= repetition_penalty
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

        # N-Gram blocking
        if no_repeat_ngram > 0 and len(generated_tokens) >= no_repeat_ngram - 1:
            current_gram = tuple(generated_tokens[-(no_repeat_ngram-1):])
            for k in range(len(generated_tokens) - no_repeat_ngram + 1):
                if tuple(generated_tokens[k:k+no_repeat_ngram-1]) == current_gram:
                    probs[:, generated_tokens[k+no_repeat_ngram-1]] = 0.0
            probs /= (probs.sum(dim=-1, keepdim=True) + 1e-12)

        # Force Min Length
        if len(generated_tokens) < min_len:
            probs[:, eos_id] = 0.0
            probs /= (probs.sum(dim=-1, keepdim=True) + 1e-12)

        next_token = torch.argmax(probs, dim=-1).item()
        
        if next_token == eos_id:
            break
            
        generated_tokens.append(next_token)
        decoder_input = torch.cat([
            decoder_input,
            torch.tensor([[next_token]], dtype=torch.long).to(device)
        ], dim=1)
            
    return [bos_id] + generated_tokens


def beam_search_decode(model, encoder_output, encoder_mask, encoder_input_ids, tokenizer, max_len, device, 
                       beam_size=4, no_repeat_ngram=3, length_penalty=0.6, min_len=10):
    # Batched Beam Search decoding with support for Copy Mechanism and N-Gram blocking.
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    
    full_causal_mask = torch.tril(torch.ones((1, 1, max_len + 1, max_len + 1), dtype=torch.bool, device=device))
    
    # Beam: list of dicts for easier batching
    beams = [{"tokens": [bos_id], "score": 0.0, "finished": False}]
    
    use_copy = model.copy_mechanism is not None
    batch_size = 1 # We are decoding one example at a time, but with multiple beams
    
    for step in range(max_len):
        candidates = []
        active_beams = [b for b in beams if not b["finished"]]
        finished_beams = [b for b in beams if b["finished"]]
        
        if not active_beams:
            break
            
        # Batch all active beams into a single forward pass
        # shape: (num_active_beams, current_len)
        decoder_input = torch.tensor([b["tokens"] for b in active_beams], dtype=torch.long, device=device)
        cur_len = decoder_input.size(1)
        decoder_mask = full_causal_mask[:, :, :cur_len, :cur_len] # (1, 1, T, T) - broadcasts to batch
        
        # Expand encoder info for the beam batch
        # encoder_output: (1, S, D) -> (num_active_beams, S, D)
        exp_enc_output = encoder_output.expand(len(active_beams), -1, -1)
        exp_enc_mask = encoder_mask.expand(len(active_beams), -1, -1, -1)
        exp_src_id = encoder_input_ids.expand(len(active_beams), -1)
        
        with torch.no_grad():
            if use_copy:
                decoder_output, cross_attn = model.decode(
                    exp_enc_output, exp_enc_mask, decoder_input, decoder_mask,
                    return_cross_attn=True
                )
                vocab_logits = model.project(decoder_output)
                tgt_embed = model.tgt_pos(model.tgt_embed(decoder_input))
                context_vector = torch.bmm(cross_attn, exp_enc_output)
                final_dist, _ = model.copy_mechanism(
                    decoder_output, context_vector, tgt_embed,
                    vocab_logits, cross_attn, exp_src_id
                )
                all_probs = final_dist[:, -1, :] # (num_active_beams, vocab_size)
            else:
                decoder_output = model.decode(
                    exp_enc_output, exp_enc_mask, decoder_input, decoder_mask
                )
                all_probs = torch.softmax(model.project(decoder_output[:, -1, :]), dim=-1)
        
        # Step through each beam's probabilities
        for i, b in enumerate(active_beams):
            probs = all_probs[i]
            
            # Repetition / N-Gram penalty
            if no_repeat_ngram > 0 and len(b["tokens"]) >= no_repeat_ngram:
                current_gram = tuple(b["tokens"][-(no_repeat_ngram-1):])
                for k in range(len(b["tokens"]) - no_repeat_ngram + 1):
                    if tuple(b["tokens"][k:k+no_repeat_ngram-1]) == current_gram:
                        probs[b["tokens"][k+no_repeat_ngram-1]] = 0.0
                probs /= (probs.sum() + 1e-12)
            
            # Force Min Length
            if len(b["tokens"]) < min_len:
                probs[eos_id] = 0.0
                probs /= (probs.sum() + 1e-12)
                
            top_probs, top_ids = torch.topk(probs, beam_size)
            
            for j in range(beam_size):
                token_id = top_ids[j].item()
                prob = top_probs[j].item()
                new_score = b["score"] + math.log(prob + 1e-12)
                
                is_finished = (token_id == eos_id)
                candidates.append({
                    "tokens": b["tokens"] + [token_id],
                    "score": new_score,
                    "finished": is_finished
                })
        
        # Combine with already finished beams and sort
        all_candidates = candidates + finished_beams
        # Length penalty
        def get_score(cand):
            lp = ((5.0 + len(cand["tokens"])) / 6.0) ** length_penalty
            return cand["score"] / lp
            
        beams = sorted(all_candidates, key=get_score, reverse=True)[:beam_size]
        
        if all(b["finished"] for b in beams):
            break
            
    return beams[0]["tokens"]



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


def run_validation(model, val_loader, tokenizer, config, device, num_examples=5, num_print=5):
    """Run validation with beam search and ROUGE."""
    model.eval()
    
    predictions = []
    references = []
    
    print("\n" + "-"*60)
    print("VALIDATION")
    print("-"*60)
    
    # Randomly select indices to print
    print_indices = random.sample(range(min(len(val_loader), num_examples)), min(num_print, num_examples))
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_examples:
                break
            
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Encode
            enc_out = model.encode(encoder_input, encoder_mask)
            
            # Decode - Use Beam Search if beam_size > 1
            if config.get('beam_size', 1) > 1:
                out_ids = beam_search_decode(
                    model, enc_out, encoder_mask, encoder_input,
                    tokenizer, config['tgt_seq_len'], device,
                    beam_size=config['beam_size'],
                    no_repeat_ngram=3, 
                    length_penalty=config.get('length_penalty', 0.8)
                )
            else:
                out_ids = greedy_decode(
                    model, enc_out, encoder_mask, encoder_input,
                    tokenizer, config['tgt_seq_len'], device,
                    no_repeat_ngram=config.get('no_repeat_ngram', 3)
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
            
            if i in print_indices:
                print(f"\nExample {i+1}:")
                print(f"  REF: {ref_text}")
                print(f"  GEN: {decoded}")
    
    # Compute ROUGE
    rouge_scores = compute_rouge(predictions, references)
    
    print(f"\nROUGE Scores (n={len(predictions)}):")
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print("-" * 60)
    
    return rouge_scores


def save_checkpoint(model, optimizer, scheduler, global_step, loss, rouge1, path):
    """Save a comprehensive training checkpoint."""
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'rouge1': rouge1,
    }
    torch.save(checkpoint, path)
    
    # Simple verification
    if Path(path).exists():
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        return size_mb
    return 0.0


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
    
    # Helper: safe loader that warns and returns an empty iterable if dataset missing
    def safe_load(name, version=None, split=None):
        try:
            if version:
                return load_dataset(name, version, split=split)
            return load_dataset(name, split=split)
        except ds_exceptions.DatasetNotFoundError:
            print(f"Warning: dataset '{name}' not found on the Hub — skipping.")
            return []

    # Load dataset — support mixed-dataset recipes to reduce domain overfitting
    print("\nLoading training data (mixed recipe)...")
    train_samples = config['train_samples']
    train_examples = []
    if config.get('dataset_mix'):
        for ds in config['dataset_mix']:
            name = ds['name']
            version = ds.get('version')
            frac = ds.get('fraction', 0.0)
            n = max(1, int(train_samples * frac))
            print(f"  - Loading {n} samples from {name}{(' v'+str(version)) if version else ''}")
            if version:
                part = safe_load(name, version, split=f"train[:{n}]")
            else:
                part = safe_load(name, split=f"train[:{n}]")
            train_examples.extend(part)
        # Shuffle mixed examples
        random.shuffle(train_examples)
    else:
        print("  - Loading CNN/DailyMail as single source")
        train_examples = safe_load(
            config['datasource'],
            config['dataset_version'],
            split=f"train[:{train_samples}]"
        )

    # Validation remains CNN/DailyMail by default
    val_data = safe_load(
        config['datasource'],
        config['dataset_version'],
        split=f"validation[:{config['val_samples']}]"
    )
    
    train_dataset = SummarizationDataset(
        train_examples, tokenizer, config['src_seq_len'], config['tgt_seq_len'],
        lead_mask_prob=config.get('lead_mask_prob', 0.0)
    )
    # Validation remains unmasked for strict parity check
    val_dataset = SummarizationDataset(
        val_data, tokenizer, config['src_seq_len'], config['tgt_seq_len'],
        lead_mask_prob=0.0
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
    
    # Copy mechanism warmup: disable copy initially to let generator learn
    copy_warmup_steps = config.get('copy_warmup_steps', 0)
    use_copy_initially = config.get('use_copy', True) and model.copy_mechanism is not None
    use_copy = use_copy_initially  # Will be toggled based on global_step later
    
    # Load pretrained weights OR initialize from scratch
    pretrain_path = config.get('pretrain_weights')
    if pretrain_path and Path(pretrain_path).exists():
        print(f"\n✓ Loading pretrained weights: {pretrain_path}")
        checkpoint = load_checkpoint(pretrain_path, map_location=device)
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
        
        # ── CRITICAL FIX: Reset copy mechanism bias AFTER loading weights ──
        # The pretrained checkpoint overwrites our w_gen.bias=2.0 initialization.
        # Without this reset, the model starts with old copy-biased weights,
        # making it stuck in a "copying is easy" loop from Step 0.
        if model.copy_mechanism is not None and hasattr(model.copy_mechanism, 'w_gen'):
            old_bias = model.copy_mechanism.w_gen.bias.data.item()
            import math
            new_bias = 0.0  # sigmoid(0.0) = 0.5 → Neutral starting point
            nn.init.constant_(model.copy_mechanism.w_gen.bias, new_bias)
            print(f"  🔧 CopyMechanism bias reset: {old_bias:.4f} → {new_bias:.4f} "
                  f"(sigmoid: {1/(1+math.exp(-old_bias)):.2f} → {1/(1+math.exp(-new_bias)):.2f})")
            # Also reset the w_gen WEIGHT to small values so it starts "fresh"
            # This prevents the old trained weights from immediately pulling bias back down
            nn.init.xavier_uniform_(model.copy_mechanism.w_gen.weight)
            print(f"  🔧 CopyMechanism w_gen.weight re-initialized (Xavier)")

        
    else:
        print("\nInitializing model from scratch (Xavier)...")
        from model import initialize_weights
        initialize_weights(model, n_layers=config['num_layers'])
        
    print(f"  Model parameter norm: {torch.norm(next(model.parameters())).item():.4f}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # (Removed destructive reinit_collapsed_heads call that was wiping decoder weights)
    
    # Optimizer with layer-wise LR (decoder gets lower LR to prevent collapse)
    decoder_lr_scale = config.get('decoder_lr_scale', 0.33)
    print(f"\n📊 Layer-wise Learning Rates (decoder scale: {decoder_lr_scale}):")
    param_groups = get_layerwise_param_groups(model, config['lr'], decoder_lr_scale)

    # CRITICAL: We NO LONGER set group['lr'] = 0.0 here.
    # We keep the base LR so the LambdaLR scheduler can multiply it by 0.0/1.0.
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['lr'],
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    accumulation_steps = config['gradient_accumulation']
    import math
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * config['num_epochs']
    warmup_steps = config['warmup_steps']
    staged_unfreeze = config.get('staged_unfreeze', False)
    freeze_steps = config.get('freeze_steps', 0)

    # Multi-lambda scheduler to handle staged unfreeze correctly.
    # LambdaLR multiplies the *initial* LR of each group.
    # We must start with full LR in optimizer and use 0.0 factor in lambda.
    def get_lr_lambda(group_type):
        def _lambda(step):
            # 1. Staged Unfreeze for Encoder
            if group_type == 'encoder' and staged_unfreeze and step < freeze_steps:
                return 0.0
            
            # 2. Standard Warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            
            # 3. Cosine Annealing
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(0.2, cosine_decay)
        return _lambda

    lambdas = [get_lr_lambda(pg.get('group_type')) for pg in param_groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    
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
    # Cockpit instrumentation (optional)
    cockpit = None
    if config.get('enable_cockpit', False):
        try:
            from cockpit_integration import Cockpit
            cockpit = Cockpit(out_dir=config.get('cockpit_out', 'diagnostics_output/cockpit'), writer=writer)
            # Allow Cockpit to wrap optimizer (if it provides that feature)
            try:
                optimizer = cockpit.attach(model, optimizer=optimizer, attn_store=None)
            except Exception:
                # attach may require optimizer to be present earlier; ignore if not supported
                pass
            print('[Cockpit] attached')
        except Exception as e:
            print(f'[Cockpit] failed to initialize: {e}')
    
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
    
    # Initialize rouge_scores to prevent UnboundLocalError at epoch end
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_loss = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        optimizer.zero_grad()
        
        # Phase 19.2: Scheduled Detox (Epoch-based) - ALIGNED FOR 4-EPOCH RUN
        # Lead Mask Prob: 0.90 -> 0.60 -> 0.30 -> 0.15 (Transitioning from Detox to Fluency)
        # Entropy Reg: 0.002 -> 0.0017 -> 0.0015 -> 0.0010 (Reducing diversity pressure)
        if epoch == 0:
            current_lead_mask_prob = 0.90
            entropy_reg_weight = 0.0020
        elif epoch == 1:
            current_lead_mask_prob = 0.60
            entropy_reg_weight = 0.0017
        elif epoch == 2:
            current_lead_mask_prob = 0.30
            entropy_reg_weight = 0.0015
        else: # Epoch 4
            current_lead_mask_prob = 0.15
            entropy_reg_weight = 0.0010
            
        if hasattr(train_loader.dataset, 'lead_mask_prob'):
            train_loader.dataset.lead_mask_prob = current_lead_mask_prob
        
        print(f"\n🚀 EPOCH {epoch+1} CONFIG: lead_mask_prob={current_lead_mask_prob:.2f}, entropy_reg={entropy_reg_weight:.4f}")
        
        for batch_idx, batch in enumerate(progress):

            enc_input = batch['encoder_input'].to(device)
            dec_input = batch['decoder_input'].to(device)
            enc_mask = batch['encoder_mask'].to(device)
            dec_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # GENERATOR WARMUP: disable copy for first N steps so model learns generation
            use_copy_now = use_copy_initially
            if global_step < copy_warmup_steps:
                use_copy_now = False  # Disable copy during warmup
            else:
                use_copy_now = use_copy_initially
            
            # Forward pass with diagnostic hooks
            amp_enabled = config['use_amp']
            with autocast(enabled=amp_enabled):
                if use_copy_now:
                    # HARD POINTER DROPOUT (Nuclear Fix): Force p_gen = 1.0 for ~5% of batches
                    # This breaks the copying plateau by occasionally forcing pure generation
                    hard_dropout_prob = config.get('hard_pointer_dropout_prob', 0.05)
                    force_all_gen = False
                    if global_step >= copy_warmup_steps and torch.rand(1).item() < hard_dropout_prob:
                        force_all_gen = True  # Force pure generation (p_gen = 1.0) for this batch
                    
                    # Pointer-dropout: with some probability, disable the pointer for the whole example
                    pointer_dropout = config.get('pointer_dropout', 0.0)
                    force_no_pointer = False
                    if pointer_dropout > 0 and model.training:
                        # apply same decision across the batch (paper: decision holds for all words in summary)
                        if torch.rand(1, device=enc_input.device).item() < pointer_dropout:
                            force_no_pointer = True

                    final_dist, p_gen, live_cross_attn = model.forward_with_copy(
                        enc_input, enc_mask, dec_input, dec_mask,
                        force_no_pointer=force_no_pointer,
                        force_all_gen=force_all_gen  # Hard Pointer Dropout: forces p_gen=1.0 INSIDE the model
                    )
                    # NOTE: force_all_gen is now handled INSIDE the model, so p_gen and final_dist
                    # are already correct. No post-hoc override needed.
                    
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
                    p_gen = None
                    live_cross_attn = None
                    enc_out = model.encode(enc_input, enc_mask)
                    dec_out = model.decode(enc_out, enc_mask, dec_input, dec_mask)
                    logits = model.project(dec_out)
                    # CRITICAL: During warmup, loss_fn is NLLLoss (expects log-probs),
                    # but logits are RAW. Use F.cross_entropy which applies log_softmax internally.
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size), label.view(-1),
                        ignore_index=tokenizer.pad_id,
                        label_smoothing=0.1  # Fix 5: Apply label smoothing globally
                    )
                
                # ── Phase 8: Weighted Hybrid Loss (Refined) ──
                # 1. Base Loss (Factual Accuracy via smooth_nll_loss) is already calculated above as 'loss'
                # Smooth log-loss already handles sum()/mask.sum() averaging
                base_loss = loss 
                
                # 2. Attention Sharpening (Target 1.6 - "The Drill Sergeant")
                # Apply entropy regularization over multiple decoder cross-attention
                ent_reg_loss = 0.0
                lambda_attn = config.get('lambda_attn', 0.02)
                if lambda_attn > 0:
                    import math
                    # Use unpadded source length for true entropy max
                    src_len = enc_mask.sum(dim=-1).float()  # [B]
                    target_ent = (0.3 * torch.log(src_len + 1e-9)).mean().item()
                    ent_accum = 0.0
                    ent_count = 0

                    # Last two decoder cross-attention layers (strongest signal)
                    for layer in model.decoder.layers[-2:]:
                        ca = getattr(layer, 'cross_attention_block', None)
                        if ca is not None:
                            attn = getattr(ca, 'attention_scores', None)
                            if attn is not None:
                                ent_accum += entropy_regularization(attn, target_entropy=target_ent)
                                ent_count += 1

                    # Also include last decoder self-attention heads (helps focus generation)
                    for layer in model.decoder.layers[-2:]:
                        sa = getattr(layer, 'self_attention_block', None)
                        if sa is not None:
                            attn = getattr(sa, 'attention_scores', None)
                            if attn is not None:
                                ent_accum += entropy_regularization(attn, target_entropy=target_ent)
                                ent_count += 1

                    if ent_count > 0:
                        ent_reg_loss = ent_accum / float(ent_count)

                # 3. Pointer Control
                pgen_loss = 0.0
                apply_ptr_after = config.get('apply_pointer_loss_after_steps', 0)
                if use_copy_now and p_gen is not None and global_step >= (copy_warmup_steps + apply_ptr_after):
                    lambda_p = 0.2  # Fix 4: Adjusted p_gen regularizer scale
                    pgen_loss = lambda_p * pgen_balance_loss(p_gen, target=0.5)

                # 4. Coverage Loss (prevents repetition) — uses LIVE cross_attn
                cov_loss = 0.0
                apply_cov_after = config.get('apply_coverage_after_steps', 0)
                lambda_cov = coverage_loss_weight
                if use_copy_now and lambda_cov > 0 and global_step >= (copy_warmup_steps + apply_cov_after):
                    if live_cross_attn is not None:
                        # Coverage is already normalized per step inside compute_coverage_loss
                        cov_loss = compute_coverage_loss(live_cross_attn)

                # 5. Combined Final Loss
                loss = base_loss
                if lambda_attn > 0 and ent_reg_loss is not None:
                    loss = loss + (lambda_attn * ent_reg_loss)
                    writer.add_scalar('finetune/ent_reg_loss', float(ent_reg_loss.item() if hasattr(ent_reg_loss, 'item') else ent_reg_loss), global_step)

                if use_copy_now and p_gen is not None and pgen_loss is not None:
                    loss = loss + pgen_loss
                    writer.add_scalar('finetune/pgen_loss', float(pgen_loss.item() if hasattr(pgen_loss, 'item') else pgen_loss), global_step)

                loss = loss + (lambda_cov * cov_loss)

                # ── DIAGNOSTIC: Log per-component loss every 50 steps ──
                if global_step % 50 == 0:
                    _bl = float(base_loss.item()) if hasattr(base_loss, 'item') else float(base_loss)
                    _pl = float(pgen_loss.item()) if hasattr(pgen_loss, 'item') else float(pgen_loss)
                    _cl = float(cov_loss.item()) if hasattr(cov_loss, 'item') else float(cov_loss)
                    _el = float(ent_reg_loss.item()) if hasattr(ent_reg_loss, 'item') else float(ent_reg_loss)
                    _pg = float(p_gen.mean().item()) if p_gen is not None else 0.0
                    tqdm.write(f"  [Step {global_step}] base_nll={_bl:.2f}  pgen_loss={_pl:.2f}  "
                               f"cov_loss={lambda_cov * _cl:.2f}  ent_loss={lambda_attn * _el:.4f}  "
                               f"TOTAL={float(loss.item()):.2f}  p_gen_mean={_pg:.3f}")

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
                global_step += 1
                
                # Log global gradient health BEFORE zeroing
                grad_report = check_gradient_health(model, writer, global_step)
                
                # Print warnings only on diagnostic steps to avoid terminal spam
                if global_step % config.get('diagnostic_every', 100) == 0:
                    if grad_report['warnings']:
                        for w in grad_report['warnings'][:3]:
                            tqdm.write(f"  {w}")
                
                writer.add_scalar('finetune/loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('finetune/lr', scheduler.get_last_lr()[0], global_step)
                
                optimizer.zero_grad()

                # Staged encoder unfreeze: ramp encoder LR from 0 -> base_lr over freeze_steps
                if staged_unfreeze and freeze_steps > 0:
                    # compute scale in [0,1]
                    scale = min(1.0, float(global_step) / float(max(1, freeze_steps)))
                    for pg in optimizer.param_groups:
                        if pg.get('group_type') == 'encoder':
                            pg['lr'] = config['lr'] * scale
                
                # ── Periodic diagnostics & Health ──
                if global_step % diagnostic_every == 0:
                    # Log the "Health" of primary cross-attention (Sharpness)
                    last_layer_attn = model.decoder.layers[-1].cross_attention_block.attention_scores
                    if last_layer_attn is not None:
                        # Mean entropy across top heads
                        head_ent = -(last_layer_attn * (last_layer_attn + 1e-12).log()).sum(-1).mean(dim=[0, 2])
                        avg_ent = head_ent.mean().item()
                        writer.add_scalar("Diagnostics/Attention_Entropy", avg_ent, global_step)
                    
                    # Log how much the model is "cheating" by copying
                    if p_gen is not None:
                        avg_pgen = p_gen.mean().item()
                        writer.add_scalar("Diagnostics/P_Gen_Mean", avg_pgen, global_step)

                # (Removed Head-Shock logic as it was destructively erasing decoder linear layers)
                    
                    # Log p_gen if using copy
                    if use_copy and p_gen is not None:
                        log_pgen(writer, p_gen, global_step)
                    
                    # (Gradients are checked and warnings printed at every update above)
                
                # ── Intra-epoch Validation ──
                if global_step % config['save_every'] == 0:
                    # Validation with beam search + ROUGE
                    rouge_scores = run_validation(
                        model, val_loader, tokenizer, config, device,
                        num_examples=config['num_validation_examples'], num_print=5
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
                        size = save_checkpoint(model, optimizer, scheduler, global_step, loss.item() * accumulation_steps, best_rouge, best_path)
                        tqdm.write(f"  ⭐ New best ROUGE-1: {best_rouge:.4f} (Saved to {best_path}, {size:.1f}MB)")
                    else:
                        patience_counter += 1
                        if patience_counter >= config['patience']:
                            tqdm.write("  ⚠️ Early stopping triggered!")
                            return
                    
                    # Periodic checkpoint
                    ckpt_path = Path(config['model_folder']) / f"{config['model_basename']}step_{global_step}.pt"
                    save_checkpoint(model, optimizer, scheduler, global_step, loss.item() * accumulation_steps, current_rouge, ckpt_path)
            
            progress.set_postfix({'loss': f'{loss.item() * accumulation_steps:.3f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Force validation at end of epoch if we haven't done it recently
        if global_step % config['save_every'] != 0:
            print("\nRunning End-of-Epoch Validation...")
            rouge_scores = run_validation(
                model, val_loader, tokenizer, config, device,
                num_examples=config['num_validation_examples'], num_print=5
            )
            
            writer.add_scalar('finetune/rouge1', rouge_scores['rouge1'], global_step)
            writer.add_scalar('finetune/rouge2', rouge_scores['rouge2'], global_step)
            writer.add_scalar('finetune/rougeL', rouge_scores['rougeL'], global_step)
            
            current_rouge = rouge_scores['rouge1']
            if current_rouge > best_rouge:
                best_rouge = current_rouge
                patience_counter = 0
                best_path = Path(config['model_folder']) / f"{config['model_basename']}best.pt"
                size = save_checkpoint(model, optimizer, scheduler, global_step, loss.item() * accumulation_steps, best_rouge, best_path)
                print(f"  ⭐ New best ROUGE-1: {best_rouge:.4f} (Saved to {best_path}, {size:.1f}MB)")
            else:
                patience_counter += 1
            
        # ── Phase 16: Mandatory Epoch Checkpoint ──
        epoch_path = Path(config['model_folder']) / f"{config['model_basename']}epoch_{epoch+1}.pt"
        size = save_checkpoint(model, optimizer, scheduler, global_step, loss.item() * accumulation_steps, rouge_scores['rouge1'], epoch_path)
        print(f"  💾 Epoch {epoch+1} Checkpoint Saved: {epoch_path} ({size:.1f}MB)")

        # End of epoch summary
        current_rouge = rouge_scores['rouge1']
        tqdm.write(f"\nEpoch {epoch+1} Complete - Current ROUGE-1: {current_rouge:.4f} (Best: {best_rouge:.4f})")
        
        # Early stopping check
        if patience_counter >= config['patience']:
            print("\n⚠ Early stopping triggered!")
            break
        
        # Clear any accumulated patience if we had a breakthrough this epoch
        if current_rouge > best_rouge:
            patience_counter = 0
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"\nBest ROUGE-1: {best_rouge:.4f}")
    print(f"Models saved in: {config['model_folder']}/")
    
    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    finetune()