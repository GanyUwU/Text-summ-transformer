"""
Professional Pretraining Script (FIXED + CHECKPOINTS)

- BART-style denoising on Wikipedia
- AMP + gradient accumulation
- Correct LR scheduling (per optimizer update)
- Correct attention mask shapes
- Periodic checkpoints + best + final
- Resume training from checkpoint
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
import warnings
from tqdm import tqdm
from pathlib import Path
from collections import deque

from datasets import load_dataset
from model import build_transformer
from pretrain_config import get_pretrain_config, get_pretrain_weights_path
from tokenizer_utils import get_tokenizer, train_tokenizer_on_wikipedia
from diagnostics import (
    get_layerwise_param_groups, reinit_collapsed_heads,
    entropy_regularization, log_attention_entropy,
    check_gradient_health, print_diagnostic_summary
)


class DenoisingDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len, config):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.config = config

        self.mask_prob = config.get('mask_prob', 0.30)
        self.span_lambda = config.get('mask_span_lambda', 3.0)
        self.shuffle_sentences = config.get('shuffle_sentences', True)

        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.mask_id = tokenizer.mask_id

    def __len__(self):
        return len(self.texts)

    def _causal_mask(self, size):
        # Returns (1, T, T) so batched it becomes (B, 1, T, T)
        return torch.tril(torch.ones((1, size, size), dtype=torch.bool))


    def mask_spans(self, tokens):
        """
        Mask spans of tokens (BART-style) for denoising objective.
        
        Actually masks mask_prob fraction of tokens.
        E.g., mask_prob=0.30 -> mask 30% of input tokens.
        """
        if len(tokens) < 5:
            return tokens.copy()
        
        tokens = list(tokens)
        num_to_mask = max(1, int(len(tokens) * self.mask_prob))
        
        # Randomly select which positions to start masks at
        # (not which tokens to mask, but which positions start spans)
        mask_start_positions = []
        mask_count = 0
        i = 0
        
        while mask_count < num_to_mask and i < len(tokens):
            # Decide span length
            span_len = min(
                max(1, int(np.random.poisson(self.span_lambda) + 1)),
                len(tokens) - i,
                num_to_mask - mask_count
            )
            
            # If random chance says mask this position (independent of num_to_mask)
            if random.random() < 0.5:  # 50% chance to mask at each decision point
                for j in range(span_len):
                    if mask_count < num_to_mask:
                        tokens[i + j] = self.mask_id
                        mask_count += 1
                i += span_len
            else:
                i += 1
        
        return tokens

    def shuffle_sentence_order(self, text):
        if not self.shuffle_sentences:
            return text
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            random.shuffle(sentences)
        return ' '.join(sentences)

    def __getitem__(self, idx):
        text = self.texts[idx]
        noisy_text = self.shuffle_sentence_order(text)

        original_tokens = self.tokenizer.encode(text)[:self.seq_len - 2]
        noisy_tokens = self.tokenizer.encode(noisy_text)[:self.seq_len - 2]
        masked_tokens = self.mask_spans(noisy_tokens)

        enc_input = [self.bos_id] + masked_tokens + [self.eos_id]
        enc_input = enc_input[:self.seq_len]
        enc_input += [self.pad_id] * (self.seq_len - len(enc_input))

        dec_input = [self.bos_id] + original_tokens
        dec_input = dec_input[:self.seq_len]
        dec_input += [self.pad_id] * (self.seq_len - len(dec_input))

        label = original_tokens + [self.eos_id]
        label = label[:self.seq_len]
        label += [self.pad_id] * (self.seq_len - len(label))

        encoder_input = torch.tensor(enc_input, dtype=torch.long)
        decoder_input = torch.tensor(dec_input, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        encoder_padding = (encoder_input != self.pad_id)
        encoder_mask = encoder_padding.unsqueeze(0).unsqueeze(0)
        
        causal = self._causal_mask(self.seq_len)  # (1, T, T)
        decoder_padding = (decoder_input != self.pad_id)  # (T,)
        # Need: (1, T, 1) for broadcasting with (1, T, T)
        decoder_padding_mask = decoder_padding.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
        decoder_mask = causal & decoder_padding_mask  # (1, T, T)

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label,
        }


def get_lr_scheduler(optimizer, warmup_steps, total_update_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_update_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain():
    config = get_pretrain_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    writer = SummaryWriter(config['experiment_name'])

    # Tokenizer
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()

    # Load Wikipedia
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    texts = []
    for i, article in enumerate(wiki):
        if i >= config.get('max_articles', 500000):
            break
        if 'text' in article and len(article['text']) > 100:
            texts.append(article['text'][:20000])

    dataset = DenoisingDataset(texts, tokenizer, config['seq_len'], config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    # Model
    model = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)

    # Layer-wise LR (decoder gets lower LR to prevent collapse)
    decoder_lr_scale = config.get('decoder_lr_scale', 0.33)
    print(f"\n[Layer-wise Learning Rates] (decoder scale: {decoder_lr_scale}):")
    param_groups = get_layerwise_param_groups(model, config['lr'], decoder_lr_scale)
    
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )

    accumulation_steps = config['gradient_accumulation']
    total_update_steps = max(1, config['num_steps'] // accumulation_steps)
    scheduler = get_lr_scheduler(optimizer, config['warmup_steps'], total_update_steps)

    scaler = GradScaler() if config['use_amp'] else None
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Checkpoint setup
    ckpt_dir = Path(config['model_folder'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = config.get('resume_from', None)

    global_step = 0
    best_loss = float('inf')

    # Resume if provided
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        
        # Load model weights (handle both full ckpt and state_dict only)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)
        
        # Optional states
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  [OK] Resumed optimizer state")
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("  [OK] Resumed scheduler state")
            
        global_step = ckpt.get('step', 0)
        best_loss = ckpt.get('best_loss', best_loss)
        print(f"[OK] Resumed at step {global_step}")
        
        # Reinitialize collapsed heads upon resume if requested
        # This is key for the "Reboot" phase to break out of United States loops
        if config.get('reinit_decoder_heads', False):
            # If the user wants to fix the 40/48 encoder collapse, we reinit the whole model
            only_decoder = config.get('reinit_only_decoder', True)
            reinit_collapsed_heads(model, only_decoder=only_decoder)
    else:
        print("\nInitializing model from scratch (Xavier)...")
        from model import initialize_weights
        initialize_weights(model)


    writer = SummaryWriter(config['experiment_name'])

    model.train()
    optimizer.zero_grad()

    data_iter = iter(dataloader)
    progress = tqdm(range(config['num_steps']), desc="Pretraining")

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        enc_input = batch['encoder_input'].to(device)
        dec_input = batch['decoder_input'].to(device)
        enc_mask = batch['encoder_mask'].to(device)
        dec_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)

        with autocast(enabled=config['use_amp']):
            enc_out = model.encode(enc_input, enc_mask)
            dec_out = model.decode(enc_out, enc_mask, dec_input, dec_mask)
            logits = model.project(dec_out)
            
            # Primary Loss
            loss = loss_fn(logits.view(-1, vocab_size), label.view(-1))
            
            # ── Auxiliary Losses ──
            # Entropy regularization (prevent decoder self-attention collapse)
            entropy_reg_weight = config.get('entropy_reg_weight', 0.0)
            if entropy_reg_weight > 0:
                # Target Layer 1 or 2 often collapses first
                # Check Decoder
                layer_to_check_dec = model.decoder.layers[1] 
                self_attn_dec = layer_to_check_dec.self_attention_block.attention_scores
                if self_attn_dec is not None:
                    ent_loss_dec = entropy_regularization(self_attn_dec, target_entropy=2.0)
                    loss = loss + entropy_reg_weight * ent_loss_dec
                
                # Check Encoder (to fix the 40/48 collapse issue)
                layer_to_check_enc = model.encoder.layers[1]
                self_attn_enc = layer_to_check_enc.self_attention_block.attention_scores
                if self_attn_enc is not None:
                    ent_loss_enc = entropy_regularization(self_attn_enc, target_entropy=2.0)
                    loss = loss + entropy_reg_weight * ent_loss_enc
            
            loss = loss / accumulation_steps

        if config['use_amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
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

            loss_val = float(loss.item() * accumulation_steps)
            writer.add_scalar('pretrain/loss', loss_val, global_step)
            writer.add_scalar('pretrain/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # ── Periodic Diagnostics ──
            if global_step % config.get('log_every', 100) == 0:
                # Log attention entropy for encoder layers
                for li, layer in enumerate(model.encoder.layers):
                    sa_attn = layer.self_attention_block.attention_scores
                    if sa_attn is not None:
                        log_attention_entropy(writer, sa_attn, f"enc_self_L{li}", global_step)

                # Log attention entropy for decoder layers
                for li, layer in enumerate(model.decoder.layers):
                    sa_attn = layer.self_attention_block.attention_scores
                    if sa_attn is not None:
                        log_attention_entropy(writer, sa_attn, f"dec_self_L{li}", global_step)
                    ca_attn = layer.cross_attention_block.attention_scores
                    if ca_attn is not None:
                        log_attention_entropy(writer, ca_attn, f"dec_cross_L{li}", global_step)
                
                # Gradient health check
                grad_report = check_gradient_health(model, writer, global_step)
                if grad_report['warnings']:
                    for w in grad_report['warnings'][:3]:
                        tqdm.write(f"  {w}")

            # Save checkpoint
            if global_step % config['save_every'] == 0:
                save_path = get_pretrain_weights_path(config, f"step_{global_step}")
                torch.save({
                    'step': global_step,
                    'best_loss': best_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_val,
                }, save_path)
                tqdm.write(f"[OK] Checkpoint saved: {save_path}")

            # Save best
            if loss_val < best_loss:
                best_loss = loss_val
                best_path = get_pretrain_weights_path(config, "best")
                torch.save({
                    'step': global_step,
                    'best_loss': best_loss,
                    'model_state_dict': model.state_dict(),
                }, best_path)
                tqdm.write(f"[OK] New best model: {best_path} (loss: {best_loss:.4f})")

            progress.set_postfix({'loss': f'{loss_val:.3f}'})

    # Final save
    final_path = get_pretrain_weights_path(config, "final1")
    torch.save({
        'step': global_step,
        'best_loss': best_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, final_path)
    print(f"[OK] Final model saved: {final_path}")
    
    # Final summary
    print_diagnostic_summary(model)

    writer.close()
    print("Pretraining complete.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pretrain()
