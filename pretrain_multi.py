"""
Professional Pretraining Script (Multi-Dataset: Wiki + BookCorpus)

- BART-style denoising on mixed datasets
- Resumes from rebooted weights (Phase 1.5)
- Learned narrative context and dialogue
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

from datasets import load_dataset, interleave_datasets
from model import build_transformer
from pretrain_config import get_multi_dataset_config, get_pretrain_weights_path
from tokenizer_utils import get_tokenizer
from diagnostics import (
    get_layerwise_param_groups, reinit_collapsed_heads,
    entropy_regularization, log_attention_entropy,
    check_gradient_health, print_diagnostic_summary
)


class MixedDenoisingDataset(Dataset):
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
        mask_count = 0
        i = 0
        
        while mask_count < num_to_mask and i < len(tokens):
            # Decide span length
            span_len = min(
                max(1, int(np.random.poisson(self.span_lambda) + 1)),
                len(tokens) - i,
                num_to_mask - mask_count
            )
            
            # If random chance says mask this position
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

        # Prepare tensors
        input_ids = [self.bos_id] + masked_tokens + [self.eos_id]
        label_ids = [self.bos_id] + original_tokens + [self.eos_id]

        # Padding
        input_pad = [self.pad_id] * (self.seq_len - len(input_ids))
        label_pad = [self.pad_id] * (self.seq_len - len(label_ids))

        input_ids += input_pad
        label_ids += label_pad

        return {
            "input": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label_ids, dtype=torch.long)
        }


def get_mixed_texts(config):
    """Load and interleave datasets."""
    print("Loading datasets...")
    
    # Wiki
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    # BookCorpus (using a Parquet-only community version to avoid script execution issues)
    try:
        books = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
    except Exception:
        # Fallback to another reputable Parquet source
        books = load_dataset("allennlp/bookcorpus", split="train", streaming=True)
    
    max_count = config.get('max_articles', 50000)
    texts = []
    
    print(f"Sampling {max_count} from Wikipedia and BookCorpus...")
    
    # Interleave manually for streaming control
    for i, (w, b) in enumerate(zip(wiki, books)):
        if i >= max_count:
            break
        texts.append(w['text'][:3000])
        texts.append(b['text'][:3000])
        
        if i % 10000 == 0:
            print(f"  Collected {len(texts)} samples...")
            
    random.shuffle(texts)
    return texts


def pretrain_multi():
    config = get_multi_dataset_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    writer = SummaryWriter(config['experiment_name'])

    # Tokenizer
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()

    # Data
    texts = get_mixed_texts(config)
    dataset = MixedDenoisingDataset(texts, tokenizer, config['seq_len'], config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

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

    # Layer-wise LR
    decoder_lr_scale = config.get('decoder_lr_scale', 0.5) # slightly higher than reboot
    print(f"\n📊 Multi-Dataset Fine-tuning LR (decoder scale: {decoder_lr_scale}):")
    param_groups = get_layerwise_param_groups(model, config['lr'], decoder_lr_scale)
    
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )

    accumulation_steps = config['gradient_accumulation']
    scaler = GradScaler(enabled=config['use_amp'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=config['label_smoothing'])

    global_step = 0
    best_loss = float('inf')
    resume_path = config.get('resume_from')

    # Resume from Reboot Weights
    if resume_path and Path(resume_path).exists():
        print(f"Resuming from reboot weights: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Reboot weights loaded")

    model.train()
    progress = tqdm(range(config['num_steps']), desc="Multi-Pretrain")

    dataset_iterator = iter(dataloader)

    for step in progress:
        try:
            batch = next(dataset_iterator)
        except StopIteration:
            dataset_iterator = iter(dataloader)
            batch = next(dataset_iterator)

        input_ids = batch['input'].to(device)
        label_ids = batch['label'].to(device)

        # For denoising: encoder input is masked, decoder reconstructs original
        dec_input = label_ids[:, :-1]  # Remove last token for decoder input
        label = label_ids[:, 1:]       # Remove first token for labels
        
        enc_input = input_ids
        
        # Encoder mask: (batch, 1, 1, seq_len) - just padding, no causality
        enc_mask = (enc_input != tokenizer.pad_id).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
        
        # Decoder mask: (batch, 1, seq_len, seq_len) - causal + padding
        batch_size = dec_input.size(0)
        seq_len = dec_input.size(1)
        
        # Create causal mask: (1, 1, T, T)
        causal = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device))
        
        # Create padding mask: (B, 1, 1, T)
        padding = (dec_input != tokenizer.pad_id).unsqueeze(1).unsqueeze(1)
        
        # Combine: causal AND padding
        dec_mask = causal & padding  # (B, 1, T, T)

        with autocast(enabled=config['use_amp']):
            enc_out = model.encode(enc_input, enc_mask)
            dec_out = model.decode(enc_out, enc_mask, dec_input, dec_mask)
            logits = model.project(dec_out)
            
            # Loss
            loss = loss_fn(logits.view(-1, vocab_size), label.reshape(-1))
            
            # Entropy regularization (subtle)
            entropy_reg_weight = config.get('entropy_reg_weight', 0.0)
            if entropy_reg_weight > 0:
                layer_to_check = model.decoder.layers[1] 
                self_attn = layer_to_check.self_attention_block.attention_scores
                if self_attn is not None:
                    ent_loss = entropy_regularization(self_attn, target_entropy=2.5)
                    loss = loss + entropy_reg_weight * ent_loss
            
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            loss_val = float(loss.item() * accumulation_steps)
            writer.add_scalar('multi_pretrain/loss', loss_val, global_step)
            
            # Periodic Diagnostics
            if global_step % config.get('log_every', 100) == 0:
                check_gradient_health(model, writer, global_step)

            # Save
            if global_step % config['save_every'] == 0:
                ckpt_path = get_pretrain_weights_path(config, f"step_{global_step}")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, ckpt_path)

            if loss_val < best_loss:
                best_loss = loss_val
                best_path = get_pretrain_weights_path(config, "best")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss,
                }, best_path)

            progress.set_postfix({'loss': f'{loss_val:.3f}'})

    # Final Summary
    print_diagnostic_summary(model)
    print("Multi-Dataset Pretraining complete.")
    writer.close()


if __name__ == "__main__":
    pretrain_multi()
