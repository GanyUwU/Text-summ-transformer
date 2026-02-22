"""
Professional Pretraining Script (Multi-Dataset: Wiki + BookCorpus)

- BART-style denoising on mixed datasets
- Resumes from rebooted weights (Phase 1.5)
- Learned narrative context and dialogue
"""
# file: pretrain_multi.py (edited)
# Multi-dataset pretraining (Wikipedia + BookCorpus) with diagnostics & fixes

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import random
import warnings
from tqdm import tqdm
from pathlib import Path

from datasets import load_dataset
from model import build_transformer
from pretrain_config import get_multi_dataset_config, get_pretrain_weights_path
from tokenizer_utils import get_tokenizer
from diagnostics import (
    get_layerwise_param_groups, reinit_collapsed_heads,
    entropy_regularization, log_attention_entropy,
    check_gradient_health, print_diagnostic_summary
)
from cockpit_integration import Cockpit


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
        if len(tokens) < 5:
            return tokens.copy()

        tokens = list(tokens)
        num_to_mask = max(1, int(len(tokens) * self.mask_prob))

        mask_count = 0
        i = 0
        while mask_count < num_to_mask and i < len(tokens):
            span_len = min(
                max(1, int(np.random.poisson(self.span_lambda) + 1)),
                len(tokens) - i,
                num_to_mask - mask_count
            )
            if random.random() < 0.5:
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

        label_ids = original_tokens + [self.eos_id]
        label_ids = label_ids[:self.seq_len]
        label_ids += [self.pad_id] * (self.seq_len - len(label_ids))

        return {
            "encoder_input": torch.tensor(enc_input, dtype=torch.long),
            "decoder_input": torch.tensor(dec_input, dtype=torch.long),
            "label": torch.tensor(label_ids, dtype=torch.long)
        }


def _concatenate_bookcorpus_samples(it_b, min_length=2000, max_length=3000):
    """
    BookCorpus stores individual sentences (~50-200 chars each).
    Concatenate multiple samples until reaching min_length.
    This gives us proper passages for training.
    """
    passage = ""
    while len(passage) < min_length:
        try:
            b = next(it_b)
            text = b.get('text', '').strip()
            if text:
                if passage:
                    passage += " " + text
                else:
                    passage = text
        except StopIteration:
            break
    return passage[:max_length] if passage else ""


def get_mixed_texts(config):
    """
    Load and interleave three diverse datasets:
    - Wikipedia (encyclopedic, factual) - full articles (~5-50k chars)
    - BookCorpus (literary, narrative) - concatenated to ~2k chars passages
    - OpenWebText (web content, diverse topics) - full articles (~2-15k chars)
    
    Streaming-friendly, no full RAM loading required.
    """
    print("\n" + "="*60)
    print("Loading MIXED DATASETS for diverse pretraining")
    print("="*60)
    
    # Load datasets
    print("📚 Loading Wikipedia (encyclopedic)...")
    wiki = load_dataset("wikimedia/wikipedia", config.get('dataset_config', "20231101.en"),
                        split="train", streaming=True)
    
    print("📖 Loading BookCorpus (literary) - will concatenate sentences...")
    try:
        books = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
    except Exception:
        print("  ! rojagtap/bookcorpus unavailable, trying allennlp version...")
        books = load_dataset("allennlp/bookcorpus", split="train", streaming=True)
    
    print("🌐 Loading OpenWebText (web, diverse)...")
    try:
        web = load_dataset("openwebtext", split="train", streaming=True)
    except Exception:
        print("  ! OpenWebText unavailable, will use Wikipedia + BookCorpus only")
        web = None

    max_count = config.get('max_articles', 500000)
    debug_cap = config.get('debug_samples', None)

    if debug_cap is not None:
        max_count = min(max_count, debug_cap)
        print(f"[DEBUG] Capping dataset to {max_count} samples for fast iteration")
    texts = []
    
    # Create iterators
    it_w = iter(wiki)
    it_b = iter(books)
    it_web = iter(web) if web else None
    
    # Sampling strategy: round-robin through sources
    # Wikipedia (0.4), BookCorpus (0.3), OpenWebText (0.3)
    i = 0
    sample_idx = 0
    
    if web:
        print(f"\n📊 Balanced sampling: Wikipedia 40% + BookCorpus 30% + OpenWebText 30%")
        print(f"Sampling up to {max_count} examples from 3 sources...\n")
        
        while i < max_count:
            source = sample_idx % 10  # 10 samples = 4 wiki + 3 book + 3 web
            
            try:
                if source < 4:  # 40% Wikipedia
                    w = next(it_w)
                    text = w.get('text', '')[:3000]
                    if text:
                        texts.append(text)
                elif source < 7:  # 30% BookCorpus (concatenated)
                    text = _concatenate_bookcorpus_samples(it_b, min_length=2000, max_length=3000)
                    if text:
                        texts.append(text)
                else:  # 30% OpenWebText
                    entry = next(it_web)
                    text = entry.get('text', '')[:3000]
                    if text:
                        texts.append(text)
                    
            except StopIteration:
                # Restart exhausted source
                if source < 4:
                    it_w = iter(load_dataset("wikimedia/wikipedia", 
                                            config.get('dataset_config', "20231101.en"),
                                            split="train", streaming=True))
                    continue
                elif source < 7:
                    try:
                        it_b = iter(load_dataset("rojagtap/bookcorpus", 
                                                split="train", streaming=True))
                    except:
                        it_b = iter(load_dataset("allennlp/bookcorpus", 
                                                split="train", streaming=True))
                    continue
                else:
                    it_web = iter(load_dataset("openwebtext", split="train", streaming=True))
                    continue
            
            sample_idx += 1
            i += 1
            if i % 10000 == 0:
                print(f"  ✓ Collected {len(texts)} samples...")
    
    else:
        # Fallback: Wikipedia + BookCorpus only (50-50)
        print(f"\n📊 Balanced sampling: Wikipedia 50% + BookCorpus 50%")
        print(f"Sampling up to {max_count} examples from 2 sources...\n")
        
        while i < max_count:
            try:
                if i % 2 == 0:
                    w = next(it_w)
                    text = w.get('text', '')[:3000]
                    if text:
                        texts.append(text)
                else:
                    text = _concatenate_bookcorpus_samples(it_b, min_length=2000, max_length=3000)
                    if text:
                        texts.append(text)
            except StopIteration:
                if i % 2 == 0:
                    it_w = iter(load_dataset("wikimedia/wikipedia",
                                            config.get('dataset_config', "20231101.en"),
                                            split="train", streaming=True))
                else:
                    try:
                        it_b = iter(load_dataset("rojagtap/bookcorpus",
                                                split="train", streaming=True))
                    except:
                        it_b = iter(load_dataset("allennlp/bookcorpus",
                                                split="train", streaming=True))
                continue
            
            i += 1
            if i % 10000 == 0:
                print(f"  ✓ Collected {len(texts)} samples...")
    
    print(f"\n✓ Dataset loading complete! Total samples: {len(texts)}")
    print(f"  Shuffling {len(texts)} samples...")
    random.shuffle(texts)
    print(f"  Ready for training!\n")
    return texts


# ---- Mask helper (canonical masks) ----
def make_masks_batch(encoder_input, decoder_input, pad_id, device):
    """
    Returns:
      encoder_mask: (B, 1, 1, src_len)  bool
      decoder_mask: (B, 1, tgt_len, tgt_len) bool (causal & padding)
    """
    B, S = encoder_input.size()
    _, T = decoder_input.size()

    encoder_padding = (encoder_input != pad_id).to(torch.bool)             # (B,S)
    encoder_mask = encoder_padding.unsqueeze(1).unsqueeze(1)               # (B,1,1,S)

    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device)).unsqueeze(0)  # (1,T,T)
    decoder_padding = (decoder_input != pad_id).to(torch.bool).unsqueeze(1).unsqueeze(-1)  # (B,1,T,1)
    decoder_mask = causal.unsqueeze(0) & decoder_padding                     # (B,1,T,T)

    return encoder_mask, decoder_mask


# ---- Attention hooks (capture attn weights if not stored) ----
def register_attention_hooks(model, attn_store):
    """
    Registers forward hooks to capture attention weights.
    attn_store will contain module_id -> attn_tensor (B,H,Tq,Tk)
    """
    def hook(module, inp, out):
        # out may be (output, attn_weights) for some modules
        try:
            if isinstance(out, tuple) and len(out) >= 2:
                attn = out[1]
                if attn is not None:
                    attn_store[id(module)] = attn.detach().cpu()
        except Exception:
            # best-effort; don't crash training
            pass

    for m in model.modules():
        tname = m.__class__.__name__.lower()
        if 'multihead' in tname or 'attention' in tname or 'selfattention' in tname:
            try:
                m.register_forward_hook(hook)
            except Exception:
                # some modules don't accept hook registration; skip
                pass


def pretrain_multi():
    config = get_multi_dataset_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Logging
    writer = SummaryWriter(config['experiment_name'])

    # Tokenizer
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()

    # Data
    texts = get_mixed_texts(config)
    dataset = MixedDenoisingDataset(texts, tokenizer, config['seq_len'], config)
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

    # Register attention hooks to capture attn weights if model doesn't store them
    attention_store = {}
    register_attention_hooks(model, attention_store)

    # Cockpit instrumentation (optional)
    cockpit = None
    if config.get('enable_cockpit', False):
        cockpit = Cockpit(out_dir=config.get('cockpit_out', 'diagnostics_output/cockpit'), writer=writer)

    # Layer-wise LR param groups (expect function to return groups with 'lr' keys)
    decoder_lr_scale = config.get('decoder_lr_scale', 0.33)
    print(f"\n📊 Multi-Dataset Pretrain (decoder_lr_scale={decoder_lr_scale})")
    param_groups = get_layerwise_param_groups(model, base_lr=config['lr'], decoder_lr_scale=decoder_lr_scale)
    # NOTE: get_layerwise_param_groups should return groups with 'lr' set.
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['weight_decay'])

    if cockpit:
        optimizer = cockpit.attach(model, optimizer=optimizer, attn_store=attention_store)

    # --- LR scheduler (linear warmup) ---
    warmup_steps = config.get("warmup_steps", 2000)
    base_lr = config.get('lr', 1e-4)

    def lr_lambda(step):
        # linear warmup from 0 -> 1 across warmup_steps
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    accumulation_steps = config['gradient_accumulation']
    scaler = GradScaler() if config.get('use_amp', False) else None
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=config.get('label_smoothing', 0.0))

    global_step = 0
    best_loss = float('inf')
    resume_path = config.get('resume_from', None)

    # Resume safely if provided
    if resume_path and Path(resume_path).exists():
        print(f"[RESUME] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        try:
            model.load_state_dict(state_dict, strict=False)
            print("  ✓ Model state loaded (strict=False)")
        except Exception as e:
            print("  ! Failed strict load, try full assignment:", e)
            model.load_state_dict(state_dict, strict=False)

        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("  ✓ Optimizer state resumed")
            except Exception as e:
                print("  ! Could not resume optimizer state:", e)

        if 'scheduler_state_dict' in ckpt:
            try:
                # scheduler may require exact optimizer state; try best-effort
                print("  ✓ Scheduler state present (skipping strict restore if incompatible)")
            except Exception:
                pass

        global_step = ckpt.get('step', global_step)
        best_loss = ckpt.get('best_loss', best_loss)
        print(f"[RESUME] at step {global_step}, best_loss={best_loss:.4f}")

        if config.get('reinit_decoder_heads', False):
            only_decoder = config.get('reinit_only_decoder', True)
            reinit_collapsed_heads(model, only_decoder=only_decoder)
            print("[RESUME] Reinitialized collapsed heads as requested")

    model.train()
    progress = tqdm(range(config['num_steps']), desc="Multi-Pretrain")

    dataset_iter = iter(dataloader)
    first_masks_printed = False

    for step in progress:
        try:
            batch = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataloader)
            batch = next(dataset_iter)

        enc_input = batch['encoder_input'].to(device)
        dec_input = batch['decoder_input'].to(device)
        labels = batch['label'].to(device)

        # Create canonical masks using helper (B,1,1,S) and (B,1,T,T)
        enc_mask, dec_mask = make_masks_batch(enc_input, dec_input, tokenizer.pad_id, device)

        # Sanity print once
        if not first_masks_printed:
            tqdm.write(f"DEBUG: enc_mask shape={enc_mask.shape}, dec_mask shape={dec_mask.shape}")
            first_masks_printed = True

        with autocast(enabled=config.get('use_amp', False)):
            enc_out = model.encode(enc_input, enc_mask)
            dec_out = model.decode(enc_out, enc_mask, dec_input, dec_mask)
            logits = model.project(dec_out)

            loss = loss_fn(logits.view(-1, vocab_size), labels.reshape(-1))

            # ── Auxiliary Losses ──
            entropy_reg_weight = config.get('entropy_reg_weight', 0.0)
            if entropy_reg_weight > 0:
                ent_accum = 0.0
                ent_count = 0
                target_ent = config.get('target_entropy', 2.0)
                
                # Check Decoder (Target inner layers which often collapse)
                for layer in model.decoder.layers[1:3]:
                    attn = layer.self_attention_block.attention_scores
                    if attn is not None:
                        ent_accum += entropy_regularization(attn, target_entropy=target_ent)
                        ent_count += 1
                        
                # Check Encoder (to fix the 40/48 encoder collapse)
                for layer in model.encoder.layers[1:3]:
                    attn = layer.self_attention_block.attention_scores
                    if attn is not None:
                        ent_accum += entropy_regularization(attn, target_entropy=target_ent)
                        ent_count += 1
                        
                if ent_count > 0:
                    loss = loss + (entropy_reg_weight * (ent_accum / max(1, ent_count)))

            loss = loss / accumulation_steps

        # Backprop (AMP-aware)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        # Optimizer step (after gradient accumulation)
        if (step + 1) % accumulation_steps == 0:

            if cockpit:
                try:
                    cockpit.before_step(global_step)
                except Exception as e:
                    tqdm.write(f"[Cockpit] before_step failed: {e}")

            if scaler:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            global_step += 1

            # 🔁 Step LR scheduler (after optimizer step)
            try:
                scheduler.step()
            except Exception:
                pass

            loss_val = float(loss.item() * accumulation_steps)

            if cockpit:
                try:
                    cockpit.after_step(global_step, loss_val)
                except Exception as e:
                    tqdm.write(f"[Cockpit] after_step failed: {e}")

            # 📈 Logging: loss + LR (all param groups) + warmup fraction
            writer.add_scalar('multi_pretrain/loss', loss_val, global_step)

            for i, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'multi_pretrain/lr_group_{i}', float(pg.get('lr', config['lr'])), global_step)

            # warmup progress (0 → 1)
            warmup_frac = min(1.0, float(global_step) / float(max(1, warmup_steps)))
            writer.add_scalar('multi_pretrain/warmup_frac', warmup_frac, global_step)

            # Diagnostics: log attention entropy using captured attn tensors
            if global_step % config.get('log_every', 100) == 0:
                # Try encoder self-attention enropy logging
                for li, layer in enumerate(model.encoder.layers):
                    attn = getattr(layer.self_attention_block, 'attention_scores', None)
                    if attn is None:
                        # check hooks storage
                        for m in layer.modules():
                            if id(m) in attention_store:
                                attn = attention_store[id(m)].to(device)
                                break
                    if attn is not None:
                        try:
                            log_attention_entropy(writer, attn, f"enc_self_L{li}", global_step)
                        except Exception:
                            pass

                # Decoder logging
                for li, layer in enumerate(model.decoder.layers):
                    attn = getattr(layer.self_attention_block, 'attention_scores', None)
                    if attn is None:
                        for m in layer.modules():
                            if id(m) in attention_store:
                                attn = attention_store[id(m)].to(device)
                                break
                    if attn is not None:
                        try:
                            log_attention_entropy(writer, attn, f"dec_self_L{li}", global_step)
                        except Exception:
                            pass
                    # cross-attention
                    ca = getattr(layer.cross_attention_block, 'attention_scores', None)
                    if ca is None:
                        for m in layer.modules():
                            if id(m) in attention_store:
                                ca = attention_store[id(m)].to(device)
                                break
                    if ca is not None:
                        try:
                            log_attention_entropy(writer, ca, f"dec_cross_L{li}", global_step)
                        except Exception:
                            pass

                # Gradient health
                report = check_gradient_health(model, writer, global_step)
                if report and report.get('warnings'):
                    for w in report['warnings'][:3]:
                        tqdm.write(f"[GRAD_WARN] {w}")

            # Save & checkpoints
            if global_step % config.get('save_every', 500) == 0:
                ckpt_path = get_pretrain_weights_path(config, f"step_{global_step}")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, ckpt_path)
                tqdm.write(f"[OK] Checkpoint saved: {ckpt_path}")

            if loss_val < best_loss:
                best_loss = loss_val
                best_path = get_pretrain_weights_path(config, "best")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'best_loss': best_loss,
                }, best_path)
                tqdm.write(f"[OK] New best model saved: {best_path}")

            progress.set_postfix({'loss': f'{loss_val:.3f}'})

    # Finalization
    print_diagnostic_summary(model)
    writer.close()
    print("Multi-Dataset Pretraining complete.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pretrain_multi()
