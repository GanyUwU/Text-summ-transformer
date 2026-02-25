"""
Detailed Attention Collapse Debugger
====================================

Traces attention computation step-by-step to find WHERE attention collapses.
Shows Q, K, attention scores at each layer to identify the failure point.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from model import build_transformer
from tokenizer_utils import get_tokenizer
from pretrain_config import get_multi_dataset_config


def _num_heads(attn_block):
    if hasattr(attn_block, "num_heads"):
        return int(attn_block.num_heads)
    if hasattr(attn_block, "h"):
        return int(attn_block.h)
    if hasattr(attn_block, "n_heads"):
        return int(attn_block.n_heads)
    raise AttributeError("Attention block has no head-count attribute (num_heads/h/n_heads).")


def get_qkv(attn_block, x):
    """Get projected Q, K, V — including LayerNorm if present (matches model.py)."""
    if hasattr(attn_block, "w_q"):
        q = attn_block.w_q(x)
        k = attn_block.w_k(x)
        v = attn_block.w_v(x)
    elif hasattr(attn_block, "linear_q"):
        q = attn_block.linear_q(x)
        k = attn_block.linear_k(x)
        v = attn_block.linear_v(x)
    elif hasattr(attn_block, "q_proj"):
        q = attn_block.q_proj(x)
        k = attn_block.k_proj(x)
        v = attn_block.v_proj(x)
    else:
        raise AttributeError("No Q/K/V projections found in attention block.")

    # Apply Q/K LayerNorm if present (critical — model.py normalizes Q/K
    # before computing dot products; without this, stats are misleading)
    q_raw_std = q.std().item()
    k_raw_std = k.std().item()
    if hasattr(attn_block, "q_norm"):
        q = attn_block.q_norm(q)
    if hasattr(attn_block, "k_norm"):
        k = attn_block.k_norm(k)

    return q, k, v, q_raw_std, k_raw_std


def to_bhtd(t, x_ref, attn_block, name="tensor"):
    h = _num_heads(attn_block)

    if t.ndim == 3:
        if t.shape[0] == x_ref.shape[0] and t.shape[1] == x_ref.shape[1]:
            btd = t
        elif t.shape[0] == x_ref.shape[1] and t.shape[1] == x_ref.shape[0]:
            btd = t.transpose(0, 1)
        else:
            raise RuntimeError(
                f"{name} shape {tuple(t.shape)} does not align with reference {tuple(x_ref.shape)}"
            )
        B, T, D = btd.shape
        if D % h != 0:
            raise RuntimeError(f"{name} last dim {D} not divisible by num_heads={h}")
        d_k = D // h
        return btd.contiguous().view(B, T, h, d_k).transpose(1, 2).contiguous()

    if t.ndim == 4:
        if t.shape[1] == h:
            return t.contiguous()
        if t.shape[2] == h:
            return t.permute(0, 2, 1, 3).contiguous()
        raise RuntimeError(f"{name} 4D shape {tuple(t.shape)} does not contain head axis={h}")

    raise RuntimeError(f"{name} unsupported rank {t.ndim}; expected 3D/4D")


def debug_attention_flow(model, tokenizer, device, test_text="The quick brown fox jumps"):
    """
    Trace attention computation through encoder layers.
    Shows Q, K, and attention entropy at each step.
    """
    print("\n" + "="*80)
    print("ATTENTION FLOW DEBUGGING")
    print("="*80)
    print(f"Test text: {test_text!r}\n")
    
    # Tokenize
    tokens = tokenizer.encode(test_text)
    input_ids = [tokenizer.bos_id] + tokens + [tokenizer.eos_id]
    enc_input = torch.tensor([input_ids], dtype=torch.long).to(device)
    enc_mask = (enc_input != tokenizer.pad_id).unsqueeze(1).unsqueeze(1)
    
    print(f"Input shape: {enc_input.shape}")
    print(f"Mask shape: {enc_mask.shape}\n")
    
    model.eval()
    with torch.no_grad():
        # Embed
        x = model.src_embed(enc_input)  # (B, T, d_model)
        x = model.src_pos(x)
        
        print(f"Initial embedding shape: {x.shape}")
        print(f"Initial embedding stats: mean={x.mean():.4f}, std={x.std():.4f}\n")
        
        # Process through encoder layers
        for layer_idx, layer in enumerate(model.encoder.layers):
            print(f"{'='*80}")
            print(f"LAYER {layer_idx}")
            print(f"{'='*80}")
            
            sa = layer.self_attention_block
            
            # Compute Q, K, V (with LayerNorm applied — matches model.py)
            q, k, v, q_raw_std, k_raw_std = get_qkv(sa, x)
            q = to_bhtd(q, x, sa, name="q")
            k = to_bhtd(k, x, sa, name="k")
            v = to_bhtd(v, x, sa, name="v")
            B, h, T, d_k = q.shape
            d_model = h * d_k
            
            print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
            print(f"Q stats (after LN): mean={q.mean():.6f}, std={q.std():.6f}, min={q.min():.6f}, max={q.max():.6f}")
            print(f"K stats (after LN): mean={k.mean():.6f}, std={k.std():.6f}, min={k.min():.6f}, max={k.max():.6f}")
            print(f"Q raw std (pre-LN): {q_raw_std:.6f}, K raw std (pre-LN): {k_raw_std:.6f}")
            
            # Compute attention scores (matching model.py exactly)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # (B, h, T, T)
            
            # Apply per-head temperature scaling (same as model.py)
            if hasattr(sa, 'logit_temp'):
                import torch.nn.functional as Fn
                temp = Fn.softplus(sa.logit_temp)
                temp = torch.clamp(temp, min=getattr(sa, 'min_temp', 0.1),
                                         max=getattr(sa, 'max_temp', 2.0))
                scores = scores * temp
                print(f"\n  Temperature per head: {temp.squeeze().tolist()}")
            
            # Apply logit clamping (same as model.py)
            clamp_min = getattr(sa, 'logit_clamp_min', -20.0)
            clamp_max = getattr(sa, 'logit_clamp_max', 20.0)
            scores = scores.clamp(min=clamp_min, max=clamp_max)
            
            print(f"\nAttention scores (before softmax, after temp+clamp):")
            print(f"  Shape: {scores.shape}")
            print(f"  Stats: mean={scores.mean():.6f}, std={scores.std():.6f}")
            print(f"  Min: {scores.min():.6f}, Max: {scores.max():.6f}")
            print(f"  Range: {scores.max() - scores.min():.6f}")
            
            # Apply softmax
            attn = torch.softmax(scores, dim=-1)  # (B, h, T, T)
            
            print(f"\nAttention weights (after softmax):")
            print(f"  Shape: {attn.shape}")
            print(f"  Stats: mean={attn.mean():.6f}, std={attn.std():.6f}")
            print(f"  Min: {attn.min():.6f}, Max: {attn.max():.6f}")
            
            # Analyze per-head entropy
            print(f"\nPer-head entropy:")
            for head_idx in range(h):
                head_attn = attn[0, head_idx]  # (T, T)
                entropy = -(head_attn * torch.clamp(head_attn, min=1e-10).log()).sum(-1)  # (T,)
                avg_entropy = entropy.mean().item()
                max_entropy = np.log(T)
                normalized = avg_entropy / max_entropy if max_entropy > 0 else 0
                
                # Check if collapsed
                max_weight = head_attn.max().item()
                is_collapsed = "🔴 COLLAPSED" if max_weight > 0.95 else "✓"
                
                print(f"  Head {head_idx}: entropy={avg_entropy:.4f} (norm={normalized:.2%}), " 
                      f"max_weight={max_weight:.4f} {is_collapsed}")
            
            # Apply attention to values
            attn_out = torch.matmul(attn, v)  # (B, h, T, d_k)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d_model)  # (B, T, d_model)
            
            print(f"\nAttention output:")
            print(f"  Shape: {attn_out.shape}")
            print(f"  Stats: mean={attn_out.mean():.6f}, std={attn_out.std():.6f}")
            
            # Continue through feed-forward and layer norm
            x = layer(x, enc_mask)
            
            print(f"\nAfter layer processing:")
            print(f"  Output stats: mean={x.mean():.6f}, std={x.std():.6f}")
            print()


def check_entropy_regularization(checkpoint, config):
    """Check if entropy regularization is being computed correctly."""
    print("\n" + "="*80)
    print("ENTROPY REGULARIZATION CHECK")
    print("="*80)
    
    entropy_weight = config.get('entropy_reg_weight', 0.0)
    print(f"Entropy regularization weight: {entropy_weight}")
    
    if entropy_weight == 0:
        print("❌ ENTROPY REGULARIZATION DISABLED!")
        print("   This is the ROOT CAUSE of attention collapse.")
        print("   Set 'entropy_reg_weight': 1e-3 in config")
        return False
    
    print("✅ Entropy regularization enabled")
    return True


def check_initialization(model):
    """Check if model parameters are initialized properly."""
    print("\n" + "="*80)
    print("PARAMETER INITIALIZATION CHECK")
    print("="*80)
    
    for name, param in model.named_parameters():
        if 'attention' in name.lower():
            std = param.std().item()
            mean = param.mean().item()
            print(f"{name:50s}: mean={mean:8.6f}, std={std:8.6f}")
            
            if std < 0.001:
                print(f"  ⚠️  SUSPICIOUSLY LOW STD - may be stuck")
            if std > 1.0:
                print(f"  ⚠️  SUSPICIOUSLY HIGH STD - may cause instability")


def main():
    config = get_multi_dataset_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_path = "pretrain_weights_multi_fixed/pretrain_multi_fixed_best.pt"
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    step = ckpt.get('step', '?')
    print(f"Step: {step}")
    
    # Build model
    tokenizer = get_tokenizer(config['tokenizer_model'])
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config['num_layers'],
        h=config['num_heads'],
        dropout=0.0,  # No dropout for clarity
        d_ff=config['d_ff']
    ).to(device)
    
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    # Debug sequence
    print("\n" + "#"*80)
    print("# DEBUGGING ATTENTION COLLAPSE")
    print("#"*80)
    
    # 1. Check entropy regularization
    has_entropy = check_entropy_regularization(ckpt, config)
    
    # 2. Check parameter initialization
    check_initialization(model)
    
    # 3. Trace attention flow
    test_sentences = [
        "The quick brown fox jumps",
        "Science has shown that",
        "The United States is"
    ]
    
    for sent in test_sentences:
        debug_attention_flow(model, tokenizer, device, sent)
    
    # Summary
    print("\n" + "#"*80)
    print("# ROOT CAUSE ANALYSIS")
    print("#"*80)
    
    if not has_entropy:
        print("\n🔴 PRIMARY CAUSE: Entropy regularization disabled (weight=0)")
        print("   Without entropy loss, there's no penalty for collapsed attention.")
        print("   FIX: Set config['entropy_reg_weight'] = 1e-3")
    else:
        print("\n✅ Entropy regularization is enabled")
        print("   If collapse still occurs, it's likely:")
        print("   - Entropy loss value too low to overcome other gradients")
        print("   - Learning rate too high (causing instability)")
        print("   - Insufficient warmup steps allowing early collapse")
    
    print("\n" + "#"*80)
    print("\nTo continue training with fixes:")
    print("  python pretrain_multi.py")
    print("\nTo test again after more steps:")
    print("  python test_encoder.py pretrain_weights_multi/pretrain_multi_best.pt")


if __name__ == "__main__":
    main()
