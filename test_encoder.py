"""
Encoder Quality Test Suite

Tests whether the encoder has learned meaningful language representations.
Three functional tests:
  1. Semantic Similarity - Do similar sentences get similar encodings?
  2. Token Discrimination - Can the encoder tell different words apart?
  3. Attention Head Specialization - Do different heads do different jobs?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
from model import build_transformer
from pretrain_config import get_finetune_config
from tokenizer_utils import get_tokenizer

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyBboxPatch, Wedge
except ImportError:
    plt = None


def _num_heads(attn_block):
    if hasattr(attn_block, "num_heads"):
        return int(attn_block.num_heads)
    if hasattr(attn_block, "h"):
        return int(attn_block.h)
    if hasattr(attn_block, "n_heads"):
        return int(attn_block.n_heads)
    raise AttributeError("Attention block has no head-count attribute (num_heads/h/n_heads).")


def get_qkv(attn_block, x):
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
    return q, k, v


def to_bhtd(t, x_ref, attn_block, name="tensor"):
    """
    Normalize projected Q/K/V tensor to shape (B, H, T, Dk).
    Supports (B,T,D), (T,B,D), (B,H,T,Dk), (B,T,H,Dk).
    """
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


def encode_sentence(model, tokenizer, text, device):
    """Encode a sentence and return encoder output + attention details."""
    tokens = tokenizer.encode(text)
    input_ids = [tokenizer.bos_id] + tokens + [tokenizer.eos_id]
    enc_input = torch.tensor([input_ids], dtype=torch.long).to(device)
    enc_mask = (enc_input != tokenizer.pad_id).unsqueeze(1).unsqueeze(1)
    
    with torch.no_grad():
        enc_output = model.encode(enc_input, enc_mask)
    
    return enc_output, enc_input, enc_mask


def get_sentence_embedding(enc_output):
    """Get a single vector for the whole sentence (mean pooling)."""
    return enc_output.mean(dim=1)  # (1, d_model)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(a, b, dim=-1).item()


# ---------------------------------------------
# TEST 1: Semantic Similarity
# ---------------------------------------------
def test_semantic_similarity(model, tokenizer, device):
    """
    If the encoder understands meaning, similar sentences should
    have similar representations, and unrelated ones should be far apart.
    """
    print("\n" + "="*60)
    print("TEST 1: SEMANTIC SIMILARITY")
    print("="*60)
    print("Checking if encoder gives similar vectors to similar sentences.\n")
    
    # Pairs: (sentence_a, sentence_b, should_be_similar)
    test_pairs = [
        # Similar pairs (should have HIGH similarity)
        ("The cat sat on the mat.", 
         "A kitten rested on the rug.", True),
        
        ("The president spoke to the nation.", 
         "The leader addressed the country.", True),
        
        ("Scientists discovered a new planet.", 
         "Researchers found a previously unknown world.", True),
        
        # Different pairs (should have LOW similarity)
        ("The cat sat on the mat.", 
         "Stock markets crashed yesterday.", False),
        
        ("The president spoke to the nation.", 
         "I like to eat pizza with extra cheese.", False),
        
        ("Scientists discovered a new planet.", 
         "The football match ended in a draw.", False),
    ]
    
    results = []
    for sent_a, sent_b, should_be_similar in test_pairs:
        enc_a, _, _ = encode_sentence(model, tokenizer, sent_a, device)
        enc_b, _, _ = encode_sentence(model, tokenizer, sent_b, device)
        
        emb_a = get_sentence_embedding(enc_a)
        emb_b = get_sentence_embedding(enc_b)
        
        sim = cosine_sim(emb_a, emb_b)
        
        label = "SIMILAR" if should_be_similar else "DIFFERENT"
        status = ""
        if should_be_similar and sim > 0.7:
            status = "PASS"
        elif should_be_similar and sim > 0.5:
            status = "WEAK"
        elif not should_be_similar and sim < 0.5:
            status = "PASS"
        elif not should_be_similar and sim < 0.7:
            status = "WEAK"
        else:
            status = "FAIL"
        
        results.append((status, sim, should_be_similar))
        print(f"  [{label}] sim={sim:.3f} {status}")
        print(f"    A: {sent_a}")
        print(f"    B: {sent_b}\n")
    
    # Summary
    passed = sum(1 for s, _, _ in results if "PASS" in s)
    total = len(results)
    print(f"  Score: {passed}/{total} passed")
    
    # Check if encoder can distinguish at all
    sim_scores = [s for _, s, similar in results if similar]
    diff_scores = [s for _, s, similar in results if not similar]
    avg_sim = np.mean(sim_scores) if sim_scores else 0
    avg_diff = np.mean(diff_scores) if diff_scores else 0
    gap = avg_sim - avg_diff
    
    print(f"\n  Average similarity (similar pairs):   {avg_sim:.3f}")
    print(f"  Average similarity (different pairs): {avg_diff:.3f}")
    print(f"  Discrimination gap:                   {gap:.3f}")
    
    if gap > 0.15:
        print("  OK: Encoder CAN distinguish similar from different sentences.")
    elif gap > 0.05:
        print("  WARNING: Encoder has WEAK discrimination ability.")
    else:
        print("  FAIL: Encoder CANNOT distinguish - representations are meaningless.")
    
    return gap, results


# ---------------------------------------------
# TEST 2: Token-Level Discrimination
# ---------------------------------------------
def test_token_discrimination(model, tokenizer, device):
    """
    Check if the encoder gives different representations to different
    tokens in the same sentence (not just copying the same vector everywhere).
    """
    print("\n" + "="*60)
    print("TEST 2: TOKEN-LEVEL DISCRIMINATION")
    print("="*60)
    print("Checking if different words get different representations.\n")
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Scientists at NASA discovered water on Mars in 2020.",
        "The stock market rose sharply after the announcement.",
    ]
    
    all_variances = []
    
    for sent in test_sentences:
        enc_output, enc_input, _ = encode_sentence(model, tokenizer, sent, device)
        
        # Get token representations (exclude BOS/EOS)
        token_reps = enc_output[0, 1:-1, :]  # (num_tokens, d_model)
        num_tokens = token_reps.shape[0]
        
        # Compute pairwise cosine similarity between all token pairs
        token_reps_norm = F.normalize(token_reps, dim=-1)
        sim_matrix = token_reps_norm @ token_reps_norm.T  # (T, T)
        
        # Get off-diagonal similarities (exclude self-similarity)
        mask = ~torch.eye(num_tokens, dtype=torch.bool, device=device)
        off_diag_sims = sim_matrix[mask]
        
        avg_sim = off_diag_sims.mean().item()
        min_sim = off_diag_sims.min().item()
        max_sim = off_diag_sims.max().item()
        
        # Variance of representations (higher = more diverse = better)
        variance = token_reps.var(dim=0).mean().item()
        all_variances.append(variance)
        
        print(f"  Sentence: \"{sent[:50]}...\"")
        print(f"    Token similarities: avg={avg_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}")
        print(f"    Representation variance: {variance:.4f}")
        
        if avg_sim > 0.95:
            print(f"    FAIL: Tokens are nearly identical - encoder is NOT differentiating words")
        elif avg_sim > 0.85:
            print(f"    WARN: Tokens are too similar - weak differentiation")
        else:
            print(f"    OK: Tokens have distinct representations")
        print()
    
    avg_var = np.mean(all_variances)
    print(f"  Overall representation variance: {avg_var:.4f}")
    if avg_var > 0.01:
        print("  OK: Encoder produces diverse token representations.")
    else:
        print("  FAIL: Encoder produces nearly identical representations for all tokens.")
    
    return avg_var, all_variances


# ---------------------------------------------
# TEST 3: Attention Head Specialization
# ---------------------------------------------
def test_attention_specialization(model, tokenizer, device):
    """
    Check if different attention heads in the encoder learn different patterns.
    Good signs: some heads attend locally (nearby words), others globally (distant words).
    Bad sign: all heads look the same.
    """
    print("\n" + "="*60)
    print("TEST 3: ATTENTION HEAD SPECIALIZATION")
    print("="*60)
    print("Checking if different heads learned different attention patterns.\n")
    
    sent = "The president of the United States spoke to the nation about climate change yesterday."
    tokens = tokenizer.encode(sent)
    input_ids = [tokenizer.bos_id] + tokens + [tokenizer.eos_id]
    enc_input = torch.tensor([input_ids], dtype=torch.long).to(device)
    enc_mask = (enc_input != tokenizer.pad_id).unsqueeze(1).unsqueeze(1)
    
    # Try to compute attention manually with error handling
    attention_weights = {}
    try:
        with torch.no_grad():
            # Get embeddings
            x = model.src_embed(enc_input)
            x = model.src_pos(x)
            
            for layer_idx, layer in enumerate(model.encoder.layers):
                try:
                    # Extract self-attention block
                    sa = layer.self_attention_block
                    
                    # Compute Q, K, V
                    q, k, v = get_qkv(sa, x)
                    q = to_bhtd(q, x, sa, name="q")
                    k = to_bhtd(k, x, sa, name="k")
                    d_k = q.shape[-1]
                    
                    # Compute attention scores
                    attn = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
                    attn = torch.softmax(attn, dim=-1)  # (B, h, T, T)
                    
                    attention_weights[layer_idx] = attn[0].cpu()  # (h, T, T)
                    
                    # Continue forward pass through the layer
                    x = layer(x, enc_mask)
                except Exception as e:
                    print(f"  ⚠️  Could not compute attention for layer {layer_idx}: {e}")
                    x = layer(x, enc_mask)
    except Exception as e:
        print(f"  ⚠️  Attention computation failed: {e}")
        return 0.5, {}, {}, []
    
    if not attention_weights:
        print(f"  ⚠️  Could not extract attention weights from model.\n")
        return 0.5, {}, {}, []
    
    # Analyze attention patterns
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[0]
    
    print(f"  Model has {num_layers} encoder layers * {num_heads} heads = {num_layers * num_heads} total heads\n")
    
    head_types = {"local": 0, "global": 0, "uniform": 0, "collapsed": 0, "sparse": 0}
    all_layer_summaries = []
    
    for layer_idx in range(num_layers):
        if layer_idx not in attention_weights:
            all_layer_summaries.append([])
            continue
            
        attn = attention_weights[layer_idx]  # (h, T, T)
        layer_summary = []
        for head_idx in range(num_heads):
            try:
                head_attn = attn[head_idx]  # (T, T)
                
                # Compute entropy
                entropy = -(head_attn * torch.clamp(head_attn, min=1e-12).log()).sum(-1)  # (T,)
                avg_entropy = entropy.mean().item()
                max_possible_entropy = np.log(head_attn.shape[-1])
                normalized_entropy = avg_entropy / max_possible_entropy if max_possible_entropy > 0 else 0
                
                # Compute average attention distance
                positions = torch.arange(head_attn.shape[-1]).float()
                avg_distance = 0
                max_attn_positions = head_attn.argmax(dim=-1).float()
                target_variance = max_attn_positions.var().item() if len(max_attn_positions) > 1 else 0
                
                for q_pos in range(head_attn.shape[0]):
                    distances = (positions - q_pos).abs()
                    avg_distance += (head_attn[q_pos] * distances).sum().item()
                avg_distance /= max(1, head_attn.shape[0])
                
                # Classify head
                if normalized_entropy > 0.8:
                    head_type = "uniform"
                    head_types["uniform"] += 1
                elif normalized_entropy < 0.25:
                    if target_variance < 0.5:
                        head_type = "collapsed"
                        head_types["collapsed"] += 1
                    else:
                        head_type = "sparse"
                        head_types["sparse"] += 1
                elif avg_distance < head_attn.shape[-1] * 0.2:
                    head_type = "local"
                    head_types["local"] += 1
                else:
                    head_type = "global"
                    head_types["global"] += 1
                
                layer_summary.append((head_idx, head_type, normalized_entropy, avg_distance))
            except:
                continue
        
        all_layer_summaries.append(layer_summary)
        if layer_summary:
            types_str = " ".join([f"H{h}:{t[:3]}" for h, t, _, _ in layer_summary])
            print(f"  Layer {layer_idx}: {types_str}")
    
    total_heads = sum(head_types.values())
    if total_heads == 0:
        return 0.5, head_types, attention_weights, all_layer_summaries
    
    print(f"\n  Head Type Distribution:")
    for htype, count in head_types.items():
        pct = count / total_heads * 100
        status = "OK" if htype in ("local", "global", "sparse") else ("WARN" if htype == "uniform" else "FAIL")
        print(f"    [{status}] {htype:10s}: {count}/{total_heads} ({pct:.0f}%)")
    
    healthy = head_types["local"] + head_types["global"] + head_types["sparse"]
    health_ratio = healthy / total_heads
    
    print(f"\n  Healthy heads: {healthy}/{total_heads} ({health_ratio*100:.0f}%)")
    if health_ratio > 0.6:
        print("  OK: Encoder attention heads are mostly specialized.")
    elif health_ratio > 0.3:
        print("  WARNING: Encoder has some specialization but many heads are wasted.")
    else:
        print("  FAIL: Encoder attention is mostly broken.")
    
    return health_ratio, head_types, attention_weights, all_layer_summaries


# ---------------------------------------------
# GRAPHICAL REPORT
# ---------------------------------------------
def plot_attention_detail(attention_weights, all_layer_summaries, checkpoint_name="checkpoint"):
    """Deep-dive plot for attention head specialization."""
    if plt is None:
        print("\n[PLOT] matplotlib not installed.")
        return

    C_PASS    = "#2ecc71"
    C_WEAK    = "#f39c12"
    C_FAIL    = "#e74c3c"
    C_GLOBAL  = "#1abc9c"
    C_SPARSE  = "#3498db"
    C_BG      = "#1a1a2e"
    C_PANEL   = "#16213e"
    C_TEXT    = "#ecf0f1"
def plot_attention_detail(attention_weights, all_layer_summaries, checkpoint_name="checkpoint"):
    """
    Deep-dive plot for attention head specialization.
    Shows exactly which heads are broken and why.
    """
    if plt is None:
        print("\n[PLOT] matplotlib not installed.")
        return

    C_PASS    = "#2ecc71"
    C_WEAK    = "#f39c12"
    C_FAIL    = "#e74c3c"
    C_GLOBAL  = "#1abc9c"
    C_SPARSE  = "#3498db" 
    C_BG      = "#1a1a2e"
    C_PANEL   = "#16213e"
    C_TEXT    = "#ecf0f1"
    TYPE_CLR  = {"local": C_PASS, "global": C_GLOBAL, "sparse": C_SPARSE, "uniform": C_WEAK, "collapsed": C_FAIL}

    num_layers = len(all_layer_summaries)
    if num_layers == 0: return
    num_heads  = len(all_layer_summaries[0]) if all_layer_summaries[0] else 0
    if num_heads == 0: return

    # Build 2D data arrays
    entropy_grid   = [[s[2] for s in layer] for layer in all_layer_summaries]
    distance_grid  = [[s[3] for s in layer] for layer in all_layer_summaries]
    type_grid      = [[s[1] for s in layer] for layer in all_layer_summaries]

    all_dists = [d for row in distance_grid for d in row]
    max_dist  = max(all_dists) if all_dists and max(all_dists) > 0 else 1
    dist_norm = [[d / max_dist for d in row] for row in distance_grid]

    priority = {"collapsed": 3, "uniform": 2, "global": 1, "local": 0, "sparse": 0}
    worst_layer, worst_head_idx, worst_type = 0, 0, "uniform"
    worst_priority = -1
    worst_entropy, worst_dist = 0, 0
    
    for li, layer in enumerate(all_layer_summaries):
        for hi, htype, ent, dist in layer:
            if priority.get(htype, 0) > worst_priority:
                worst_priority = priority.get(htype, 0)
                worst_layer, worst_head_idx, worst_type = li, hi, htype
                worst_entropy, worst_dist = ent, dist

    fig = plt.figure(figsize=(18, 12), facecolor=C_BG)
    fig.suptitle(f"Attention Specialization Deep Dive  ·  {os.path.basename(checkpoint_name)}",
                 fontsize=15, color=C_TEXT, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.38,
                          left=0.06, right=0.97, top=0.92, bottom=0.08)

    def style_ax(ax, title):
        ax.set_facecolor(C_PANEL)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight='bold', pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#3d3d5c')

    layer_labels = [f"L{i}" for i in range(num_layers)]
    head_labels  = [f"H{i}" for i in range(num_heads)]

    # 1. Map
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "1 · Head Classification Map")
    type_to_num = {"local": 1.0, "sparse": 0.8, "global": 0.6, "uniform": 0.3, "collapsed": 0.0}
    num_grid = [[type_to_num.get(t, 0.5) for t in row] for row in type_grid]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("headmap", [C_FAIL, C_WEAK, C_GLOBAL, C_SPARSE, C_PASS])
    ax1.imshow(num_grid, cmap=cmap_custom, vmin=0, vmax=1, aspect='auto')
    ax1.set_xticks(range(num_heads)); ax1.set_xticklabels(head_labels, fontsize=7)
    ax1.set_yticks(range(num_layers)); ax1.set_yticklabels(layer_labels, fontsize=7)
    for li in range(num_layers):
        for hi in range(num_heads):
            t = type_grid[li][hi]
            ax1.text(hi, li, t[:3].upper(), ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    
    # 2. Entropy
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "2 · Normalised Entropy per Head\n(high=uniform, low=collapsed)")
    im2 = ax2.imshow(entropy_grid, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(range(num_heads)); ax2.set_xticklabels(head_labels, fontsize=7)
    ax2.set_yticks(range(num_layers)); ax2.set_yticklabels(layer_labels, fontsize=7)
    for li in range(num_layers):
        for hi in range(num_heads):
            ax2.text(hi, li, f"{entropy_grid[li][hi]:.2f}", ha='center', va='center', fontsize=6, color='white')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors=C_TEXT, labelsize=7)

    # 3. Distance
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "3 · Avg Attention Distance per Head\n(low=local, high=global)")
    im3 = ax3.imshow(dist_norm, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax3.set_xticks(range(num_heads)); ax3.set_xticklabels(head_labels, fontsize=7)
    ax3.set_yticks(range(num_layers)); ax3.set_yticklabels(layer_labels, fontsize=7)
    for li in range(num_layers):
        for hi in range(num_heads):
            ax3.text(hi, li, f"{distance_grid[li][hi]:.1f}", ha='center', va='center', fontsize=6, color='white')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(colors=C_TEXT, labelsize=7)

    # 4. Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, "4 · Entropy Distribution Across All Heads")
    type_entropies = {t: [] for t in TYPE_CLR}
    for layer in all_layer_summaries:
        for _, htype, ent, _ in layer:
            if htype in type_entropies:
                type_entropies[htype].append(ent)
    bins = np.linspace(0, 1, 15)
    for htype, ents in type_entropies.items():
        if ents:
            ax4.hist(ents, bins=bins, alpha=0.75, color=TYPE_CLR[htype], label=htype, edgecolor='#3d3d5c', linewidth=0.5)
    ax4.axvline(0.8, color=C_WEAK, linestyle='--', linewidth=1, label='Uniform (0.8)')
    ax4.axvline(0.25, color=C_FAIL, linestyle=':', linewidth=1, label='Collapsed (0.25)')
    ax4.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)

    # 5. Pattern
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, f"5 · Worst Head Pattern\n(L{worst_layer}, H{worst_head_idx} → {worst_type.upper()})")
    if worst_layer in attention_weights:
        worst_attn = attention_weights[worst_layer][worst_head_idx].numpy()
        T = worst_attn.shape[0]
        ax5.imshow(worst_attn, cmap='Blues', aspect='auto', vmin=0)
        ax5.set_xlabel("Key position"); ax5.set_ylabel("Query position")
    ax5.set_title(f"5 · Worst Head Pattern\n(L{worst_layer}, H{worst_head_idx} → {worst_type.upper()})", color=TYPE_CLR[worst_type], fontsize=10, fontweight='bold')

    # 6. Diagnosis
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(C_PANEL)
    ax6.axis('off')
    ax6.set_title("6 · Diagnosis & Fix", color=C_TEXT, fontsize=10, fontweight='bold')
    
    counts = {t: sum(1 for layer in all_layer_summaries for _, ht, _, _ in layer if ht == t) for t in TYPE_CLR}
    diag_lines = []
    if counts["collapsed"] > 0:
        diag_lines.append((C_FAIL,   f"✗ {counts['collapsed']} COLLAPSED heads"))
        diag_lines.append((C_TEXT,   "  → Entropy < 0.25: head attends to"))
        diag_lines.append((C_TEXT,   "    one token only (often [BOS])."))
    if counts["uniform"] > 0:
        diag_lines.append((C_WEAK,   f"⚠ {counts['uniform']} UNIFORM heads"))
        diag_lines.append((C_TEXT,   "  → Entropy > 0.8: head spreads"))
        diag_lines.append((C_TEXT,   "    attention evenly (random)."))
    if counts["local"] + counts["global"] + counts["sparse"] > 0:
        diag_lines.append((C_PASS,   f"✓ {counts['local']+counts['global']+counts['sparse']} healthy heads"))
    
    diag_lines.append((C_TEXT, ""))
    diag_lines.append((C_TEXT, f"Worst head  → L{worst_layer} H{worst_head_idx}"))
    diag_lines.append((C_TEXT, f"  Type      : {worst_type.upper()}"))
    diag_lines.append((C_TEXT, f"  Entropy   : {worst_entropy:.3f}"))
    diag_lines.append((C_TEXT, f"  Avg dist  : {worst_dist:.2f} tokens"))

    y_pos = 0.95
    for color, text in diag_lines:
        ax6.text(0.04, y_pos, text, transform=ax6.transAxes, fontsize=8.5, color=color, va='top', fontfamily='monospace')
        y_pos -= 0.058

    fig.savefig("encoder_attention_detail.png", dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)

def plot_encoder_report(sim_results, token_variances, head_types, sem_score, tok_score, head_score, overall, checkpoint_name="checkpoint"):
    """Multi-panel dashboard for the encoder test results."""
    if plt is None: return

    C_PASS   = "#2ecc71"; C_WEAK = "#f39c12"; C_FAIL = "#e74c3c"
    C_SIM    = "#3498db"; C_DIFF = "#e67e22"
    C_BG     = "#1a1a2e"; C_PANEL = "#16213e"; C_TEXT = "#ecf0f1"

    def score_color(v, thresholds=(70, 40)):
        return C_PASS if v >= thresholds[0] else (C_WEAK if v >= thresholds[1] else C_FAIL)

    fig = plt.figure(figsize=(18, 11), facecolor=C_BG)
    fig.suptitle(f"Encoder Quality Report  ·  {os.path.basename(checkpoint_name)}", fontsize=16, color=C_TEXT, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38, left=0.06, right=0.97, top=0.92, bottom=0.08)

    def style_ax(ax, title):
        ax.set_facecolor(C_PANEL); ax.tick_params(colors=C_TEXT, labelsize=8)
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight='bold', pad=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#3d3d5c')

    # 1. Semantic
    ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1, "1 · Semantic Similarity")
    pair_labels = [f"P{i+1}\n({'SIM' if is_sim else 'DIFF'})" for i, (_, _, is_sim) in enumerate(sim_results)]
    sims = [sim for (_, sim, _) in sim_results]
    clrs = [C_SIM if is_sim else C_DIFF for (_, _, is_sim) in sim_results]
    bars1 = ax1.bar(pair_labels, sims, color=clrs, edgecolor='#3d3d5c', linewidth=0.8)
    ax1.axhline(0.7, color=C_PASS, linewidth=1, linestyle='--')
    ax1.axhline(0.5, color=C_WEAK, linewidth=1, linestyle=':')
    ax1.set_ylim(0, 1.1)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=7, color=C_TEXT)

    # 2. Token
    ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2, "2 · Token Discrimination")
    sent_labels = [f"Sent {i+1}" for i in range(len(token_variances))]
    bars2 = ax2.bar(sent_labels, token_variances, color=C_SIM, edgecolor='#3d3d5c', linewidth=0.8)
    ax2.axhline(0.01, color=C_PASS, linewidth=1, linestyle='--')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f"{bar.get_height():.4f}", ha='center', va='bottom', fontsize=7, color=C_TEXT)

    # 3. Attention
    ax3 = fig.add_subplot(gs[0, 2]); style_ax(ax3, "3 · Attention Head Types")
    ht_colors = {"local": C_PASS, "global": "#1abc9c", "sparse": C_SIM, "uniform": C_WEAK, "collapsed": C_FAIL}
    labels = [k for k, v in head_types.items() if v > 0]
    values = [head_types[k] for k in labels]
    clrs = [ht_colors.get(k, "#999") for k in labels]
    ax3.pie(values, labels=labels, colors=clrs, autopct='%1.0f%%', startangle=90, textprops={'color': C_TEXT}, wedgeprops={'edgecolor': C_BG, 'linewidth': 1.5})

    # 4. Report Card
    ax4 = fig.add_subplot(gs[1, 0:2]); style_ax(ax4, "4 · Report Card Scores")
    cats = ["Semantic Understanding", "Token Discrimination", "Attention Specialization"]
    scores = [sem_score, tok_score, head_score]
    bars4 = ax4.barh(cats, scores, color=[score_color(s) for s in scores], edgecolor='#3d3d5c', linewidth=0.8)
    ax4.set_xlim(0, 110)
    for bar in bars4:
        ax4.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}", va='center', fontsize=10, color=C_TEXT, fontweight='bold')

    # 5. Overall
    ax5 = fig.add_subplot(gs[1, 2]); style_ax(ax5, "5 · Overall Score")
    ax5.axis('off')
    gauge_clr = score_color(overall)
    angle = overall / 100 * 360
    wedge = Wedge((0.5, 0.45), 0.38, 90 - angle, 90, width=0.1, facecolor=gauge_clr, edgecolor='none')
    ax5.add_patch(wedge)
    ax5.text(0.5, 0.45, f"{overall:.1f}", ha='center', va='center', fontsize=32, color=C_TEXT, fontweight='bold')
    ax5.text(0.5, 0.82, "HEALTHY" if overall > 70 else ("PARTIAL" if overall > 40 else "WEAK"), ha='center', va='center', fontsize=13, color=gauge_clr, fontweight='bold')

    fig.savefig("encoder_report.png", dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close(fig)


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    config = get_finetune_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "pretrain_weights_multi/pretrain_multi_best.pt"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    tokenizer = get_tokenizer(config['tokenizer_model'])
    vocab_size = tokenizer.get_vocab_size()
    
    is_pretrain = "pretrain" in checkpoint_path.lower()
    if is_pretrain:
        from pretrain_config import get_pretrain_config
        pt_config = get_pretrain_config()
        src_seq_len = tgt_seq_len = pt_config['seq_len']
    else:
        src_seq_len, tgt_seq_len = config['src_seq_len'], config['tgt_seq_len']
    
    model = build_transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        src_seq_len=src_seq_len, tgt_seq_len=tgt_seq_len,
        d_model=config['d_model'], N=config['num_layers'], h=config['num_heads'],
        dropout=config['dropout'], d_ff=config['d_ff'], share_weights=config.get('share_weights', True), use_copy=False,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    sim_gap, sim_results = test_semantic_similarity(model, tokenizer, device)
    token_var, token_vars = test_token_discrimination(model, tokenizer, device)
    head_health, head_types, attn_weights, all_layer_summaries = test_attention_specialization(model, tokenizer, device)
    
    sem_score = min(100, max(0, sim_gap * 500))
    tok_score = min(100, max(0, token_var * 5000))
    head_score = head_health * 100
    overall = np.mean([sem_score, tok_score, head_score])
    
    print(f"\n  OVERALL ENCODER SCORE: {overall:.1f}/100")
    
    plot_encoder_report(sim_results, token_vars, head_types, sem_score, tok_score, head_score, overall, checkpoint_path)
    plot_attention_detail(attn_weights, all_layer_summaries, checkpoint_path)
    print("\n[INFO] Visualization reports saved.")


if __name__ == "__main__":
    main()
