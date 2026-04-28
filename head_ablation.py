"""
Causal Head Ablation Diagnostic (Wang et al. 2023)

Entropy-based head classification is correlational — a head can have healthy
entropy but contribute nothing to the output. The correct diagnostic is causal:
zero-ablate the head and measure the loss increase.

For a 6-layer, 8-head model this is only 48 forward passes — cheap and
causally correct unlike entropy.

Thresholds (from IOI circuit analysis, Wang et al. 2023):
  - importance < 0.001 nats → head is redundant
  - importance > 0.1 nats  → head is load-bearing

Usage:
    python head_ablation.py --checkpoint weights_v11_nuclear/nuclear_summarizer_epoch_4.pt --samples 50
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import copy
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrain_config import get_finetune_config
from model import build_transformer
from tokenizer_utils import get_tokenizer
from checkpoint_utils import load_checkpoint
from train_summarization import SummarizationDataset
from datasets import load_dataset
from torch.utils.data import DataLoader


def compute_loss(model, batch, device):
    """Compute cross-entropy loss for a single batch."""
    enc_input = batch['encoder_input'].to(device)
    enc_mask = batch['encoder_mask'].to(device)
    dec_input = batch['decoder_input'].to(device)
    dec_mask = batch['decoder_mask'].to(device)
    label = batch['label'].to(device)

    enc_output = model.encode(enc_input, enc_mask)

    # Standard decode (no copy mechanism for clean loss measurement)
    dec_output = model.decode(enc_output, enc_mask, dec_input, dec_mask)
    logits = model.project(dec_output)  # (B, T, vocab)

    # Cross-entropy ignoring padding
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        label.view(-1),
        ignore_index=0,  # Padding token
        reduction='mean'
    )
    return loss.item()


def get_all_attention_modules(model):
    """Find all MultiHeadAttentionBlock modules in the model."""
    attention_modules = []
    
    # Encoder self-attention
    for i, layer in enumerate(model.encoder.layers):
        attention_modules.append({
            "name": f"encoder.layer_{i}.self_attn",
            "module": layer.self_attention_block,
            "layer_idx": i,
            "type": "encoder_self"
        })
    
    # Decoder self-attention and cross-attention
    for i, layer in enumerate(model.decoder.layers):
        attention_modules.append({
            "name": f"decoder.layer_{i}.self_attn",
            "module": layer.self_attention_block,
            "layer_idx": i,
            "type": "decoder_self"
        })
        attention_modules.append({
            "name": f"decoder.layer_{i}.cross_attn",
            "module": layer.cross_attention_block,
            "layer_idx": i,
            "type": "decoder_cross"
        })
    
    return attention_modules


def ablate_head(module, head_idx, num_heads, head_dim):
    """Zero-ablate a single attention head's output projection.
    
    This is the standard mean-ablation approach from Wang et al. 2023:
    we zero out the specific head's slice of the output projection weight,
    effectively silencing that head's contribution to the residual stream.
    """
    # Save original weights
    original_weight = module.out_proj.weight.data.clone()
    original_bias = None
    if module.out_proj.bias is not None:
        original_bias = module.out_proj.bias.data.clone()
    
    # Zero out the head's columns in out_proj
    # out_proj.weight shape: (d_model, d_model)
    # Head h corresponds to columns [h*head_dim : (h+1)*head_dim]
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim
    module.out_proj.weight.data[:, start:end] = 0.0
    
    return original_weight, original_bias


def restore_head(module, original_weight, original_bias):
    """Restore original weights after ablation."""
    module.out_proj.weight.data = original_weight
    if original_bias is not None:
        module.out_proj.bias.data = original_bias


def run_ablation_study(model, dataloader, device, num_samples, config):
    """Run causal head ablation across all heads.
    
    For each head, zero-ablate it and measure loss increase.
    48 forward passes total for a 6-layer, 8-head model.
    """
    model.eval()
    num_heads = config['num_heads']
    d_model = config['d_model']
    head_dim = d_model // num_heads
    
    # Step 1: Compute baseline loss
    print("Computing baseline loss...")
    baseline_losses = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            loss = compute_loss(model, batch, device)
            baseline_losses.append(loss)
    baseline_loss = np.mean(baseline_losses)
    print(f"Baseline loss: {baseline_loss:.4f}")
    
    # Step 2: Ablate each head and measure loss increase
    attention_modules = get_all_attention_modules(model)
    results = {
        "baseline_loss": float(baseline_loss),
        "heads": [],
        "redundant_heads": 0,
        "load_bearing_heads": 0,
        "total_heads": 0,
    }
    
    print(f"\nRunning ablation on {len(attention_modules)} attention blocks × {num_heads} heads...")
    
    for attn_info in tqdm(attention_modules, desc="Attention blocks"):
        module = attn_info["module"]
        
        for h in range(num_heads):
            # Ablate head h
            orig_w, orig_b = ablate_head(module, h, num_heads, head_dim)
            
            # Measure ablated loss
            ablated_losses = []
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= num_samples:
                        break
                    loss = compute_loss(model, batch, device)
                    ablated_losses.append(loss)
            
            ablated_loss = np.mean(ablated_losses)
            importance = ablated_loss - baseline_loss
            
            # Restore original weights
            restore_head(module, orig_w, orig_b)
            
            # Classify (Wang et al. 2023 thresholds)
            if importance < 0.001:
                classification = "redundant"
                results["redundant_heads"] += 1
            elif importance > 0.1:
                classification = "load_bearing"
                results["load_bearing_heads"] += 1
            else:
                classification = "contributing"
            
            results["total_heads"] += 1
            results["heads"].append({
                "name": attn_info["name"],
                "head_idx": h,
                "type": attn_info["type"],
                "layer_idx": attn_info["layer_idx"],
                "baseline_loss": round(float(baseline_loss), 4),
                "ablated_loss": round(float(ablated_loss), 4),
                "importance_delta": round(float(importance), 4),
                "classification": classification,
            })
    
    # Summary statistics
    importances = [h["importance_delta"] for h in results["heads"]]
    results["summary"] = {
        "mean_importance": round(float(np.mean(importances)), 4),
        "max_importance": round(float(np.max(importances)), 4),
        "min_importance": round(float(np.min(importances)), 4),
        "redundant_pct": round(results["redundant_heads"] / max(1, results["total_heads"]) * 100, 1),
        "load_bearing_pct": round(results["load_bearing_heads"] / max(1, results["total_heads"]) * 100, 1),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Causal Head Ablation (Wang et al. 2023)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--samples", type=int, default=50, help="Number of test samples for loss computation")
    parser.add_argument("--output", type=str, default="head_ablation_report.json", help="Output file")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    config = get_finetune_config()
    tokenizer = get_tokenizer(config['tokenizer_model'])
    
    # Build & load model
    model = build_transformer(
        src_vocab_size=tokenizer.get_vocab_size(),
        tgt_vocab_size=tokenizer.get_vocab_size(),
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
    
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.checkpoint}")
    
    # Load test data
    dataset_raw = load_dataset(config['datasource'], config['dataset_version'], split="test")
    dataset_subset = dataset_raw.select(range(min(args.samples, len(dataset_raw))))
    eval_dataset = SummarizationDataset(
        dataset_subset, tokenizer, config['src_seq_len'], config['tgt_seq_len'],
        lead_mask_prob=0.0
    )
    dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    # Run ablation
    results = run_ablation_study(model, dataloader, device, args.samples, config)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"HEAD ABLATION RESULTS (Wang et al. 2023)")
    print(f"{'='*60}")
    print(f"Baseline Loss: {results['baseline_loss']:.4f}")
    print(f"Total Heads:   {results['total_heads']}")
    print(f"Redundant:     {results['redundant_heads']} ({results['summary']['redundant_pct']}%)")
    print(f"Load-bearing:  {results['load_bearing_heads']} ({results['summary']['load_bearing_pct']}%)")
    print(f"Mean ΔL:       {results['summary']['mean_importance']:.4f}")
    print(f"Max ΔL:        {results['summary']['max_importance']:.4f}")
    
    # Top 5 most important heads
    sorted_heads = sorted(results["heads"], key=lambda x: x["importance_delta"], reverse=True)
    print(f"\nTop 5 Most Load-Bearing Heads:")
    for h in sorted_heads[:5]:
        print(f"  {h['name']} Head {h['head_idx']}: ΔL = {h['importance_delta']:.4f} [{h['classification']}]")
    
    # Save report
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
