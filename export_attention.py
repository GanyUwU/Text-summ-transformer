"""
Export All Attention Weights

Extracts and saves attention weights from ALL layers and heads to files.
Creates both CSV (human-readable) and NPZ (numpy) formats.

Output structure:
    attention_export/
    ├── encoder_self/
    │   ├── layer_0_head_0.csv
    │   ├── layer_0_head_1.csv
    │   └── ...
    ├── decoder_self/
    │   └── ...
    ├── decoder_cross/
    │   └── ...
    ├── summary.txt
    └── all_attention.npz
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from model import build_transformer
from config_summarization import get_config, latest_weights_file_path
from tokenizers import Tokenizer
from datasets import load_dataset


def extract_all_attention(model, encoder_input, encoder_mask, decoder_input, decoder_mask, device):
    """Extract attention weights from ALL layers and heads."""
    model.eval()
    
    attention_data = {
        'encoder_self': [],
        'decoder_self': [],
        'decoder_cross': [],
    }
    
    hooks = []
    
    # Hooks for encoder self-attention
    for i, layer in enumerate(model.encoder.layers):
        def make_hook(layer_idx, attn_type):
            def hook(module, input, output):
                if hasattr(module, 'attention_scores'):
                    attn = module.attention_scores.detach().cpu().numpy()
                    attention_data[attn_type].append({
                        'layer': layer_idx,
                        'attention': attn[0]  # Remove batch dim: [heads, seq, seq]
                    })
            return hook
        hooks.append(layer.self_attention_block.register_forward_hook(
            make_hook(i, 'encoder_self')
        ))
    
    # Hooks for decoder
    for i, layer in enumerate(model.decoder.layers):
        def make_hook(layer_idx, attn_type):
            def hook(module, input, output):
                if hasattr(module, 'attention_scores'):
                    attn = module.attention_scores.detach().cpu().numpy()
                    attention_data[attn_type].append({
                        'layer': layer_idx,
                        'attention': attn[0]
                    })
            return hook
        hooks.append(layer.self_attention_block.register_forward_hook(
            make_hook(i, 'decoder_self')
        ))
        hooks.append(layer.cross_attention_block.register_forward_hook(
            make_hook(i, 'decoder_cross')
        ))
    
    # Forward pass
    with torch.no_grad():
        encoder_output = model.encode(encoder_input.to(device), encoder_mask.to(device))
        model.decode(encoder_output, encoder_mask.to(device), 
                    decoder_input.to(device), decoder_mask.to(device))
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return attention_data


def save_attention_csv(attention_matrix, filepath, row_labels=None, col_labels=None):
    """Save a single attention matrix to CSV."""
    with open(filepath, 'w') as f:
        # Header row
        if col_labels:
            f.write(',' + ','.join(col_labels[:attention_matrix.shape[1]]) + '\n')
        else:
            f.write(',' + ','.join([f'pos_{j}' for j in range(attention_matrix.shape[1])]) + '\n')
        
        # Data rows
        for i in range(attention_matrix.shape[0]):
            if row_labels and i < len(row_labels):
                row_label = row_labels[i]
            else:
                row_label = f'pos_{i}'
            
            values = ','.join([f'{attention_matrix[i, j]:.6f}' for j in range(attention_matrix.shape[1])])
            f.write(f'{row_label},{values}\n')


def export_attention():
    """Main export function."""
    print("\n" + "="*60)
    print("EXPORTING ALL ATTENTION WEIGHTS")
    print("="*60)
    
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format('shared'))
    if not tokenizer_path.exists():
        # Try SentencePiece tokenizer
        from tokenizer_utils import get_tokenizer
        tokenizer = get_tokenizer("tokenizer_sp.model")
        vocab_size = tokenizer.get_vocab_size()
        use_sp = True
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab_size = tokenizer.get_vocab_size()
        use_sp = False
    
    print(f"Vocabulary: {vocab_size}")
    
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
        d_ff=config['d_ff']
    ).to(device)
    
    # Load weights
    weights_path = latest_weights_file_path(config)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded: {weights_path}")
    else:
        print("⚠ No weights found, using random initialization")
    
    # Load sample data
    print("\nLoading sample...")
    ds = load_dataset(config['datasource'], config['dataset_version'], split='validation[:1]')
    sample = ds[0]
    
    article = sample['article']
    summary = sample['highlights']
    
    print(f"Article: {article[:100]}...")
    print(f"Summary: {summary[:100]}...")
    
    # Tokenize
    if use_sp:
        src_ids = tokenizer.encode(article)[:config['src_seq_len']-2]
        tgt_ids = tokenizer.encode(summary)[:config['tgt_seq_len']-2]
        
        src_ids = [tokenizer.bos_id] + src_ids + [tokenizer.eos_id]
        tgt_ids = [tokenizer.bos_id] + tgt_ids + [tokenizer.eos_id]
        
        pad_id = tokenizer.pad_id
        
        # Pad
        src_ids = src_ids + [pad_id] * (config['src_seq_len'] - len(src_ids))
        tgt_ids = tgt_ids + [pad_id] * (config['tgt_seq_len'] - len(tgt_ids))
        
        src_tokens = [tokenizer.sp.id_to_piece(i) for i in src_ids[:50]]
        tgt_tokens = [tokenizer.sp.id_to_piece(i) for i in tgt_ids[:30]]
    else:
        enc = tokenizer.encode(article)
        src_ids = enc.ids[:config['src_seq_len']]
        
        enc2 = tokenizer.encode(summary)
        tgt_ids = enc2.ids[:config['tgt_seq_len']]
        
        pad_id = tokenizer.token_to_id('[PAD]')
        
        # Pad
        src_ids = src_ids + [pad_id] * (config['src_seq_len'] - len(src_ids))
        tgt_ids = tgt_ids + [pad_id] * (config['tgt_seq_len'] - len(tgt_ids))
        
        src_tokens = [tokenizer.decode([i]) for i in src_ids[:50]]
        tgt_tokens = [tokenizer.decode([i]) for i in tgt_ids[:30]]
    
    # Create tensors
    encoder_input = torch.tensor([src_ids], dtype=torch.long)
    decoder_input = torch.tensor([tgt_ids], dtype=torch.long)
    
    encoder_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(1)
    decoder_mask = (decoder_input != pad_id).unsqueeze(1) & torch.tril(
        torch.ones((1, config['tgt_seq_len'], config['tgt_seq_len']), dtype=torch.bool)
    )
    
    # Extract attention
    print("\n🔍 Extracting attention from all layers...")
    attention_data = extract_all_attention(
        model, encoder_input, encoder_mask, decoder_input, decoder_mask, device
    )
    
    # Create output directory
    output_dir = Path('attention_export')
    output_dir.mkdir(exist_ok=True)
    
    for subdir in ['encoder_self', 'decoder_self', 'decoder_cross']:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Save all attention matrices
    all_attention_npz = {}
    total_files = 0
    
    summary_lines = [
        "ATTENTION WEIGHTS EXPORT SUMMARY",
        "=" * 50,
        f"Model: {config['num_layers']} layers, {config['num_heads']} heads",
        f"Source sequence length: {config['src_seq_len']}",
        f"Target sequence length: {config['tgt_seq_len']}",
        "",
        "FILES EXPORTED:",
        ""
    ]
    
    for attn_type in ['encoder_self', 'decoder_self', 'decoder_cross']:
        print(f"\n📁 {attn_type}/")
        summary_lines.append(f"\n{attn_type}/")
        
        for layer_data in attention_data[attn_type]:
            layer_idx = layer_data['layer']
            attn_matrix = layer_data['attention']  # Shape: [heads, seq, seq]
            num_heads = attn_matrix.shape[0]
            
            for head_idx in range(num_heads):
                head_attn = attn_matrix[head_idx]  # Shape: [seq, seq]
                
                # Determine labels based on attention type
                if attn_type == 'encoder_self':
                    row_labels = src_tokens
                    col_labels = src_tokens
                elif attn_type == 'decoder_self':
                    row_labels = tgt_tokens
                    col_labels = tgt_tokens
                else:  # decoder_cross
                    row_labels = tgt_tokens
                    col_labels = src_tokens
                
                # Save CSV
                csv_filename = f"layer_{layer_idx}_head_{head_idx}.csv"
                csv_path = output_dir / attn_type / csv_filename
                save_attention_csv(head_attn, csv_path, row_labels, col_labels)
                
                # Store for NPZ
                npz_key = f"{attn_type}_L{layer_idx}_H{head_idx}"
                all_attention_npz[npz_key] = head_attn
                
                total_files += 1
                print(f"   ✓ {csv_filename} [{head_attn.shape[0]}x{head_attn.shape[1]}]")
                summary_lines.append(f"   {csv_filename} - shape {head_attn.shape}")
    
    # Save all as single NPZ file
    npz_path = output_dir / "all_attention.npz"
    np.savez(npz_path, **all_attention_npz)
    print(f"\n✓ Saved all attention to: {npz_path}")
    
    # Save token mappings
    tokens_path = output_dir / "tokens.json"
    with open(tokens_path, 'w') as f:
        json.dump({
            'source_tokens': src_tokens,
            'target_tokens': tgt_tokens,
            'article_preview': article[:500],
            'summary_preview': summary[:200]
        }, f, indent=2)
    print(f"✓ Saved token mappings to: {tokens_path}")
    
    # Save summary
    summary_lines.extend([
        "",
        "=" * 50,
        f"Total attention matrices exported: {total_files}",
        f"All matrices also saved in: all_attention.npz",
        f"Token mappings saved in: tokens.json"
    ])
    
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"\nTotal files: {total_files} CSV files")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nStructure:")
    print(f"  attention_export/")
    print(f"  ├── encoder_self/     (self-attention in encoder)")
    print(f"  ├── decoder_self/     (self-attention in decoder)")
    print(f"  ├── decoder_cross/    (cross-attention decoder→encoder)")
    print(f"  ├── all_attention.npz (all matrices in one file)")
    print(f"  ├── tokens.json       (token labels)")
    print(f"  └── summary.txt       (export summary)")


if __name__ == '__main__':
    export_attention()
