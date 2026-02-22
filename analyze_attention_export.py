import numpy as np
import json
from pathlib import Path


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=-1)
    return ent


def analyze(npz_path, tokens_path=None, out_dir=None):
    npz_path = Path(npz_path)
    if out_dir is None:
        out_dir = npz_path.parent
    out_dir = Path(out_dir)
    data = np.load(str(npz_path), allow_pickle=True)

    # Group keys by attn_type and layer
    grouped = {}
    for k in data.files:
        # keys like 'encoder_self_L0_H0'
        parts = k.split('_')
        attn_type = parts[0] + '_' + parts[1]
        layer_part = [s for s in parts if s.startswith('L')][0]
        head_part = [s for s in parts if s.startswith('H')][0]
        layer = int(layer_part[1:])
        head = int(head_part[1:])
        grouped.setdefault(attn_type, {}).setdefault(layer, {})[head] = data[k]

    report = {'per_type': {}}

    for attn_type, layers in grouped.items():
        report['per_type'][attn_type] = {}
        for layer, heads in sorted(layers.items()):
            head_metrics = {}
            # Collect flattened vectors to compute pairwise similarity
            flat_heads = []
            for h_idx, mat in sorted(heads.items()):
                arr = np.asarray(mat)
                # Ensure non-negative and rows sum to 1
                if arr.ndim == 2:
                    rows = arr
                else:
                    rows = arr

                # per-row entropy
                ent = entropy(rows)
                mean_entropy = float(np.mean(ent))
                # normalized entropy (0..1) by log(ncols)
                ncols = rows.shape[1] if rows.shape[1] > 1 else 1
                norm_ent = mean_entropy / (np.log(ncols) + 1e-12)

                # top-k mass
                top1 = float(np.mean(np.max(rows, axis=-1)))
                top3 = float(np.mean(np.sort(rows, axis=-1)[:, -3:].sum(axis=-1)))

                head_metrics[h_idx] = {
                    'mean_entropy': mean_entropy,
                    'norm_entropy': norm_ent,
                    'top1_mass': top1,
                    'top3_mass': top3,
                    'shape': rows.shape,
                }

                flat_heads.append(rows.flatten())

            # Compute pairwise cosine similarity
            flat_mat = np.stack([f / (np.linalg.norm(f) + 1e-12) for f in flat_heads], axis=0)
            sim = np.dot(flat_mat, flat_mat.T)

            # Detect suspects
            suspects = []
            for h_idx, metrics in head_metrics.items():
                if metrics['norm_entropy'] < 0.18 or metrics['top1_mass'] > 0.62:
                    suspects.append({'layer': layer, 'head': h_idx, 'reason': 'low_entropy_or_high_top1', 'metrics': metrics})

            duplicates = []
            num_heads = sim.shape[0]
            for i in range(num_heads):
                for j in range(i+1, num_heads):
                    if sim[i, j] > 0.98:
                        duplicates.append({'layer': layer, 'head_pair': (i, j), 'sim': float(sim[i, j])})

            report['per_type'][attn_type][layer] = {
                'head_metrics': head_metrics,
                'pairwise_sim': sim.tolist(),
                'suspected_collapsed_heads': suspects,
                'suspected_duplicate_heads': duplicates,
            }

    # Save JSON report
    out_path = out_dir / 'attention_analysis.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Print a concise human-readable summary
    print('\nAttention Analysis Summary')
    print('='*40)
    for attn_type, layers in report['per_type'].items():
        print(f'\n{attn_type}:')
        for layer, info in sorted(layers.items()):
            suspects = info['suspected_collapsed_heads']
            duplicates = info['suspected_duplicate_heads']
            print(f'  Layer {layer}:')
            for h, m in sorted(info['head_metrics'].items()):
                print(f"    Head {h}: entropy={m['mean_entropy']:.4f} norm={m['norm_entropy']:.3f} top1={m['top1_mass']:.3f} top3={m['top3_mass']:.3f}")
            if suspects:
                print(f'    >> Suspected collapsed heads: {[f"H{d["head"]}" for d in suspects]}')
            if duplicates:
                print(f'    >> Suspected duplicate head pairs: {[p["head_pair"] for p in duplicates]}')

    print(f'\nSaved JSON report to: {out_path}')
    return report


if __name__ == '__main__':
    base = Path('attention_export')
    npz = base / 'all_attention.npz'
    tokens = base / 'tokens.json'
    if not npz.exists():
        print('ERROR: attention_export/all_attention.npz not found. Run export_attention.py first.')
    else:
        analyze(npz, tokens, base)
