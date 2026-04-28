"""
Model Health Analyzer — ONNX-focused, model-agnostic.

Analyzes attention softmax outputs and activations to produce:
  1. Per-layer/head entropy heatmap data
  2. Head type classification (local/global/sparse/uniform/collapsed)
  3. Signal propagation statistics (representation growth across layers)
  4. Overall health score (0–100)

Designed to work with any transformer ONNX model that exposes Softmax nodes
via GraphInstrumenter.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _entropy(attn_row: np.ndarray) -> float:
    """Shannon entropy of a single attention distribution (1-D)."""
    p = np.clip(attn_row, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _classify_head(norm_ent: float, avg_dist: float, target_var: float,
                   seq_len: int, sink_score: float = 0.0) -> str:
    """Classify a head based on entropy, locality, sparsity, and sink score.
    
    Attention sinks (Xiao et al. 2023) are heads that consistently attend to
    position 0 (BOS). They are load-bearing structural features, NOT failures.
    """
    # Sink detection first — these are healthy, load-bearing heads
    if sink_score > 0.5:
        if sink_score > 0.8 and norm_ent < 0.15:
            return "pure_sink"
        return "sink"
    if norm_ent > 0.80:
        return "uniform"
    if norm_ent < 0.25:
        return "collapsed" if target_var < 0.5 else "sparse"
    if avg_dist < seq_len * 0.20:
        return "local"
    return "global"


# ─────────────────────────────────────────────────────────────
# Core analyzer
# ─────────────────────────────────────────────────────────────

class ModelHealthAnalyzer:
    """
    Accepts the same ``attention_tensors`` dict as ``AttentionAnalyzer``
    (tensor_name → ndarray [B, H, T, T]) and optional ``activation_tensors``
    (tensor_name → ndarray), and produces a rich health report.
    """

    # ── Attention Health ────────────────────────────────────

    def analyze_attention_health(
        self,
        attention_tensors: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Returns a structured report:
        {
          "layers": [
            {
              "name": "softmax_output_0",
              "layer_idx": 0,
              "num_heads": 8,
              "heads": [
                {
                  "head_idx": 0,
                  "avg_entropy": 1.72,
                  "normalized_entropy": 0.96,
                  "avg_distance": 4.3,
                  "target_variance": 12.1,
                  "max_weight": 0.31,
                  "type": "global"
                }, ...
              ]
            }, ...
          ],
          "summary": { ... },
          "heatmap_data": [[norm_ent per head] per layer],  # for Plotly
          "heatmap_types": [["global", ...] per layer],
          "health_score": 87.5,
          "alerts": ["Layer 3 Head 1 is COLLAPSED", ...]
        }
        """
        layers: List[Dict[str, Any]] = []
        all_alerts: List[str] = []

        # Sort tensor names for consistent layer ordering
        sorted_names = sorted(attention_tensors.keys())

        for name in sorted_names:
            attn = attention_tensors[name]
            if attn.ndim != 4:
                continue

            batch, heads, seq_len, _ = attn.shape
            max_ent = np.log(seq_len) if seq_len > 1 else 1.0
            positions = np.arange(seq_len)

            head_reports: List[Dict[str, Any]] = []
            for h in range(heads):
                h_attn = attn[:, h, :, :]              # (B, T, T)

                # 1. Entropy
                h_ent = -(h_attn * np.log(np.clip(h_attn, 1e-12, 1.0))).sum(axis=-1)
                avg_entropy = float(np.mean(h_ent))
                norm_ent = avg_entropy / max_ent if max_ent > 0 else 0.0

                # 2. Locality (average attention distance)
                avg_dist = 0.0
                for q in range(seq_len):
                    dist = np.abs(positions - q).astype(np.float32)
                    avg_dist += float(np.mean((h_attn[:, q, :] * dist).sum(axis=-1)))
                avg_dist /= max(1, seq_len)

                # 3. Sparsity (target variance of argmax positions)
                max_pos = np.argmax(h_attn, axis=-1)
                target_var = float(np.var(max_pos))

                # 4. Max weight (peak sharpness)
                max_w = float(np.max(h_attn))

                # 5. Sink score (Xiao et al. 2023)
                sink_score = float(np.mean(h_attn[:, :, 0]))

                # 6. Classification (with sink awareness)
                htype = _classify_head(norm_ent, avg_dist, target_var, seq_len, sink_score)

                head_reports.append({
                    "head_idx": h,
                    "avg_entropy": round(avg_entropy, 4),
                    "normalized_entropy": round(norm_ent, 4),
                    "avg_distance": round(avg_dist, 4),
                    "target_variance": round(target_var, 4),
                    "max_weight": round(max_w, 4),
                    "sink_score": round(sink_score, 4),
                    "type": htype,
                })

                # Alerts
                if htype == "collapsed":
                    all_alerts.append(
                        f"🔴 {name} Head {h}: COLLAPSED (entropy={norm_ent:.0%})")
                elif htype == "uniform":
                    all_alerts.append(
                        f"🟡 {name} Head {h}: UNIFORM (entropy={norm_ent:.0%})")

            layers.append({
                "name": name,
                "layer_idx": len(layers),
                "num_heads": heads,
                "heads": head_reports,
            })

        # Build heatmap arrays
        heatmap_data = []
        heatmap_types = []
        for layer in layers:
            heatmap_data.append(
                [h["normalized_entropy"] for h in layer["heads"]])
            heatmap_types.append(
                [h["type"] for h in layer["heads"]])

        # Summary
        total = sum(l["num_heads"] for l in layers)
        type_counts = {"local": 0, "global": 0, "sparse": 0,
                       "uniform": 0, "collapsed": 0, "sink": 0, "pure_sink": 0}
        for layer in layers:
            for h in layer["heads"]:
                if h["type"] not in type_counts:
                    type_counts[h["type"]] = 0
                type_counts[h["type"]] += 1

        # Sinks are healthy, load-bearing heads (Xiao et al. 2023)
        healthy = (type_counts["local"] + type_counts["global"] +
                   type_counts["sparse"] + type_counts["sink"] +
                   type_counts["pure_sink"])
        health_score = (healthy / total * 100) if total > 0 else 0.0

        return {
            "layers": layers,
            "summary": {
                "total_heads": total,
                "healthy_heads": healthy,
                "head_types": type_counts,
            },
            "heatmap_data": heatmap_data,
            "heatmap_types": heatmap_types,
            "health_score": round(health_score, 1),
            "alerts": all_alerts,
        }

    # ── Signal Propagation ──────────────────────────────────

    def analyze_signal_propagation(
        self,
        activation_tensors: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Track how activation statistics evolve across layers.

        Returns:
        {
          "layers": [
            {"name": ..., "mean": ..., "std": ..., "min": ..., "max": ..., "sparsity": ...},
          ],
          "std_values": [float],       # for line chart
          "growth_alerts": ["..."],
        }
        """
        rows: List[Dict[str, Any]] = []
        sorted_names = sorted(activation_tensors.keys())

        for name in sorted_names:
            t = activation_tensors[name]
            if t.size == 0:
                continue
            flat = t.flatten()
            rows.append({
                "name": name,
                "mean": round(float(np.mean(flat)), 5),
                "std": round(float(np.std(flat)), 5),
                "min": round(float(np.min(flat)), 5),
                "max": round(float(np.max(flat)), 5),
                "sparsity": round(float(np.mean(np.abs(flat) < 1e-5)) * 100, 2),
            })

        # Growth alerts
        alerts: List[str] = []
        std_vals = [r["std"] for r in rows]
        for i in range(1, len(std_vals)):
            if std_vals[i - 1] > 0:
                ratio = std_vals[i] / std_vals[i - 1]
                # Greg Yang's Tensor Programs: 1.5× is the warning threshold
                # for detecting semantic melt early. Previous 3.0× was too lax.
                if ratio > 1.5:
                    severity = "🔴" if ratio > 3.0 else "🟡"
                    alerts.append(
                        f"{severity} {rows[i-1]['name']} → {rows[i]['name']}: "
                        f"std grew {ratio:.1f}× (gradient explosion risk)")
                elif ratio < 0.3:
                    alerts.append(
                        f"🟡 {rows[i-1]['name']} → {rows[i]['name']}: "
                        f"std shrank to {ratio:.2f}× (vanishing signal risk)")

        return {
            "layers": rows,
            "std_values": std_vals,
            "layer_names": [r["name"] for r in rows],
            "growth_alerts": alerts,
        }

    # ── Full report ─────────────────────────────────────────

    def full_report(
        self,
        attention_tensors: Dict[str, np.ndarray],
        activation_tensors: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        attn_health = self.analyze_attention_health(attention_tensors)
        signal = self.analyze_signal_propagation(activation_tensors)

        return {
            "attention_health": attn_health,
            "signal_propagation": signal,
        }
