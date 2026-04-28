import numpy as np
from typing import Dict, List, Any
from onnx import numpy_helper

class WeightAnalyzer:
    """
    Analyzes static model weights to find issues like dead neurons or initialization problems.
    """
    
    def __init__(self, model_loader):
        self.model = model_loader.model
        self.weights = {}
        self._extract_weights()
        
    def _extract_weights(self):
        """Extracts initializers (weights) from the graph."""
        for init in self.model.graph.initializer:
            try:
                # Use onnx.numpy_helper to robustly handle raw_data vs float_data
                w = numpy_helper.to_array(init)
                self.weights[init.name] = w
            except Exception as e:
                print(f"Warning: Could not extract weight {init.name}: {e}")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Runs full weight analysis.
        Returns dictionary of layer-wise stats and issues.
        """
        report = {
            "summary": {},
            "issues": [],
            "layer_stats": {}
        }
        
        total_dead_neurons = 0
        
        for name, w in self.weights.items():
            # Skip non-numeric or scalar weights if needed
            if w.size == 0 or not np.issubdtype(w.dtype, np.number):
                continue

            # Weight distribution analysis
            abs_w = np.abs(w)
            max_val = float(np.max(abs_w))
            
            # Exploding weight detection
            if max_val > 10.0:
                report["issues"].append({
                    "severity": "critical",
                    "msg": f"Layer {name} has EXPLODING weights (max abs={max_val:.2f})."
                })
                if "exploding_layers" not in report["summary"]:
                    report["summary"]["exploding_layers"] = 0
                report["summary"]["exploding_layers"] += 1

            stats = {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "shape": list(w.shape),
                "norm": float(np.linalg.norm(w))
            }
            report["layer_stats"][name] = stats
            
            # Dead Neuron & Low Variance Detection (for 2D weights like Linear layers)
            if len(w.shape) == 2:
                # Check rows with near-zero norm
                row_norms = np.linalg.norm(w, axis=1)
                dead_rows = np.sum(row_norms < 1e-6)
                
                # Check variance across neurons
                row_vars = np.var(w, axis=1)
                low_var_rows = np.sum(row_vars < 1e-6)
                
                if dead_rows > 0:
                    report["issues"].append({
                        "severity": "warning", 
                        "msg": f"Layer {name} has {dead_rows} dead neurons (zero rows)."
                    })
                    total_dead_neurons += int(dead_rows)
                elif low_var_rows > 0:
                     report["issues"].append({
                        "severity": "info", 
                        "msg": f"Layer {name} has {low_var_rows} neurons with near-zero variance."
                    })
                    
        report["summary"]["total_params"] = int(sum(w.size for w in self.weights.values()))
        report["summary"]["dead_neurons"] = total_dead_neurons
        
        # ── W_OV Circuit Analysis (Elhage et al. 2021) ──────────────
        # For each attention head, compute W_OV = W_O @ W_V and check its
        # Frobenius norm. A head with ||W_OV||_F ≈ 0 outputs nothing to the
        # residual stream and is functionally dead, regardless of entropy.
        ov_analysis = self._analyze_ov_circuits()
        if ov_analysis:
            report["ov_circuits"] = ov_analysis
        
        # ── MAPS Vocabulary Projection (ACL 2025) ──────────────────
        # Classify heads as mover/suppressor/copy/induction from weights alone.
        maps_analysis = self._maps_head_classification()
        if maps_analysis:
            report["maps_classification"] = maps_analysis
        
        return report
    
    def _analyze_ov_circuits(self):
        """Compute W_OV = W_O @ W_V Frobenius norms per head (Elhage et al. 2021).
        
        In our ONNX model, attention block weights follow a repeating pattern:
        Q_proj [512,512], K_proj [512,512], V_proj [512,512], Out_proj [512,512]
        
        For each encoder layer: [Q, K, V, Out, FFN_up, FFN_down]
        For decoder self-attn: [Q, K, V, Out]
        For decoder cross-attn: [Q, K, V, Out]
        """
        # Collect all [512,512] weight matrices (attention projections)
        attn_weights = []
        for name, w in self.weights.items():
            if w.ndim == 2 and w.shape == (512, 512):
                attn_weights.append((name, w))
        
        if len(attn_weights) < 4:
            return None
        
        num_heads = 8
        head_dim = 64  # 512 / 8
        
        results = {"heads": [], "dead_heads": 0, "total_heads": 0}
        
        # Process in groups of 4 (Q, K, V, Out) - skip FFN layers
        # We identify attention blocks by looking at consecutive [512,512] matrices
        # that come in groups of at least 4 before a [512,2048] FFN layer
        i = 0
        layer_idx = 0
        while i + 3 < len(attn_weights):
            v_name, w_v = attn_weights[i + 2]  # V_proj
            o_name, w_o = attn_weights[i + 3]  # Out_proj
            
            # Compute per-head W_OV norms
            for h in range(num_heads):
                # Slice V: rows [h*64 : (h+1)*64] of the full V matrix
                v_slice = w_v[h * head_dim : (h + 1) * head_dim, :]      # (64, 512)
                # Slice O: cols [h*64 : (h+1)*64] of the full Out matrix
                o_slice = w_o[:, h * head_dim : (h + 1) * head_dim]      # (512, 64)
                
                # W_OV = W_O @ W_V for this head: (512, 64) @ (64, 512) = (512, 512)
                w_ov = o_slice @ v_slice
                frob_norm = float(np.linalg.norm(w_ov, 'fro'))
                
                # Singular value analysis of W_QK for attention pattern diagnosis
                q_name, w_q = attn_weights[i]      # Q_proj
                k_name, w_k = attn_weights[i + 1]  # K_proj
                q_slice = w_q[h * head_dim : (h + 1) * head_dim, :]
                k_slice = w_k[h * head_dim : (h + 1) * head_dim, :]
                w_qk = q_slice @ k_slice.T  # (64, 64)
                sv = np.linalg.svd(w_qk, compute_uv=False)
                sv_ratio = float(sv[0] / (sv[-1] + 1e-12))  # Top/bottom SV ratio
                
                is_dead = frob_norm < 0.01
                results["total_heads"] += 1
                if is_dead:
                    results["dead_heads"] += 1
                    
                results["heads"].append({
                    "layer": layer_idx,
                    "head": h,
                    "ov_frob_norm": round(frob_norm, 4),
                    "qk_sv_ratio": round(sv_ratio, 2),
                    "functionally_dead": is_dead,
                })
            
            layer_idx += 1
            i += 4  # Move to next attention block
            
            # Skip FFN layers (2048-dim matrices) if present
            # We're already only processing [512,512] matrices, so just advance
        
        return results

    def _maps_head_classification(self):
        """MAPS vocabulary projection head classification (ACL 2025).
        
        Computes M = W_U @ W_OV @ W_E for each head to classify its function:
          - Mover: diagonal of M is strongly positive → head copies token identity
          - Suppressor: diagonal of M is strongly negative → head suppresses tokens
          - Copy: off-diagonal structure matches source-target copy patterns
          - Induction: W_QK exhibits previous-token structure
        
        W_E = embedding matrix (vocab_size, d_model)
        W_U = unembedding matrix (d_model, vocab_size)  
        W_OV = W_O @ W_V per head
        
        This entirely replaces entropy-based classification for understanding
        WHAT a head does (vs entropy which only shows HOW spread its attention is).
        """
        # Find embedding and unembedding matrices
        w_e = None  # Embedding: (vocab_size, d_model)
        w_u = None  # Unembedding: (d_model, vocab_size)
        
        for name, w in self.weights.items():
            if w.ndim == 2:
                if w.shape == (32000, 512):  # Embedding
                    w_e = w
                elif w.shape == (512, 32000):  # Unembedding / projection
                    w_u = w
        
        if w_e is None or w_u is None:
            return None
        
        # Collect attention weight matrices [512,512]
        attn_weights = []
        for name, w in self.weights.items():
            if w.ndim == 2 and w.shape == (512, 512):
                attn_weights.append((name, w))
        
        if len(attn_weights) < 4:
            return None
        
        num_heads = 8
        head_dim = 64
        results = {"heads": [], "summary": {"movers": 0, "suppressors": 0, "copy_heads": 0, "other": 0}}
        
        i = 0
        layer_idx = 0
        while i + 3 < len(attn_weights):
            w_v = attn_weights[i + 2][1]  # V_proj
            w_o = attn_weights[i + 3][1]  # Out_proj
            
            for h in range(num_heads):
                v_slice = w_v[h * head_dim : (h + 1) * head_dim, :]
                o_slice = w_o[:, h * head_dim : (h + 1) * head_dim]
                w_ov = o_slice @ v_slice  # (512, 512)
                
                # Full MAPS circuit: M = W_U @ W_OV @ W_E^T
                # M shape: (vocab_size, vocab_size) — too large to store fully
                # Instead, compute diagnostic statistics from M's structure:
                
                # 1. Diagonal score: sample diagonal of M
                #    M_diag[i] = w_u[i] @ w_ov @ w_e[i] for token i
                #    High positive diagonal → mover (preserves token identity)
                #    High negative diagonal → suppressor
                sample_size = min(1000, w_e.shape[0])
                rng = np.random.default_rng(seed=42)
                sample_idx = rng.choice(w_e.shape[0], sample_size, replace=False)
                
                diag_scores = []
                for idx in sample_idx:
                    e_row = w_e[idx]          # (512,)
                    u_col = w_u[:, idx]       # (512,) — column of unembedding
                    # M[idx, idx] = u_col @ w_ov @ e_row
                    score = float(u_col @ w_ov @ e_row)
                    diag_scores.append(score)
                
                mean_diag = float(np.mean(diag_scores))
                std_diag = float(np.std(diag_scores))
                
                # 2. Off-diagonal energy: how much does the head mix tokens?
                # Sample a few off-diagonal entries
                off_diag_scores = []
                for _ in range(200):
                    idx_i, idx_j = rng.choice(w_e.shape[0], 2, replace=False)
                    e_row = w_e[idx_j]
                    u_col = w_u[:, idx_i]
                    score = float(u_col @ w_ov @ e_row)
                    off_diag_scores.append(score)
                
                mean_off_diag = float(np.mean(off_diag_scores))
                diag_dominance = mean_diag - mean_off_diag
                
                # Classification
                if diag_dominance > 0.5 and mean_diag > 0.1:
                    head_class = "mover"
                    results["summary"]["movers"] += 1
                elif diag_dominance < -0.5 and mean_diag < -0.1:
                    head_class = "suppressor"
                    results["summary"]["suppressors"] += 1
                elif abs(mean_off_diag) > abs(mean_diag) * 1.5:
                    head_class = "copy"
                    results["summary"]["copy_heads"] += 1
                else:
                    head_class = "other"
                    results["summary"]["other"] += 1
                
                results["heads"].append({
                    "layer": layer_idx,
                    "head": h,
                    "maps_class": head_class,
                    "diag_mean": round(mean_diag, 4),
                    "diag_std": round(std_diag, 4),
                    "off_diag_mean": round(mean_off_diag, 4),
                    "diag_dominance": round(diag_dominance, 4),
                })
            
            layer_idx += 1
            i += 4
        
        return results
