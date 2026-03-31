import numpy as np
from scipy.stats import entropy
from typing import Dict, Any, List

class AttentionAnalyzer:
    """
    Analyzes captured attention maps (softmax outputs) for entropy and patterns.
    """
    
    def analyze(self, attention_tensors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Args:
            attention_tensors: Dict {tensor_name: numpy_array [batch, heads, seq, seq]}
        """
        report = {
            "summary": {
                "total_heads": 0,
                "healthy_heads": 0,
                "head_types": {"local": 0, "global": 0, "sparse": 0, "uniform": 0, "collapsed": 0}
            },
            "head_issues": [],
            "head_stats": {}
        }
        
        for name, attn in attention_tensors.items():
            if len(attn.shape) != 4:
                continue
                
            batch, heads, seq_len, _ = attn.shape
            report["summary"]["total_heads"] += heads
            
            layer_stats = []
            max_entropy = np.log(seq_len) if seq_len > 1 else 1.0
            
            for h in range(heads):
                # Work on a per-head basis, averaged over batch
                # shape: [batch, T, T]
                h_attn = attn[:, h, :, :] 
                
                # 1. Entropy
                # Add epsilon to avoid log(0)
                h_entropy = -(h_attn * np.log(np.clip(h_attn, 1e-12, 1.0))).sum(axis=-1) # [batch, T]
                avg_entropy = np.mean(h_entropy)
                norm_ent = float(avg_entropy / max_entropy)
                
                # 2. Distance (Locality)
                positions = np.arange(seq_len)
                avg_dist = 0.0
                for q_pos in range(seq_len):
                    # distance to query position
                    dist = np.abs(positions - q_pos)
                    # weighted average distance for this query position
                    avg_dist += np.mean((h_attn[:, q_pos, :] * dist).sum(axis=-1))
                avg_dist /= seq_len
                
                # 3. Target Variance (Sparsity check)
                # Find most attended positions
                max_pos = np.argmax(h_attn, axis=-1) # [batch, T]
                target_variance = np.var(max_pos)
                
                # 4. Classification
                if norm_ent > 0.8:
                    htype = "uniform"
                elif norm_ent < 0.25:
                    if target_variance < 0.5:
                        htype = "collapsed"
                    else:
                        htype = "sparse"
                elif avg_dist < seq_len * 0.2:
                    htype = "local"
                else:
                    htype = "global"
                
                report["summary"]["head_types"][htype] += 1
                if htype in ["local", "global", "sparse"]:
                    report["summary"]["healthy_heads"] += 1
                
                stats = {
                    "avg_entropy": float(avg_entropy),
                    "normalized_entropy": float(norm_ent),
                    "avg_distance": float(avg_dist),
                    "target_variance": float(target_variance),
                    "type": htype
                }
                
                if htype == "collapsed":
                    report["head_issues"].append({
                        "severity": "critical",
                        "msg": f"Head {h} in {name} is COLLAPSED (pointing to same token)."
                    })
                elif htype == "uniform":
                    report["head_issues"].append({
                        "severity": "warning",
                        "msg": f"Head {h} in {name} is UNIFORM (zero attention specialization)."
                    })
                
                layer_stats.append(stats)
            
            report["head_stats"][name] = layer_stats
            
        # Calculate health score
        total = report["summary"]["total_heads"]
        if total > 0:
            health_score = (report["summary"]["healthy_heads"] / total) * 100
            report["summary"]["health_score"] = float(health_score)
            
        return report
