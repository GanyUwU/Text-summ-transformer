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
                
                # 4. Sink Score (Attention to position 0) — Xiao et al. 2023
                # h_attn shape: [batch, Tq, Tk]
                sink_score = np.mean(h_attn[:, :, 0])
                
                # 5. Classification (with sink awareness)
                if sink_score > 0.5:
                    if sink_score > 0.8 and norm_ent < 0.15:
                        htype = "pure_sink"
                    else:
                        htype = "sink"
                elif norm_ent > 0.8:
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
                
                # Initialize missing keys dynamically if needed
                if htype not in report["summary"]["head_types"]:
                    report["summary"]["head_types"][htype] = 0
                    
                report["summary"]["head_types"][htype] += 1
                if htype in ["local", "global", "sparse", "sink", "pure_sink"]:
                    report["summary"]["healthy_heads"] += 1
                
                stats = {
                    "avg_entropy": float(avg_entropy),
                    "normalized_entropy": float(norm_ent),
                    "avg_distance": float(avg_dist),
                    "target_variance": float(target_variance),
                    "sink_score": float(sink_score),
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
    
    def analyze_cross_attention(self, cross_attn_tensors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze cross-attention alignment for summarization quality.
        
        Cross-attention alignment entropy is the single most important diagnostic
        for a summarizer — it reveals whether the decoder is actually grounding
        its generation in relevant source positions, or collapsing to BOS/padding.
        
        This was completely absent from the original suite and is the diagnostic
        most specific to our encoder-decoder pointer-generator architecture.
        
        Args:
            cross_attn_tensors: Dict {name: ndarray [batch, heads, tgt_seq, src_seq]}
                               These are the decoder cross-attention maps.
        """
        report = {
            "summary": {"total_cross_heads": 0, "bos_collapsed": 0, "healthy_alignment": 0},
            "heads": [],
            "issues": []
        }
        
        for name, attn in cross_attn_tensors.items():
            if attn.ndim != 4:
                continue
                
            batch, heads, tgt_len, src_len = attn.shape
            
            for h in range(heads):
                h_attn = attn[:, h, :, :]  # (B, tgt, src)
                report["summary"]["total_cross_heads"] += 1
                
                # 1. BOS collapse: is position 0 dominating across all queries?
                bos_weight = float(np.mean(h_attn[:, :, 0]))
                
                # 2. Alignment entropy: low = focused on specific source tokens (good)
                #    high = attending everywhere uniformly (bad for summarization)
                h_ent = -(h_attn * np.log(np.clip(h_attn, 1e-12, 1.0))).sum(axis=-1)
                avg_entropy = float(np.mean(h_ent))
                max_entropy = np.log(src_len) if src_len > 1 else 1.0
                norm_ent = avg_entropy / max_entropy
                
                # 3. Coverage: what fraction of source tokens get > 5% attention
                #    from at least one decoder position?
                max_attn_per_src = np.max(h_attn.mean(axis=0), axis=0)  # (src,)
                coverage = float(np.mean(max_attn_per_src > 0.05))
                
                # 4. Monotonicity: for a summarizer, cross-attention should
                #    roughly progress through the source (soft monotonic alignment)
                argmax_positions = np.argmax(h_attn.mean(axis=0), axis=-1)  # (tgt,)
                if len(argmax_positions) > 2:
                    # Spearman-like: fraction of query pairs that are monotonically ordered
                    monotonic_pairs = 0
                    total_pairs = 0
                    for i in range(len(argmax_positions) - 1):
                        if argmax_positions[i+1] >= argmax_positions[i]:
                            monotonic_pairs += 1
                        total_pairs += 1
                    monotonicity = monotonic_pairs / max(1, total_pairs)
                else:
                    monotonicity = 0.5
                
                # Classification
                if bos_weight > 0.5:
                    alignment_type = "bos_collapsed"
                    report["summary"]["bos_collapsed"] += 1
                    report["issues"].append({
                        "severity": "critical",
                        "msg": f"Cross-attn head {h} in {name}: BOS collapse "
                               f"(bos_weight={bos_weight:.2f}). Decoder ignoring source content."
                    })
                elif norm_ent > 0.85:
                    alignment_type = "diffuse"
                    report["issues"].append({
                        "severity": "warning",
                        "msg": f"Cross-attn head {h} in {name}: Diffuse attention "
                               f"(η={norm_ent:.2f}). No source grounding."
                    })
                else:
                    alignment_type = "healthy"
                    report["summary"]["healthy_alignment"] += 1
                
                report["heads"].append({
                    "name": name,
                    "head": h,
                    "bos_weight": round(bos_weight, 4),
                    "normalized_entropy": round(float(norm_ent), 4),
                    "coverage": round(coverage, 4),
                    "monotonicity": round(float(monotonicity), 4),
                    "alignment_type": alignment_type,
                })
        
        return report
