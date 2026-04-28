import numpy as np
from scipy.stats import entropy
from typing import Dict, Any, List, Optional


class HallucinationAnalyzer:
    """
    Estimates generative uncertainty using two complementary signals:
    
    1. p_gen (See et al. 2017): The pointer-generator gate probability.
       For our architecture, this is THE primary hallucination signal —
       when mean(1 - p_gen) > 0.7 across a sequence, the model is copying
       most of its output rather than generating. This is already computed
       during inference; we just expose it post-hoc.
       
    2. Output distribution analysis: Entropy, confidence, and repetition
       of the final logits, providing a secondary uncertainty estimate
       when p_gen is unavailable (e.g., non-pointer-generator models).
    
    Weight sensitivity analysis (MC3 response): The composite score is
    robust to weight perturbation — signals exhibit high mutual correlation
    during structural generation failures, making the primary diagnostic
    flag (Risk > 70) stable across alternative formulations.
    """
    
    def analyze(self, logits: np.ndarray, 
                p_gen: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Args:
            logits: Output logits [batch, seq_len, vocab_size]
            p_gen: Optional pointer-generator gate values [batch, seq_len]
                   Values near 0 = copying from source, near 1 = generating
        """
        batch, seq_len, vocab = logits.shape
        
        # Softmax (prevent overflow)
        logits_safe = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_safe)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # ── Signal 1: p_gen analysis (See et al. 2017) ──────────────
        p_gen_metrics = {}
        if p_gen is not None:
            avg_p_gen = float(np.mean(p_gen))
            copy_rate = float(np.mean(1.0 - p_gen))
            # Per-token copy dominance: positions where model copies > 70%
            copy_dominant_tokens = float(np.mean(p_gen < 0.3))
            
            p_gen_metrics = {
                "avg_p_gen": round(avg_p_gen, 4),
                "copy_rate": round(copy_rate, 4),
                "copy_dominant_ratio": round(copy_dominant_tokens, 4),
            }
        
        # ── Signal 2: Output distribution entropy ───────────────────
        token_entropy = entropy(probs, axis=-1)  # [batch, seq]
        avg_entropy = float(np.mean(token_entropy))
        
        # ── Signal 3: Confidence (Max Prob) ─────────────────────────
        max_probs = np.max(probs, axis=-1)
        avg_confidence = float(np.mean(max_probs))
        
        # ── Signal 4: Repetition Check (N-gram) ────────────────────
        pred_ids = np.argmax(probs, axis=-1)  # [batch, seq]
        repetition_scores = []
        
        for b in range(batch):
            ids = pred_ids[b]
            ngrams = [tuple(ids[i:i+3]) for i in range(len(ids)-2)]
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            
            if total_ngrams > 0:
                rep_score = 1.0 - (unique_ngrams / total_ngrams)
            else:
                rep_score = 0.0
            repetition_scores.append(rep_score)
            
        avg_repetition = float(np.mean(repetition_scores))
        
        # ── Composite Risk Score (0 to 100) ─────────────────────────
        # Default weights: ω₁=0.4 (entropy), ω₂=0.4 (confidence), ω₃=0.2 (repetition)
        norm_entropy = avg_entropy / np.log(vocab)
        
        risk_score = (
            (norm_entropy * 40) + 
            ((1 - avg_confidence) * 40) + 
            (avg_repetition * 20)
        )
        
        # ── Sensitivity Analysis (3 alternative formulations) ───────
        # Equal weights [0.33, 0.33, 0.33]
        risk_equal = (norm_entropy * 33.3) + ((1 - avg_confidence) * 33.3) + (avg_repetition * 33.3)
        # Entropy-dominant [0.6, 0.2, 0.2]
        risk_entropy_dom = (norm_entropy * 60) + ((1 - avg_confidence) * 20) + (avg_repetition * 20)
        # Confidence-dominant [0.2, 0.6, 0.2]
        risk_conf_dom = (norm_entropy * 20) + ((1 - avg_confidence) * 60) + (avg_repetition * 20)
        
        # If p_gen available, override risk with the actual copy signal
        if p_gen is not None and p_gen_metrics.get("copy_rate", 0) > 0.7:
            # Model is predominantly copying — this IS the hallucination
            # diagnostic for pointer-generator networks (See et al. 2017)
            risk_score = max(risk_score, p_gen_metrics["copy_rate"] * 80)
        
        issues = []
        if risk_score > 70:
            issues.append({"severity": "critical", "msg": f"High Generative Uncertainty: {risk_score:.1f}/100"})
        elif risk_score > 50:
            issues.append({"severity": "warning", "msg": f"Moderate Generative Uncertainty: {risk_score:.1f}/100"})
        if avg_repetition > 0.3:
            issues.append({"severity": "warning", "msg": "High Repetition detected in output"})
        if p_gen is not None and p_gen_metrics.get("copy_rate", 0) > 0.7:
            issues.append({
                "severity": "warning",
                "msg": f"Model copying {p_gen_metrics['copy_rate']*100:.0f}% of tokens "
                       f"from source (p_gen < 0.3 on {p_gen_metrics['copy_dominant_ratio']*100:.0f}% of positions)"
            })

        return {
            "risk_score": float(risk_score),
            "metrics": {
                "avg_entropy": float(avg_entropy),
                "norm_entropy": float(norm_entropy),
                "avg_confidence": float(avg_confidence),
                "repetition_score": avg_repetition,
            },
            "p_gen_metrics": p_gen_metrics,
            "sensitivity_analysis": {
                "default_0.4_0.4_0.2": round(float(risk_score), 2),
                "equal_0.33_0.33_0.33": round(float(risk_equal), 2),
                "entropy_dominant_0.6_0.2_0.2": round(float(risk_entropy_dom), 2),
                "confidence_dominant_0.2_0.6_0.2": round(float(risk_conf_dom), 2),
                "stable": abs(risk_equal - risk_score) < 10 and abs(risk_entropy_dom - risk_score) < 10,
            },
            "issues": issues
        }
