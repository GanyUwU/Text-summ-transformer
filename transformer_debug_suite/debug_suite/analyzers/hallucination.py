import numpy as np
from scipy.stats import entropy
from typing import Dict, Any, List

class HallucinationAnalyzer:
    """
    Estimates hallucination risk based on output confidence, entropy, and repetition.
    """
    
    def analyze(self, logits: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        # logits shape: [batch, seq_len, vocab_size]
        batch, seq_len, vocab = logits.shape
        
        # Softmax
        # prevent overflow
        logits_safe = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_safe)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # 1. Entropy
        # Higher entropy = model is unsure = high hallucination risk
        token_entropy = entropy(probs, axis=-1) # [batch, seq]
        avg_entropy = np.mean(token_entropy)
        
        # 2. Confidence (Max Prob)
        # Lower confidence = risk
        max_probs = np.max(probs, axis=-1)
        avg_confidence = np.mean(max_probs)
        
        # 3. Repetition Check (N-gram)
        # We need predicted tokens for this
        pred_ids = np.argmax(probs, axis=-1) # [batch, seq]
        repetition_scores = []
        
        for b in range(batch):
            ids = pred_ids[b]
            # Simple check: sliding window 3-grams
            ngrams = [tuple(ids[i:i+3]) for i in range(len(ids)-2)]
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            
            if total_ngrams > 0:
                rep_score = 1.0 - (unique_ngrams / total_ngrams)
            else:
                rep_score = 0.0
            repetition_scores.append(rep_score)
            
        avg_repetition = float(np.mean(repetition_scores))
        
        # Risk Score Calculation (0 to 100)
        # Weights: Entropy (40%), Confidence (40%), Repetition (20%)
        # Normalize entropy: max entropy ~ log(vocab)
        norm_entropy = avg_entropy / np.log(vocab)
        
        risk_score = (
            (norm_entropy * 40) + 
            ((1 - avg_confidence) * 40) + 
            (avg_repetition * 20)
        )
        
        issues = []
        if risk_score > 50:
            issues.append({"severity": "warning", "msg": f"High Hallucination Risk: {risk_score:.1f}/100"})
        if avg_repetition > 0.3:
            issues.append({"severity": "warning", "msg": "High Repetition detected in output"})

        return {
            "risk_score": float(risk_score),
            "metrics": {
                "avg_entropy": float(avg_entropy),
                "avg_confidence": float(avg_confidence),
                "repetition_score": avg_repetition
            },
            "issues": issues
        }
