import numpy as np
from typing import Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class SemanticAnalyzer:
    """
    Analyzes representational quality using two complementary methods:
    
    1. Logit Lens (Nostalgebraist 2020): Projects residual stream at each layer
       through the unembedding matrix to see what the model "predicts" at each
       depth. Healthy models show monotonically increasing confidence.
       
    2. Layer-to-layer CKA (Kornblith et al. 2019): Measures representational
       similarity between consecutive layers using Centered Kernel Alignment,
       which is invariant to isotropic scaling and orthogonal transformations.
       
    These replace the previous SBERT random-projection approach, which was
    methodologically unsound (random projections do not preserve semantic
    structure, making cosine similarity in the projected space uninterpretable).
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # SBERT kept as optional for backward compatibility, but no longer
        # used as the primary semantic metric.
        self.sbert = None
        if SentenceTransformer:
            try:
                self.sbert = SentenceTransformer(model_name)
            except Exception:
                pass

    def _linear_cka(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Linear CKA between two activation matrices.
        
        CKA (Centered Kernel Alignment, Kornblith et al. 2019) is the
        established method for comparing neural network layer representations.
        It is invariant to isotropic scaling and orthogonal transformations.
        
        Args:
            X: Activation matrix (n_samples, d1) 
            Y: Activation matrix (n_samples, d2)
        
        Returns:
            CKA similarity in [0, 1]. Values < 0.30 indicate representational
            rupture — the successor layer has discarded the learned feature
            subspace of its predecessor.
        """
        # Center the matrices
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        
        # HSIC with linear kernel: HSIC(K, L) = ||Y^T X||_F^2 / (n-1)^2
        # Simplified: CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        YtX = Y.T @ X
        XtX = X.T @ X
        YtY = Y.T @ Y
        
        numerator = np.linalg.norm(YtX, 'fro') ** 2
        denominator = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')
        
        if denominator < 1e-12:
            return 0.0
        
        return float(np.clip(numerator / denominator, 0.0, 1.0))

    def _logit_lens(self, layer_activation: np.ndarray, 
                     unembedding_matrix: np.ndarray) -> Dict[str, float]:
        """Project residual stream through unembedding to see layer predictions.
        
        Logit Lens (Nostalgebraist 2020): The most widely used single diagnostic
        in production interpretability work. It reveals WHEN a layer actually
        starts predicting meaningful tokens vs passing through noise.
        
        Args:
            layer_activation: (batch, seq, d_model) or (seq, d_model)
            unembedding_matrix: (d_model, vocab_size) — the final projection
        
        Returns:
            Dict with top_1_confidence, top_1_entropy, top_token_id
        """
        if layer_activation.ndim == 3:
            # Mean over batch, take last token position
            act = layer_activation.mean(axis=0)[-1, :]  # (d_model,)
        elif layer_activation.ndim == 2:
            act = layer_activation[-1, :]  # Last position
        else:
            act = layer_activation
            
        # Project through unembedding: logits = act @ W_unembed
        logits = act @ unembedding_matrix  # (vocab_size,)
        
        # Stable softmax
        logits_safe = logits - np.max(logits)
        exp_logits = np.exp(logits_safe)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)
        
        top_id = int(np.argmax(probs))
        top_conf = float(probs[top_id])
        
        # Entropy of prediction distribution
        ent = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        max_ent = np.log(len(probs))
        norm_ent = ent / max_ent if max_ent > 0 else 0.0
        
        return {
            "top_1_confidence": round(top_conf, 6),
            "top_token_id": top_id,
            "normalized_entropy": round(float(norm_ent), 4),
        }

    def analyze(self, activation_tensors: Dict[str, np.ndarray], 
                input_text: str,
                unembedding_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Args:
            activation_tensors: Mapping of layer names to their outputs
            input_text: The original text (kept for backward compatibility)
            unembedding_weights: Optional (d_model, vocab_size) matrix for logit lens
        """
        report = {
            "summary": {"method": "CKA + Logit Lens (2024 standard)"},
            "layer_cka": {},        # CKA similarity to previous layer
            "logit_lens": {},       # Per-layer prediction confidence
            "layer_drift": {},      # Kept for backward compat (now = 1 - CKA)
            "layer_fidelity": {},   # Kept for backward compat
            "issues": []
        }
        
        sorted_names = sorted(activation_tensors.keys())
        prev_act_2d = None
        prev_name = None
        logit_confidences = []
        
        for name in sorted_names:
            act = activation_tensors[name]
            
            # Skip gates, scalars, biases
            if act.shape[-1] < 16:
                continue
                
            # Flatten to 2D: (n_samples, features) for CKA
            if act.ndim >= 3:
                # Reshape (batch, seq, d_model) → (batch*seq, d_model)
                act_2d = act.reshape(-1, act.shape[-1])
            elif act.ndim == 2:
                act_2d = act
            else:
                continue
            
            # Skip near-zero activations (degenerate LayerNorm outputs)
            if np.linalg.norm(act_2d) < 1e-6:
                continue
            
            # ── CKA: Layer-to-layer representational similarity ──
            if prev_act_2d is not None:
                # Subsample if too many tokens (CKA is O(n²) in samples)
                max_samples = 1024
                if act_2d.shape[0] > max_samples:
                    idx = np.linspace(0, act_2d.shape[0] - 1, max_samples, dtype=int)
                    a_sub = act_2d[idx]
                    p_sub = prev_act_2d[idx] if prev_act_2d.shape[0] > max_samples else prev_act_2d
                else:
                    a_sub = act_2d
                    p_sub = prev_act_2d
                    
                # Handle dimension mismatches by truncating to min samples
                min_n = min(a_sub.shape[0], p_sub.shape[0])
                cka_score = self._linear_cka(a_sub[:min_n], p_sub[:min_n])
                
                report["layer_cka"][name] = round(cka_score, 4)
                report["layer_drift"][name] = round(1.0 - cka_score, 4)  # Backward compat
                
                if cka_score < 0.30:
                    report["issues"].append({
                        "severity": "warning",
                        "msg": f"Representational rupture at {name} (CKA={cka_score:.2f} < 0.30). "
                               f"Layer discards predecessor's feature subspace."
                    })
            
            # ── Logit Lens: What does the model predict at this depth? ──
            if unembedding_weights is not None and act.shape[-1] == unembedding_weights.shape[0]:
                lens_result = self._logit_lens(act, unembedding_weights)
                report["logit_lens"][name] = lens_result
                logit_confidences.append(lens_result["top_1_confidence"])
                
                # Backward compat: use logit confidence as "fidelity"
                report["layer_fidelity"][name] = lens_result["top_1_confidence"]
            
            prev_act_2d = act_2d
            prev_name = name
        
        # ── Check for monotonic confidence growth ──
        if len(logit_confidences) >= 3:
            # Healthy: confidence should generally increase through layers
            decreases = 0
            for i in range(1, len(logit_confidences)):
                if logit_confidences[i] < logit_confidences[i-1] * 0.5:
                    decreases += 1
            if decreases > len(logit_confidences) * 0.3:
                report["issues"].append({
                    "severity": "warning",
                    "msg": f"Logit Lens shows non-monotonic confidence: model may not be "
                           f"learning progressively through depth."
                })
        
        return report
