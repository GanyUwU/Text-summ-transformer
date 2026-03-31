import numpy as np
from typing import Dict, Any, List, Optional
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None

class SemanticAnalyzer:
    """
    Uses Sentence-BERT to analyze the semantic fidelity of internal representations.
    Tracks how the "meaning" of a sentence changes as it passes through layers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.sbert = None
        if SentenceTransformer:
            try:
                self.sbert = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load SBERT model {model_name}: {e}")
        self._projections = {}

    def _get_projection(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Creates or retrieves a deterministic random projection matrix."""
        key = (in_dim, out_dim)
        if key not in self._projections:
            # Seed based on dimensions to ensure consistency across analysis runs
            rng = np.random.default_rng(seed=in_dim * 1000 + out_dim)
            # Use a semi-orthogonal or simple random matrix
            # For JL lemma preservation, a standard normal matrix scaled by 1/sqrt(out_dim) works
            proj = rng.standard_normal((in_dim, out_dim)) / np.sqrt(out_dim)
            self._projections[key] = proj
        return self._projections[key]

    def _project(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Projects vector to target dimension deterministically."""
        if vec.shape[0] == target_dim:
            return vec
        proj = self._get_projection(vec.shape[0], target_dim)
        return vec @ proj

    def analyze(self, activation_tensors: Dict[str, np.ndarray], input_text: str) -> Dict[str, Any]:
        """
        Args:
            activation_tensors: Mapping of layer names to their outputs (batch, seq, d_model)
            input_text: The original text that generated these activations
        """
        report = {
            "summary": {"sbert_model": self.model_name},
            "layer_fidelity": {}, # Similarity to ground truth (SBERT embedding of text)
            "layer_drift": {},    # Similarity to previous layer
            "issues": []
        }
        
        if not self.sbert:
            report["issues"].append({"severity": "info", "msg": "SBERT not available. Install sentence-transformers for semantic analysis."})
            return report

        # 1. Get Ground Truth Embedding
        # We use SBERT to encode the original text as the "Ideal" representation
        gt_embedding = self.sbert.encode(input_text, convert_to_numpy=True)
        
        prev_embedding = None
        
        # Sort activations by layer index if possible (heuristic: name containing 'layer' followed by numbers)
        sorted_names = sorted(activation_tensors.keys())
        
        for name in sorted_names:
            act = activation_tensors[name]
            # Pooling: [batch, seq, d_model] -> [d_model]
            # ONLY analyze layers with substantial hidden dimension (semantic representations)
            # Skip gates, scalars, and biases (d_model < 16)
            if act.shape[-1] < 16:
                continue
                
            layer_embedding = np.mean(act, axis=tuple(range(act.ndim - 1)))
            layer_norm = np.linalg.norm(layer_embedding)
            
            # Skip near-zero vectors (e.g., zero-meaned LayerNorm outputs with no bias)
            # These are non-semantic and lead to sim=-1.00 math errors.
            if layer_norm < 1e-6:
                continue
            
            # Fidelity: Similarity to original input text (projected if necessary)
            if gt_embedding is not None:
                # Use deterministic projection if dimensions mismatch
                proj_act = self._project(layer_embedding, gt_embedding.shape[0])
                
                dot_gt = np.dot(proj_act, gt_embedding)
                norm_gt = np.linalg.norm(proj_act) * np.linalg.norm(gt_embedding)
                fidelity = float(dot_gt / (norm_gt + 1e-12)) 
                report["layer_fidelity"][name] = np.clip(fidelity, -1.0, 1.0)
                
                if layer_embedding.shape != gt_embedding.shape:
                    msg = f"Fidelity calculated via projection ({layer_embedding.shape[0]} -> {gt_embedding.shape[0]})"
                    if msg not in [i["msg"] for i in report["issues"]]:
                         report["issues"].append({"severity": "info", "msg": msg})

            # Drift: Similarity to previous layer (projected if necessary)
            if prev_embedding is not None:
                # Project smaller to larger or vice-versa? 
                # Let's project prev to current layer's dimension
                proj_prev = self._project(prev_embedding, layer_embedding.shape[0])
                
                eps = 1e-12
                dot = np.dot(layer_embedding, proj_prev)
                norm = np.linalg.norm(layer_embedding) * np.linalg.norm(proj_prev)
                similarity = float(dot / (norm + eps)) 
                similarity = np.clip(similarity, -1.0, 1.0)
                report["layer_drift"][name] = similarity
                
                if similarity < 0.15: # Lower threshold for cross-dim comparisons
                    report["issues"].append({
                        "severity": "warning", 
                        "msg": f"Significant semantic drift at {name} (sim={similarity:.2f})."
                    })
            
            prev_embedding = layer_embedding

        return report
