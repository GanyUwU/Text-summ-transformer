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
        
        return report
