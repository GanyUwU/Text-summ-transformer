import numpy as np
from typing import Dict, Any

class ActivationAnalyzer:
    """
    Analyzes activation maps (e.g. outputs of ReLU/GELU) for sparsity and variance.
    """
    
    def analyze(self, activation_tensors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Args:
            activation_tensors: Dict {tensor_name: numpy_array}
        """
        report = {
            "summary": {"total_activations": len(activation_tensors), "issues_count": 0},
            "layer_stats": {},
            "issues": []
        }
        
        variances = []
        
        for name, act in activation_tensors.items():
            # 1. Numerical Stability
            if np.any(np.isnan(act)) or np.any(np.isinf(act)):
                report["issues"].append({
                    "severity": "critical", 
                    "msg": f"Activation {name} contains NaNs or Infs!"
                })
                report["summary"]["issues_count"] += 1
                continue

            # 2. Per-Neuron Dead Check (Michel et al. 2019)
            # A neuron is dead only if its max absolute activation across ALL
            # batch/sequence positions is near-zero. Per-element sparsity is wrong
            # because a neuron that fires on even one example is alive.
            if act.ndim >= 2:
                # Reduce all axes except the last (neuron/feature dim)
                axes_to_reduce = tuple(range(act.ndim - 1))
                max_abs_per_neuron = np.max(np.abs(act), axis=axes_to_reduce)
                sparsity = float(np.mean(max_abs_per_neuron < 1e-6))
            else:
                sparsity = np.mean(np.abs(act) < 1e-6)
            
            # 3. Variance & Saturation
            variance = np.var(act)
            variances.append(variance)
            max_val = np.max(np.abs(act))
            
            stats = {
                "sparsity": float(sparsity),
                "variance": float(variance),
                "mean": float(np.mean(act)),
                "max": float(max_val)
            }
            report["layer_stats"][name] = stats
            
            if sparsity > 0.9:
                report["issues"].append({
                    "severity": "warning", 
                    "msg": f"Activation {name} is {sparsity*100:.1f}% sparse (mostly dead)."
                })
                report["summary"]["issues_count"] += 1
            
            # Detect saturation (Values becoming too large and unstable)
            if max_val > 20.0:
                 report["issues"].append({
                    "severity": "warning", 
                    "msg": f"Activation {name} is SATURATING (max abs={max_val:.2f})."
                })
                 report["summary"]["issues_count"] += 1
        
        # 4. Vanishing Signal check
        if len(variances) > 1:
            start_var = variances[0]
            end_var = variances[-1]
            if start_var > 1e-6 and (end_var / start_var) < 0.01:
                  report["issues"].append({
                    "severity": "critical", 
                    "msg": "Vanishing signal detected through layers."
                })
                  report["summary"]["issues_count"] += 1

        return report
