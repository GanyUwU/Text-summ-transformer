"""
Gradient Analyzer Module

Analyzes gradient flow through the Transformer to identify:
- Vanishing/exploding gradients
- Which layers are learning most/least
- Parameter update magnitudes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class GradientAnalyzer:
    """
    Analyzes gradient flow through a Transformer model.
    
    Usage:
        analyzer = GradientAnalyzer(model)
        analyzer.compute_gradients(loss)
        analyzer.visualize_gradient_flow()
    """
    
    def __init__(self, model):
        self.model = model
        self.gradient_stats = {}
        self.gradient_history = defaultdict(list)
    
    def compute_gradients(self, loss):
        """
        Compute and store gradient statistics after backward pass.
        
        Args:
            loss: The loss tensor (backward will be called if needed)
        """
        if loss.grad_fn is not None and not any(p.grad is not None for p in self.model.parameters()):
            loss.backward(retain_graph=True)
        
        self.gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu()
                
                stats = {
                    'mean': float(grad.mean()),
                    'std': float(grad.std()),
                    'norm': float(grad.norm()),
                    'max': float(grad.abs().max()),
                    'min': float(grad.abs().min()),
                    'shape': list(grad.shape),
                    'num_zeros': int((grad == 0).sum()),
                    'total_elements': int(grad.numel()),
                }
                
                self.gradient_stats[name] = stats
                self.gradient_history[name].append(stats['norm'])
        
        return self.gradient_stats
    
    def get_layer_gradients(self):
        """
        Aggregate gradients by layer for easier analysis.
        
        Returns:
            dict: Layer-wise gradient statistics
        """
        layer_stats = defaultdict(lambda: {'norms': [], 'means': []})
        
        for name, stats in self.gradient_stats.items():
            # Extract layer info from parameter name
            parts = name.split('.')
            if 'encoder' in name:
                if 'layers' in name:
                    layer_idx = parts[parts.index('layers') + 1]
                    layer_name = f"encoder.layer_{layer_idx}"
                else:
                    layer_name = "encoder.other"
            elif 'decoder' in name:
                if 'layers' in name:
                    layer_idx = parts[parts.index('layers') + 1]
                    layer_name = f"decoder.layer_{layer_idx}"
                else:
                    layer_name = "decoder.other"
            elif 'embed' in name:
                layer_name = "embedding"
            elif 'projection' in name:
                layer_name = "projection"
            else:
                layer_name = "other"
            
            layer_stats[layer_name]['norms'].append(stats['norm'])
            layer_stats[layer_name]['means'].append(stats['mean'])
        
        # Compute averages
        result = {}
        for layer, data in layer_stats.items():
            result[layer] = {
                'avg_norm': np.mean(data['norms']),
                'max_norm': np.max(data['norms']),
                'avg_mean': np.mean(data['means']),
            }
        
        return result
    
    def check_gradient_health(self):
        """
        Check for gradient issues and return diagnostic report.
        
        Returns:
            dict: Diagnostic information about gradient health
        """
        issues = []
        warnings = []
        
        total_norm = sum(s['norm'] for s in self.gradient_stats.values())
        
        # Check for vanishing gradients
        if total_norm < 1e-7:
            issues.append("CRITICAL: Vanishing gradients detected (total norm < 1e-7)")
        elif total_norm < 1e-4:
            warnings.append("WARNING: Very small gradients (total norm < 1e-4)")
        
        # Check for exploding gradients
        if total_norm > 1000:
            issues.append("CRITICAL: Exploding gradients detected (total norm > 1000)")
        elif total_norm > 100:
            warnings.append("WARNING: Large gradients (total norm > 100)")
        
        # Check for dead parameters (zero gradients)
        dead_params = []
        for name, stats in self.gradient_stats.items():
            if stats['norm'] == 0:
                dead_params.append(name)
        
        if dead_params:
            warnings.append(f"WARNING: {len(dead_params)} parameters have zero gradients")
        
        # Check layer-wise gradient flow
        layer_grads = self.get_layer_gradients()
        
        # Sort by layer depth
        encoder_layers = sorted([k for k in layer_grads if 'encoder.layer' in k])
        decoder_layers = sorted([k for k in layer_grads if 'decoder.layer' in k])
        
        # Check for gradient degradation through layers
        if len(encoder_layers) >= 2:
            first_layer_norm = layer_grads[encoder_layers[0]]['avg_norm']
            last_layer_norm = layer_grads[encoder_layers[-1]]['avg_norm']
            if first_layer_norm > 0 and last_layer_norm / first_layer_norm < 0.01:
                warnings.append("WARNING: Gradient degradation in encoder layers")
        
        return {
            'total_gradient_norm': total_norm,
            'issues': issues,
            'warnings': warnings,
            'dead_parameters': dead_params,
            'layer_gradients': layer_grads,
            'healthy': len(issues) == 0,
        }
    
    def visualize_gradient_flow(self, save_path=None):
        """
        Visualize gradient flow through the network as a bar chart.
        """
        layer_grads = self.get_layer_gradients()
        
        # Sort layers logically
        def sort_key(name):
            if 'embedding' in name:
                return (0, name)
            elif 'encoder' in name:
                return (1, name)
            elif 'decoder' in name:
                return (2, name)
            elif 'projection' in name:
                return (3, name)
            return (4, name)
        
        sorted_layers = sorted(layer_grads.keys(), key=sort_key)
        norms = [layer_grads[l]['avg_norm'] for l in sorted_layers]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#2ecc71' if n > 0.01 else '#e74c3c' if n < 0.001 else '#f39c12' for n in norms]
        
        bars = ax.barh(range(len(sorted_layers)), norms, color=colors)
        ax.set_yticks(range(len(sorted_layers)))
        ax.set_yticklabels(sorted_layers)
        ax.set_xlabel('Average Gradient Norm')
        ax.set_title('Gradient Flow Through Network')
        ax.set_xscale('log')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Healthy (>0.01)'),
            Patch(facecolor='#f39c12', label='Weak (0.001-0.01)'),
            Patch(facecolor='#e74c3c', label='Vanishing (<0.001)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved gradient flow visualization to {save_path}")
        
        plt.show()
        return fig
    
    def visualize_gradient_history(self, param_names=None, save_path=None):
        """
        Plot gradient norm history over training steps.
        
        Args:
            param_names: List of parameter names to plot (None = aggregate)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if param_names is None:
            # Aggregate all gradients
            total_norms = []
            max_len = max(len(h) for h in self.gradient_history.values()) if self.gradient_history else 0
            
            for step in range(max_len):
                step_norm = 0
                for name, history in self.gradient_history.items():
                    if step < len(history):
                        step_norm += history[step] ** 2
                total_norms.append(np.sqrt(step_norm))
            
            ax.plot(total_norms, label='Total Gradient Norm')
        else:
            for name in param_names:
                if name in self.gradient_history:
                    ax.plot(self.gradient_history[name], label=name)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()
        return fig
    
    def print_gradient_report(self):
        """Print a human-readable gradient analysis report."""
        health = self.check_gradient_health()
        
        print("=" * 60)
        print("GRADIENT ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\nTotal Gradient Norm: {health['total_gradient_norm']:.6f}")
        print(f"Status: {'✓ HEALTHY' if health['healthy'] else '✗ ISSUES DETECTED'}")
        
        if health['issues']:
            print("\n🔴 CRITICAL ISSUES:")
            for issue in health['issues']:
                print(f"  • {issue}")
        
        if health['warnings']:
            print("\n🟡 WARNINGS:")
            for warning in health['warnings']:
                print(f"  • {warning}")
        
        print("\n📊 LAYER-WISE GRADIENTS:")
        for layer, stats in sorted(health['layer_gradients'].items()):
            norm = stats['avg_norm']
            status = '🟢' if norm > 0.01 else '🔴' if norm < 0.001 else '🟡'
            print(f"  {status} {layer}: {norm:.6f}")
        
        if health['dead_parameters']:
            print(f"\n⚠️  Dead Parameters ({len(health['dead_parameters'])} total):")
            for name in health['dead_parameters'][:5]:  # Show first 5
                print(f"  • {name}")
            if len(health['dead_parameters']) > 5:
                print(f"  ... and {len(health['dead_parameters']) - 5} more")
        
        print("=" * 60)


if __name__ == '__main__':
    print("Gradient Analyzer Module Loaded")
    print("Usage:")
    print("  from debugger.gradient_analyzer import GradientAnalyzer")
    print("  analyzer = GradientAnalyzer(model)")
    print("  analyzer.compute_gradients(loss)")
    print("  analyzer.print_gradient_report()")
