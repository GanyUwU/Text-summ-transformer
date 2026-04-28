"""
Loss Tracker Module

Tracks and visualizes training loss including:
- Real-time loss curves
- Per-token loss breakdown
- Learning dynamics
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from pathlib import Path


class LossTracker:
    """
    Tracks loss during training and provides detailed analysis.
    
    Usage:
        tracker = LossTracker()
        tracker.add_loss(loss, step)
        tracker.visualize_loss_curve()
    """
    
    def __init__(self, save_path=None):
        self.losses = []
        self.steps = []
        self.epoch_losses = defaultdict(list)
        self.per_token_losses = []
        self.save_path = save_path
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    def add_loss(self, loss, step=None, epoch=None):
        """Record a loss value."""
        loss_val = loss.item() if torch.is_tensor(loss) else loss
        
        if step is None:
            step = len(self.steps)
        
        self.losses.append(loss_val)
        self.steps.append(step)
        
        if epoch is not None:
            self.epoch_losses[epoch].append(loss_val)
    
    def compute_per_token_loss(self, logits, labels, tokenizer, ignore_index=-100):
        """
        Compute loss for each token individually.
        
        Args:
            logits: Model output (batch, seq_len, vocab_size)
            labels: Target tokens (batch, seq_len)
            tokenizer: Tokenizer for decoding
            ignore_index: Index to ignore in loss computation
        
        Returns:
            List of (token, loss) tuples
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Compute per-token cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        
        token_losses = []
        
        for pos in range(seq_len):
            label = labels[0, pos].item()
            
            if label == ignore_index:
                continue
            
            token_log_prob = log_probs[0, pos, label].item()
            token_loss = -token_log_prob
            
            token_str = tokenizer.decode([label])
            
            token_losses.append({
                'position': pos,
                'token': token_str,
                'token_id': label,
                'loss': token_loss,
                'probability': np.exp(token_log_prob),
            })
        
        self.per_token_losses.append(token_losses)
        return token_losses
    
    def get_statistics(self):
        """Get loss statistics."""
        if not self.losses:
            return {}
        
        recent_losses = self.losses[-100:] if len(self.losses) > 100 else self.losses
        
        return {
            'current': self.losses[-1],
            'mean': np.mean(self.losses),
            'std': np.std(self.losses),
            'min': np.min(self.losses),
            'max': np.max(self.losses),
            'recent_mean': np.mean(recent_losses),
            'recent_std': np.std(recent_losses),
            'total_steps': len(self.losses),
            'trend': 'decreasing' if len(self.losses) > 10 and np.mean(self.losses[-10:]) < np.mean(self.losses[:10]) else 'increasing' if len(self.losses) > 10 else 'unknown',
        }
    
    def visualize_loss_curve(self, window_size=50, save_path=None):
        """
        Plot the loss curve with smoothing.
        """
        if not self.losses:
            print("No losses recorded yet")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Raw loss
        ax.plot(self.steps, self.losses, alpha=0.3, color='blue', label='Raw Loss')
        
        # Smoothed loss (moving average)
        if len(self.losses) > window_size:
            smoothed = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = self.steps[window_size-1:]
            ax.plot(smoothed_steps, smoothed, color='blue', linewidth=2, label=f'Smoothed (window={window_size})')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        stats = self.get_statistics()
        stats_text = f"Current: {stats['current']:.4f}\nMin: {stats['min']:.4f}\nTrend: {stats['trend']}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def visualize_per_token_loss(self, token_losses=None, save_path=None):
        """
        Visualize per-token loss for a single example.
        """
        if token_losses is None:
            if not self.per_token_losses:
                print("No per-token losses recorded")
                return
            token_losses = self.per_token_losses[-1]
        
        positions = [t['position'] for t in token_losses]
        losses = [t['loss'] for t in token_losses]
        tokens = [t['token'][:8] for t in token_losses]  # Truncate
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Color by loss magnitude
        colors = ['#e74c3c' if l > 5 else '#f39c12' if l > 2 else '#2ecc71' for l in losses]
        
        bars = ax.bar(range(len(positions)), losses, color=colors)
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_xlabel('Token')
        ax.set_ylabel('Loss (Cross-Entropy)')
        ax.set_title('Per-Token Loss Analysis')
        
        # Add threshold lines
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='High loss (>5)')
        ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5, label='Medium loss (>2)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def visualize_epoch_comparison(self, save_path=None):
        """Compare loss across epochs."""
        if not self.epoch_losses:
            print("No epoch losses recorded")
            return
        
        epochs = sorted(self.epoch_losses.keys())
        means = [np.mean(self.epoch_losses[e]) for e in epochs]
        stds = [np.std(self.epoch_losses[e]) for e in epochs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(epochs, means, yerr=stds, marker='o', capsize=5, 
                   color='blue', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Loss')
        ax.set_title('Loss by Epoch')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()
        return fig
    
    def print_summary(self):
        """Print loss summary."""
        stats = self.get_statistics()
        
        print("=" * 50)
        print("LOSS TRACKING SUMMARY")
        print("=" * 50)
        print(f"Total steps: {stats.get('total_steps', 0)}")
        print(f"Current loss: {stats.get('current', 'N/A'):.4f}" if stats.get('current') else "No data")
        print(f"Mean loss: {stats.get('mean', 'N/A'):.4f}" if stats.get('mean') else "")
        print(f"Min loss: {stats.get('min', 'N/A'):.4f}" if stats.get('min') else "")
        print(f"Max loss: {stats.get('max', 'N/A'):.4f}" if stats.get('max') else "")
        print(f"Recent mean (last 100): {stats.get('recent_mean', 'N/A'):.4f}" if stats.get('recent_mean') else "")
        print(f"Trend: {stats.get('trend', 'unknown')}")
        print("=" * 50)
    
    def save(self, filepath=None):
        """Save loss history to file."""
        filepath = filepath or self.save_path
        if not filepath:
            print("No save path specified")
            return
        
        data = {
            'losses': self.losses,
            'steps': self.steps,
            'epoch_losses': dict(self.epoch_losses),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Loss history saved to {filepath}")
    
    def load(self, filepath=None):
        """Load loss history from file."""
        filepath = filepath or self.save_path
        if not filepath or not Path(filepath).exists():
            print("File not found")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.losses = data['losses']
        self.steps = data['steps']
        self.epoch_losses = defaultdict(list, {int(k): v for k, v in data['epoch_losses'].items()})
        
        print(f"Loaded {len(self.losses)} loss values")


if __name__ == '__main__':
    print("Loss Tracker Module Loaded")
    print("Usage:")
    print("  from debugger.loss_tracker import LossTracker")
    print("  tracker = LossTracker()")
    print("  tracker.add_loss(loss, step)")
    print("  tracker.visualize_loss_curve()")
