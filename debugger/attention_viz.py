"""
Attention Visualization Module

Extracts and visualizes attention patterns from all layers and heads
of the Transformer model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AttentionExtractor:
    """
    Extracts attention weights from a Transformer model.
    
    Usage:
        extractor = AttentionExtractor(model)
        attention_data = extractor.extract(encoder_input, decoder_input, ...)
        extractor.visualize_attention(attention_data, tokenizer)
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        self.hooks = []
        
        # Hook into encoder self-attention
        for i, layer in enumerate(self.model.encoder.layers):
            hook = layer.self_attention_block.register_forward_hook(
                self._make_hook(f'encoder_layer_{i}_self_attn')
            )
            self.hooks.append(hook)
        
        # Hook into decoder self-attention and cross-attention
        for i, layer in enumerate(self.model.decoder.layers):
            # Self-attention
            hook1 = layer.self_attention_block.register_forward_hook(
                self._make_hook(f'decoder_layer_{i}_self_attn')
            )
            self.hooks.append(hook1)
            
            # Cross-attention (encoder-decoder attention)
            hook2 = layer.cross_attention_block.register_forward_hook(
                self._make_hook(f'decoder_layer_{i}_cross_attn')
            )
            self.hooks.append(hook2)
    
    def _make_hook(self, name):
        """Create a hook function that stores attention weights"""
        def hook(module, input, output):
            # MultiHeadAttention stores attention_scores as an attribute
            if hasattr(module, 'attention_scores'):
                self.attention_weights[name] = module.attention_scores.detach().cpu()
        return hook
    
    def extract(self, encoder_input, encoder_mask, decoder_input, decoder_mask, device):
        """
        Run forward pass and extract all attention weights.
        
        Returns:
            dict: Attention weights for each layer/head
        """
        self.attention_weights = {}
        
        with torch.no_grad():
            encoder_output = self.model.encode(encoder_input.to(device), encoder_mask.to(device))
            decoder_output = self.model.decode(
                encoder_output, 
                encoder_mask.to(device), 
                decoder_input.to(device), 
                decoder_mask.to(device)
            )
        
        return self.attention_weights.copy()
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_attention(self, attention_data, src_tokens, tgt_tokens, 
                           layer_idx=0, head_idx=0, attn_type='cross',
                           save_path=None):
        """
        Visualize attention weights as a heatmap.
        
        Args:
            attention_data: Output from extract()
            src_tokens: List of source token strings
            tgt_tokens: List of target token strings
            layer_idx: Which layer to visualize
            head_idx: Which attention head to visualize
            attn_type: 'self' for self-attention, 'cross' for cross-attention
            save_path: Path to save figure (optional)
        """
        if attn_type == 'cross':
            key = f'decoder_layer_{layer_idx}_cross_attn'
        elif attn_type == 'encoder_self':
            key = f'encoder_layer_{layer_idx}_self_attn'
        else:
            key = f'decoder_layer_{layer_idx}_self_attn'
        
        if key not in attention_data:
            print(f"Attention key '{key}' not found. Available: {list(attention_data.keys())}")
            return
        
        # Shape: (batch, heads, seq_len_q, seq_len_k)
        attn = attention_data[key][0, head_idx].numpy()  # First batch item, specified head
        
        # Trim to actual token lengths
        attn = attn[:len(tgt_tokens), :len(src_tokens)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            attn,
            xticklabels=src_tokens,
            yticklabels=tgt_tokens,
            cmap='Blues',
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_xlabel('Source Tokens (Encoder)')
        ax.set_ylabel('Target Tokens (Decoder)')
        ax.set_title(f'{attn_type.title()} Attention - Layer {layer_idx}, Head {head_idx}')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
        
        plt.show()
        return fig
    
    def visualize_all_heads(self, attention_data, src_tokens, tgt_tokens,
                           layer_idx=0, attn_type='cross', save_path=None):
        """
        Visualize attention from all heads in a single layer.
        """
        if attn_type == 'cross':
            key = f'decoder_layer_{layer_idx}_cross_attn'
        else:
            key = f'decoder_layer_{layer_idx}_self_attn'
        
        if key not in attention_data:
            print(f"Key not found: {key}")
            return
        
        attn = attention_data[key][0].numpy()  # (heads, seq_q, seq_k)
        num_heads = attn.shape[0]
        
        # Create subplot grid
        cols = 4
        rows = (num_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            head_attn = attn[head_idx, :len(tgt_tokens), :len(src_tokens)]
            
            sns.heatmap(
                head_attn,
                ax=ax,
                cmap='Blues',
                cbar=False,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(f'Head {head_idx}')
        
        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{attn_type.title()} Attention - Layer {layer_idx} - All Heads')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def get_attention_summary(self, attention_data):
        """
        Get summary statistics for all attention patterns.
        
        Returns:
            dict: Statistics for each attention layer
        """
        summary = {}
        
        for key, attn in attention_data.items():
            # attn shape: (batch, heads, seq_q, seq_k)
            attn_np = attn.numpy()
            
            summary[key] = {
                'shape': attn_np.shape,
                'mean': float(attn_np.mean()),
                'std': float(attn_np.std()),
                'max': float(attn_np.max()),
                'min': float(attn_np.min()),
                'entropy': float(-np.sum(attn_np * np.log(attn_np + 1e-10)) / attn_np.size),
            }
        
        return summary


def demo_attention_extraction():
    """Demo function to test attention extraction"""
    print("Attention Visualization Module Loaded")
    print("Usage:")
    print("  from debugger.attention_viz import AttentionExtractor")
    print("  extractor = AttentionExtractor(model)")
    print("  attn_data = extractor.extract(enc_input, enc_mask, dec_input, dec_mask, device)")
    print("  extractor.visualize_attention(attn_data, src_tokens, tgt_tokens)")


if __name__ == '__main__':
    demo_attention_extraction()
