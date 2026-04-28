"""
Token Prediction Explainer Module

Explains why the model generates each token:
- Top-k predictions with probabilities
- Attribution to input tokens
- Confidence analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class TokenExplainer:
    """
    Explains token predictions from a Transformer model.
    
    Usage:
        explainer = TokenExplainer(model, tokenizer)
        explanations = explainer.explain_generation(input_text)
        explainer.print_explanation(explanations)
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id('[PAD]')
        self.sos_id = tokenizer.token_to_id('[SOS]')
        self.eos_id = tokenizer.token_to_id('[EOS]')
    
    def get_top_k_predictions(self, logits, k=10):
        """
        Get top-k token predictions from logits.
        
        Args:
            logits: Model output logits (vocab_size,)
            k: Number of top predictions to return
        
        Returns:
            List of (token_id, token_str, probability) tuples
        """
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k)
        
        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token_str = self.tokenizer.decode([idx])
            results.append({
                'token_id': idx,
                'token': token_str,
                'probability': prob,
                'log_prob': np.log(prob + 1e-10),
            })
        
        return results
    
    def explain_single_token(self, encoder_output, encoder_mask, decoder_input, 
                             decoder_mask, position, device):
        """
        Explain prediction at a single position.
        
        Returns detailed analysis of why the model predicted what it did.
        """
        self.model.eval()
        
        with torch.no_grad():
            decoder_output = self.model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            logits = self.model.project(decoder_output)
        
        # Get logits at the specified position
        position_logits = logits[0, position, :]  # (vocab_size,)
        
        # Get top predictions
        top_k = self.get_top_k_predictions(position_logits, k=10)
        
        # Get probability distribution statistics
        probs = F.softmax(position_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_prob = probs.max().item()
        
        # Confidence assessment
        if max_prob > 0.8:
            confidence = "HIGH"
        elif max_prob > 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            'position': position,
            'top_predictions': top_k,
            'entropy': entropy,
            'max_probability': max_prob,
            'confidence': confidence,
            'chosen_token': top_k[0] if top_k else None,
        }
    
    def explain_generation(self, encoder_input, encoder_mask, decoder_input,
                          decoder_mask, device, max_positions=None):
        """
        Explain all token predictions in a generation.
        
        Returns:
            List of explanations for each position
        """
        self.model.eval()
        
        # Get encoder output
        with torch.no_grad():
            encoder_output = self.model.encode(
                encoder_input.to(device), 
                encoder_mask.to(device)
            )
        
        decoder_input = decoder_input.to(device)
        decoder_mask = decoder_mask.to(device)
        
        # Get decoder output
        with torch.no_grad():
            decoder_output = self.model.decode(
                encoder_output, encoder_mask.to(device), decoder_input, decoder_mask
            )
            logits = self.model.project(decoder_output)
        
        # Analyze each position
        explanations = []
        seq_len = decoder_input.size(1)
        
        if max_positions:
            seq_len = min(seq_len, max_positions)
        
        for pos in range(seq_len):
            position_logits = logits[0, pos, :]
            
            top_k = self.get_top_k_predictions(position_logits, k=5)
            probs = F.softmax(position_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            
            # Get actual token at this position (if available in decoder input)
            actual_token_id = decoder_input[0, pos].item()
            actual_token = self.tokenizer.decode([actual_token_id])
            
            explanations.append({
                'position': pos,
                'actual_token': actual_token,
                'actual_token_id': actual_token_id,
                'top_predictions': top_k,
                'entropy': entropy,
                'confidence': 'HIGH' if top_k[0]['probability'] > 0.5 else 'LOW',
            })
        
        return explanations
    
    def compute_input_attribution(self, encoder_input, encoder_mask, decoder_input,
                                  decoder_mask, target_position, device):
        """
        Compute attribution scores showing which input tokens influence
        the prediction at target_position.
        
        Uses gradient-based attribution.
        """
        self.model.train()  # Need gradients
        
        encoder_input = encoder_input.to(device)
        encoder_mask = encoder_mask.to(device)
        decoder_input = decoder_input.to(device)
        decoder_mask = decoder_mask.to(device)
        
        # Enable gradient for embeddings
        encoder_input.requires_grad = False
        
        # Forward pass
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        encoder_output.retain_grad()  # Keep gradient for encoder output
        
        decoder_output = self.model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        logits = self.model.project(decoder_output)
        
        # Get the predicted token at target position
        target_logits = logits[0, target_position, :]
        predicted_token = torch.argmax(target_logits)
        
        # Compute gradient of predicted token probability w.r.t. encoder output
        target_prob = F.softmax(target_logits, dim=-1)[predicted_token]
        target_prob.backward(retain_graph=True)
        
        # Get attribution from encoder output gradient
        if encoder_output.grad is not None:
            # Sum over d_model dimension to get per-token attribution
            attribution = encoder_output.grad[0].abs().sum(dim=-1)  # (seq_len,)
            attribution = attribution / (attribution.sum() + 1e-10)  # Normalize
            attribution = attribution.detach().cpu().numpy()
        else:
            attribution = None
        
        self.model.eval()
        
        return {
            'target_position': target_position,
            'predicted_token': self.tokenizer.decode([predicted_token.item()]),
            'attribution': attribution,
        }
    
    def visualize_token_probabilities(self, explanations, save_path=None):
        """
        Visualize token prediction probabilities across positions.
        """
        positions = [e['position'] for e in explanations]
        top_probs = [e['top_predictions'][0]['probability'] for e in explanations]
        entropies = [e['entropy'] for e in explanations]
        tokens = [e['actual_token'][:10] for e in explanations]  # Truncate long tokens
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Top-1 probability
        colors = ['#2ecc71' if p > 0.5 else '#f39c12' if p > 0.2 else '#e74c3c' for p in top_probs]
        ax1.bar(positions, top_probs, color=colors)
        ax1.set_ylabel('Top-1 Probability')
        ax1.set_title('Token Prediction Confidence')
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='High confidence threshold')
        ax1.legend()
        
        # Entropy (uncertainty)
        ax2.bar(positions, entropies, color='#3498db')
        ax2.set_ylabel('Entropy (Uncertainty)')
        ax2.set_xlabel('Token Position')
        ax2.set_xticks(positions)
        ax2.set_xticklabels(tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def print_explanation(self, explanations, max_positions=20):
        """Print human-readable token explanations."""
        print("=" * 70)
        print("TOKEN PREDICTION EXPLANATION")
        print("=" * 70)
        
        for exp in explanations[:max_positions]:
            pos = exp['position']
            actual = exp['actual_token']
            top = exp['top_predictions'][0]
            conf = exp['confidence']
            
            match = "✓" if actual.strip() == top['token'].strip() else "✗"
            
            print(f"\nPosition {pos}: '{actual}'")
            print(f"  {match} Predicted: '{top['token']}' ({top['probability']:.3f})")
            print(f"  Confidence: {conf} | Entropy: {exp['entropy']:.3f}")
            
            if len(exp['top_predictions']) > 1:
                alternatives = exp['top_predictions'][1:4]
                alt_str = ", ".join([f"'{a['token']}' ({a['probability']:.2f})" for a in alternatives])
                print(f"  Alternatives: {alt_str}")
        
        print("\n" + "=" * 70)


if __name__ == '__main__':
    print("Token Explainer Module Loaded")
    print("Usage:")
    print("  from debugger.token_explainer import TokenExplainer")
    print("  explainer = TokenExplainer(model, tokenizer)")
    print("  explanations = explainer.explain_generation(...)")
