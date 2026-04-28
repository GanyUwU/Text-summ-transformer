"""
Transformer Debugger Dashboard

Unified interface for all debugging capabilities:
- Attention visualization
- Gradient analysis
- Loss tracking
- Token prediction explanation
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from debugger.attention_viz import AttentionExtractor
from debugger.gradient_analyzer import GradientAnalyzer
from debugger.token_explainer import TokenExplainer
from debugger.loss_tracker import LossTracker


class TransformerDebugger:
    """
    Unified debugging interface for Transformer models.
    
    Usage:
        debugger = TransformerDebugger(model, tokenizer)
        
        # During training
        debugger.track_training_step(loss, logits, labels, step)
        
        # After training
        debugger.analyze_generation(input_text)
        debugger.generate_report()
    """
    
    def __init__(self, model, tokenizer, device=None, output_dir='debug_output'):
        """
        Initialize the debugger with a model and tokenizer.
        
        Args:
            model: Transformer model
            tokenizer: Tokenizer instance
            device: Torch device (auto-detected if None)
            output_dir: Directory for saving visualizations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.attention_extractor = AttentionExtractor(model)
        self.gradient_analyzer = GradientAnalyzer(model)
        self.token_explainer = TokenExplainer(model, tokenizer)
        self.loss_tracker = LossTracker()
        
        print(f"TransformerDebugger initialized")
        print(f"Output directory: {self.output_dir}")
    
    def track_training_step(self, loss, logits=None, labels=None, step=None, epoch=None):
        """
        Track a single training step.
        
        Call this after loss.backward() to capture gradient info.
        """
        # Track loss
        self.loss_tracker.add_loss(loss, step=step, epoch=epoch)
        
        # Compute gradient statistics
        self.gradient_analyzer.compute_gradients(loss)
        
        # Track per-token loss if labels provided
        if logits is not None and labels is not None:
            self.loss_tracker.compute_per_token_loss(logits, labels, self.tokenizer)
    
    def analyze_batch(self, batch, verbose=True):
        """
        Comprehensive analysis of a single batch.
        
        Args:
            batch: Dictionary with encoder_input, decoder_input, etc.
            verbose: Print detailed analysis
        
        Returns:
            Dictionary with all analysis results
        """
        # Move to device
        encoder_input = batch['encoder_input'].to(self.device)
        decoder_input = batch['decoder_input'].to(self.device)
        encoder_mask = batch['encoder_mask'].to(self.device)
        decoder_mask = batch['decoder_mask'].to(self.device)
        
        results = {}
        
        # 1. Extract attention patterns
        if verbose:
            print("\n📊 Extracting attention patterns...")
        results['attention'] = self.attention_extractor.extract(
            encoder_input, encoder_mask, decoder_input, decoder_mask, self.device
        )
        
        # 2. Get attention summary
        results['attention_summary'] = self.attention_extractor.get_attention_summary(results['attention'])
        
        # 3. Explain token predictions
        if verbose:
            print("🔍 Analyzing token predictions...")
        results['token_explanations'] = self.token_explainer.explain_generation(
            encoder_input, encoder_mask, decoder_input, decoder_mask, 
            self.device, max_positions=20
        )
        
        # 4. Gradient analysis (if training mode)
        if self.model.training:
            if verbose:
                print("📈 Analyzing gradients...")
            results['gradient_health'] = self.gradient_analyzer.check_gradient_health()
        
        if verbose:
            print("✅ Analysis complete\n")
        
        return results
    
    def visualize_all(self, batch, analysis_results=None, src_text=None, tgt_text=None):
        """
        Generate all visualizations for a batch.
        
        Args:
            batch: Input batch dictionary
            analysis_results: Pre-computed results (optional)
            src_text: Source text for labeling (optional)
            tgt_text: Target text for labeling (optional)
        """
        if analysis_results is None:
            analysis_results = self.analyze_batch(batch, verbose=False)
        
        # Decode tokens for visualization
        enc_input = batch['encoder_input'][0].tolist()
        dec_input = batch['decoder_input'][0].tolist()
        
        src_tokens = [self.tokenizer.decode([t]) for t in enc_input[:50]]  # First 50
        tgt_tokens = [self.tokenizer.decode([t]) for t in dec_input[:30]]  # First 30
        
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # 1. Cross-attention heatmap
        print("\n1. Cross-Attention Heatmap (Layer 0)...")
        self.attention_extractor.visualize_attention(
            analysis_results['attention'],
            src_tokens, tgt_tokens,
            layer_idx=0, head_idx=0, attn_type='cross',
            save_path=self.output_dir / 'cross_attention_layer0.png'
        )
        
        # 2. All attention heads
        print("\n2. All Attention Heads (Layer 0)...")
        self.attention_extractor.visualize_all_heads(
            analysis_results['attention'],
            src_tokens, tgt_tokens,
            layer_idx=0, attn_type='cross',
            save_path=self.output_dir / 'all_heads_layer0.png'
        )
        
        # 3. Gradient flow
        if 'gradient_health' in analysis_results:
            print("\n3. Gradient Flow...")
            self.gradient_analyzer.visualize_gradient_flow(
                save_path=self.output_dir / 'gradient_flow.png'
            )
        
        # 4. Token prediction confidence
        print("\n4. Token Prediction Confidence...")
        self.token_explainer.visualize_token_probabilities(
            analysis_results['token_explanations'],
            save_path=self.output_dir / 'token_confidence.png'
        )
        
        # 5. Loss curve (if available)
        if len(self.loss_tracker.losses) > 0:
            print("\n5. Loss Curve...")
            self.loss_tracker.visualize_loss_curve(
                save_path=self.output_dir / 'loss_curve.png'
            )
        
        print(f"\n✅ All visualizations saved to {self.output_dir}/")
    
    def generate_report(self, batch=None, analysis_results=None):
        """
        Generate a comprehensive debugging report.
        """
        if batch is not None and analysis_results is None:
            analysis_results = self.analyze_batch(batch, verbose=False)
        
        print("\n" + "=" * 70)
        print("                    TRANSFORMER DEBUGGING REPORT")
        print("=" * 70)
        
        # 1. Loss Summary
        print("\n📉 LOSS SUMMARY")
        print("-" * 40)
        self.loss_tracker.print_summary()
        
        # 2. Gradient Health
        if self.gradient_analyzer.gradient_stats:
            print("\n📊 GRADIENT ANALYSIS")
            print("-" * 40)
            self.gradient_analyzer.print_gradient_report()
        
        # 3. Attention Summary
        if analysis_results and 'attention_summary' in analysis_results:
            print("\n👁️ ATTENTION PATTERNS")
            print("-" * 40)
            for key, stats in analysis_results['attention_summary'].items():
                print(f"  {key}:")
                print(f"    Shape: {stats['shape']}")
                print(f"    Entropy: {stats['entropy']:.4f}")
        
        # 4. Token Predictions
        if analysis_results and 'token_explanations' in analysis_results:
            print("\n🔤 TOKEN PREDICTIONS (first 10)")
            print("-" * 40)
            self.token_explainer.print_explanation(
                analysis_results['token_explanations'][:10]
            )
        
        print("\n" + "=" * 70)
        print("                         END OF REPORT")
        print("=" * 70)
    
    def cleanup(self):
        """Clean up hooks and resources."""
        self.attention_extractor.cleanup()


# Convenience function for quick debugging
def quick_debug(model, tokenizer, batch, device='cuda'):
    """
    Quick debugging function for a single batch.
    
    Usage:
        from debugger.dashboard import quick_debug
        quick_debug(model, tokenizer, batch)
    """
    debugger = TransformerDebugger(model, tokenizer, device)
    results = debugger.analyze_batch(batch)
    debugger.generate_report(analysis_results=results)
    debugger.cleanup()
    return results


if __name__ == '__main__':
    print("Transformer Debugger Dashboard")
    print("=" * 50)
    print("\nUsage:")
    print("  from debugger.dashboard import TransformerDebugger")
    print("  debugger = TransformerDebugger(model, tokenizer)")
    print("  debugger.analyze_batch(batch)")
    print("  debugger.visualize_all(batch)")
    print("  debugger.generate_report()")
