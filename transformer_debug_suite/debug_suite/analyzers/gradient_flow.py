import torch
import torch.nn as nn
import numpy as np

class GradientFlowAnalyzer:
    """
    Simulates a backward pass using a loaded PyTorch model and synthetic data
    to analyze gradient flow (vanishing / exploding gradients).
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def analyze(self):
        self.model.train()
        
        # Determine shapes based on typical config
        batch_size = 2
        src_seq_len = 128
        tgt_seq_len = 50
        
        try:
            # We assume model is the standard Transformer from model.py
            # Need to get vocab_size. We can infer it from the projection layer.
            vocab_size = -1
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and 'proj' in name and getattr(module, 'out_features', 0) > 1000:
                    vocab_size = module.out_features
                    break
            
            if vocab_size == -1:
                vocab_size = 32000 # default fallback
                
            enc_input = torch.randint(0, vocab_size, (batch_size, src_seq_len), device=self.device)
            dec_input = torch.randint(0, vocab_size, (batch_size, tgt_seq_len), device=self.device)
            
            # Dummy masks
            enc_mask = torch.ones((batch_size, 1, 1, src_seq_len), device=self.device, dtype=torch.bool)
            # causal mask
            causal = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=self.device, dtype=torch.bool)).view(1, 1, tgt_seq_len, tgt_seq_len)
            dec_mask = causal
            
            # Forward pass
            # We use forward_with_copy if it exists, otherwise standard encode/decode
            if hasattr(self.model, 'forward_with_copy'):
                final_dist, p_gen, live_attn = self.model.forward_with_copy(enc_input, enc_mask, dec_input, dec_mask)
                log_probs = torch.log(final_dist + 1e-12)
                logits = log_probs
            else:
                enc_out = self.model.encode(enc_input, enc_mask)
                dec_out = self.model.decode(enc_out, enc_mask, dec_input, dec_mask)
                logits = self.model.project(dec_out)
                
            # Dummy labels
            labels = torch.randint(0, vocab_size, (batch_size, tgt_seq_len), device=self.device)
            
            # Loss and Backward
            # final_dist is already a probability distribution.
            # We use NLLLoss on log_probs to get correct gradients for probability-blended outputs.
            loss_fn = nn.NLLLoss()
            loss = loss_fn(log_probs.view(-1, vocab_size), labels.view(-1))
            
            self.model.zero_grad()
            loss.backward()
            
            # Extract gradients
            grad_flow = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_np = param.grad.cpu().numpy()
                    grad_flow[name] = {
                        "mean": float(np.mean(grad_np)),
                        "std": float(np.std(grad_np)),
                        "max": float(np.max(np.abs(grad_np))),
                        "l2_norm": float(np.linalg.norm(grad_np))
                    }
                    
            return self._format_results(grad_flow)
            
        except Exception as e:
            return {"error": f"Gradient simulation failed: {e}"}

    def _format_results(self, grad_flow):
        # We want to show vanishing / exploding
        layers = sorted([n for n in grad_flow.keys() if 'weight' in n])
        stds = [grad_flow[l]["std"] for l in layers]
        
        issues = []
        if not stds:
            return {"error": "No gradients found."}
            
        max_std = max(stds)
        min_std = min(stds)
        
        # Vanishing: If we have layers with very small stds while others are normal
        if min_std < 1e-6 and max_std > 1e-3:
            issues.append(f"Vanishing gradients detected in some layers (min std: {min_std:.2e})")
            
        # Exploding
        if max_std > 10.0:
            issues.append(f"Exploding gradients detected (max std: {max_std:.2e})")
            
        return {
            "layer_stats": grad_flow,
            "issues": issues,
            "summary": {
                "max_std": max_std,
                "min_std": min_std,
                "num_layers": len(layers)
            }
        }
