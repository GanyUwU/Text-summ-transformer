import torch
import math
from typing import List

def get_layerwise_param_groups(model, base_lr: float, decoder_lr_scale: float = 1.0):
    encoder_params = []
    decoder_params = []
    other = []
    for name, p in model.named_parameters():
        if 'decoder' in name:
            decoder_params.append(p)
        elif 'encoder' in name:
            encoder_params.append(p)
        else:
            other.append(p)

    groups = []
    # Annotate groups with a `group_type` key so callers can identify encoder/decoder groups
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': base_lr, 'group_type': 'encoder'})
    if decoder_params:
        groups.append({'params': decoder_params, 'lr': base_lr * decoder_lr_scale, 'group_type': 'decoder'})
    if other:
        groups.append({'params': other, 'lr': base_lr, 'group_type': 'other'})
    return groups


def reinit_collapsed_heads(model, only_decoder: bool = True):
    # Reinitialize linear projections in attention blocks to break symmetry
    for name, m in model.named_modules():
        if only_decoder and 'decoder' not in name:
            continue
        if isinstance(m, torch.nn.Linear):
            try:
                m.reset_parameters()
            except Exception:
                pass


def compute_coverage_loss(attn_mean: torch.Tensor):
    # attn_mean expected [B, T, S] or [B, H, T, S]
    if attn_mean.dim() == 4:
        attn_mean = attn_mean.mean(dim=1)
    cov = torch.clamp(attn_mean.cumsum(dim=1) - 1.0, min=0.0)
    return cov.mean()


def entropy_regularization(attn: torch.Tensor, target_entropy: float = 1.6):
    # attn shape: [B, H, T, S]
    eps = 1e-12
    ent = -(attn * (attn + eps).log()).sum(-1)  # [B, H, T]
    mean_ent = ent.mean()
    # Penalty when entropy is larger than target (we want lower entropy)
    penalty = torch.relu(mean_ent - target_entropy)
    return penalty


def log_attention_entropy(writer, attn, step):
    try:
        eps = 1e-12
        ent = -(attn * (attn + eps).log()).sum(-1).mean().item()
        writer.add_scalar('diagnostics/attention_entropy', ent, step)
    except Exception:
        pass


def log_pgen(writer, p_gen, step):
    try:
        writer.add_scalar('diagnostics/pgen_mean', float(p_gen.mean().item()), step)
    except Exception:
        pass


def check_gradient_health(model, writer=None, step=None):
    total_norm = 0.0
    count = 0
    warnings = []
    for p in model.parameters():
        if p.grad is None:
            continue
        norm = p.grad.data.norm(2).item()
        total_norm += norm
        count += 1
        if math.isnan(norm) or math.isinf(norm):
            warnings.append('NaN/Inf gradient detected')
    avg = total_norm / max(1, count)
    if writer is not None and step is not None:
        writer.add_scalar('diagnostics/grad_norm', avg, step)
    return {'avg_grad_norm': avg, 'warnings': warnings}


def print_diagnostic_summary(model):
    total = sum(p.numel() for p in model.parameters())
    print(f'Diagnostics: model params={total:,}')


def smooth_nll_loss(log_probs, targets, ignore_index=0, label_smoothing=0.0, smoothing=None):
    """
    NLL loss with optional label smoothing for LOG-PROBABILITY inputs.
    
    IMPORTANT: This function expects log-probabilities (output of torch.log(softmax(x))),
    NOT raw logits. Using CrossEntropyLoss here would apply softmax a second time,
    creating a double-softmax that crushes gradients.
    """
    if smoothing is not None:
        label_smoothing = smoothing

    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    targets_flat = targets.view(-1)

    if label_smoothing > 0:
        n_classes = log_probs_flat.size(-1)
        # One-hot style: (1-eps)*nll + eps*uniform_smoothing
        nll = -log_probs_flat.gather(dim=-1, index=targets_flat.clamp(min=0).unsqueeze(1)).squeeze(1)
        smooth = -log_probs_flat.mean(dim=-1)
        mask = (targets_flat != ignore_index).float()
        loss = ((1.0 - label_smoothing) * nll + label_smoothing * smooth) * mask
        return loss.sum() / mask.sum().clamp(min=1.0)
    else:
        loss_fn = torch.nn.NLLLoss(ignore_index=ignore_index)
        return loss_fn(log_probs_flat, targets_flat)


def pgen_balance_loss(p_gen: torch.Tensor, target: float = 0.45):
    """Legacy soft MSE-based p_gen penalty (used when lambda_p is not configured)."""
    return (p_gen.mean() - target) ** 2


def aggressive_pgen_penalty(p_gen: torch.Tensor):
    """AGGRESSIVE LINEAR PENALTY for copying (Boutkan et al.).
    
    Penalizes (1 - p_gen) directly: high penalty when model copies (p_gen → 0).
    This is much stricter than MSE and forces the model to abandon the copying plateau.
    
    Loss = mean(1 - p_gen) → encourages p_gen toward 1 (generation).
    """
    return torch.mean(1.0 - p_gen)
