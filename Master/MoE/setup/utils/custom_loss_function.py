# =============================================================================
# Time-MoE aligned loss functions for Mixture-of-Experts routing
#
# Implements ONLY the auxiliary load-balancing loss described in Time-MoE:
#
#   L_aux = E * sum_i (f_i * r_i)
#
# Where:
#   - r_i: average router probability for expert i (from CLEAN logits)
#   - f_i: fraction of samples routed to expert i (top-k, possibly noisy)
#
# IMPORTANT:
#   This auxiliary loss must be used ONLY when noisy routing is enabled
#   (use_noise=True). For deterministic routing, use only prediction loss.
# =============================================================================

import torch


# -----------------------------------------------------------------------------
# Auxiliary Load Balance Loss 
# -----------------------------------------------------------------------------
def aux_loss(
    probs_clean: torch.Tensor,
    topk_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Time-MoE auxiliary routing loss.

    Args:
        probs_clean: (B, E) softmax over CLEAN router logits
        topk_idx: (B, K) indices of selected experts

    Returns:
        Scalar tensor (auxiliary loss)
    """

    B, E = probs_clean.shape

    # r_i: router probability mass per expert
    r = probs_clean.sum(dim=0)
    r = r / (r.sum() + 1e-8)

    # f_i: fraction of samples routed to each expert
    one_hot = torch.zeros_like(probs_clean)
    one_hot.scatter_(1, topk_idx, 1.0)
    f = one_hot.sum(dim=0)
    f = f / (f.sum() + 1e-8)

    # Time-MoE auxiliary loss
    loss_aux = E * torch.sum(f * r)
    return loss_aux


# -----------------------------------------------------------------------------
# Final Loss: Prediction Loss + Auxiliary Loss
# -----------------------------------------------------------------------------
def moe_custom_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    probs_clean: torch.Tensor,
    topk_idx: torch.Tensor,
    pred_loss_fn,
    alpha: float = 0.02,
) -> torch.Tensor:
    """
    Final loss used for Time-MoE-style MoE routing.

    Args:
        preds: model predictions (B, H)
        targets: ground truth (B, H)
        probs_clean: (B, E) clean router probabilities
        topk_idx: (B, K) selected experts
        pred_loss_fn: prediction loss (e.g., torch.nn.HuberLoss)
        alpha: auxiliary loss weight (Time-MoE uses ~0.02)

    Returns:
        Scalar tensor (total loss)
    """

    pred_loss = pred_loss_fn(preds, targets)

    return_aux_loss = aux_loss(
        probs_clean=probs_clean,
        topk_idx=topk_idx,
    )

    return pred_loss + alpha * return_aux_loss