# =============================================================================
# PT: Funções de loss alinhadas com Time-MoE para roteamento Mixture-of-Experts
#
#     Implementa APENAS a auxiliary loss de balanceamento de carga descrita no
#     Time-MoE / Switch Transformer:
#
#         L_aux = E * sum_i (f_i * r_i)
#
#     Onde:
#     - r_i: massa média de probabilidade do roteador para o expert i
#            (calculada a partir dos logits LIMPOS, sem ruído)
#     - f_i: fração de decisões de roteamento atribuídas ao expert i
#            (top-k, possivelmente com ruído)
#
# IMPORTANTE:
#     Esta auxiliary loss deve ser usada APENAS quando o roteamento com ruído
#     estiver habilitado (use_noise=True). Para roteamento determinístico,
#     use apenas a prediction loss.
#
# -----------------------------------------------------------------------------
# EN: Time-MoE aligned loss functions for Mixture-of-Experts routing
#
#     Implements ONLY the auxiliary load-balancing loss described in
#     Time-MoE / Switch Transformer:
#
#         L_aux = E * sum_i (f_i * r_i)
#
#     Where:
#     - r_i: average router probability mass for expert i
#            (from CLEAN router logits, without noise)
#     - f_i: fraction of routing decisions assigned to expert i
#            (top-k, possibly noisy)
#
# IMPORTANT:
#     This auxiliary loss must be used ONLY when noisy routing is enabled
#     (use_noise=True). For deterministic routing, use only prediction loss.
# =============================================================================

import torch


# -----------------------------------------------------------------------------
# PT: Auxiliary Loss de Balanceamento de Carga
# EN: Auxiliary Load Balance Loss
# -----------------------------------------------------------------------------
def aux_loss(
    probs_clean: torch.Tensor,
    topk_idx: torch.Tensor,
) -> torch.Tensor:
    """
    PT: Auxiliary loss do Time-MoE para balanceamento de roteamento.

    Args:
        probs_clean: (B, E) softmax sobre os logits LIMPOS do roteador (sem ruído)
        topk_idx: (B, K) índices dos experts selecionados (top-k)

    Returns:
        Tensor escalar (auxiliary loss)

    EN: Time-MoE auxiliary routing loss.

    Args:
        probs_clean: (B, E) softmax over CLEAN router logits (without noise)
        topk_idx: (B, K) indices of selected experts (top-k)

    Returns:
        Scalar tensor (auxiliary loss)
    """

    B, E = probs_clean.shape
    K = topk_idx.size(1)

    # -------------------------------------------------------------------------
    # PT: r_i — massa de probabilidade do roteador por expert
    #
    #     Soma as probabilidades do roteador para cada expert ao longo do batch
    #     e normaliza para obter a distribuição média de probabilidade.
    #
    #     Esta forma é matematicamente equivalente a probs_clean.mean(dim=0),
    #     já que cada linha de probs_clean soma 1 (softmax).
    #
    # EN: r_i — router probability mass per expert
    #
    #     Sums router probabilities for each expert across the batch and
    #     normalizes to obtain the average probability distribution.
    # -------------------------------------------------------------------------
    r = probs_clean.sum(dim=0)          # (E,)
    r = r / (r.sum() + 1e-8)            # normalização explícita


    # -------------------------------------------------------------------------
    # PT: f_i — fração de decisões de roteamento por expert
    #
    #     Conta quantas vezes cada expert foi selecionado no top-k,
    #     considerando que cada amostra contribui com K decisões.
    #
    #     A normalização por (B * K) garante que f represente a fração
    #     total de "slots de roteamento".
    #
    # EN: f_i — fraction of routing decisions per expert
    #
    #     Counts how many times each expert was selected in top-k,
    #     considering that each sample contributes K routing decisions.
    #
    #     Normalization by (B * K) yields a proper fraction.
    # -------------------------------------------------------------------------
    topk_flat = topk_idx.reshape(-1)    # (B * K,)
    f = torch.bincount(topk_flat, minlength=E).float()
    f = f / (B * K)


    # -------------------------------------------------------------------------
    # PT: Auxiliary loss do Time-MoE
    #     Incentiva alinhamento entre a probabilidade esperada do roteador (r)
    #     e o uso efetivo dos experts (f).
    #
    # EN: Time-MoE auxiliary loss
    #     Encourages alignment between expected router probabilities (r)
    #     and actual expert usage (f).
    # -------------------------------------------------------------------------
    loss_aux = E * torch.sum(f * r)

    return loss_aux


# -----------------------------------------------------------------------------
# PT: Loss Final — Prediction Loss + Auxiliary Loss
# EN: Final Loss — Prediction Loss + Auxiliary Loss
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
    PT: Loss final usada para treinamento com roteamento estilo Time-MoE.

    Args:
        preds: predições do modelo (B, H)
        targets: ground truth (B, H)
        probs_clean: (B, E) probabilidades limpas do roteador (sem ruído)
        topk_idx: (B, K) experts selecionados
        pred_loss_fn: função de prediction loss (ex: HuberLoss)
        alpha: peso da auxiliary loss (Time-MoE usa ~0.01–0.02)

    Returns:
        Tensor escalar (loss total)

    EN: Final loss used for Time-MoE-style MoE routing.

    Args:
        preds: model predictions (B, H)
        targets: ground truth (B, H)
        probs_clean: (B, E) clean router probabilities (without noise)
        topk_idx: (B, K) selected experts
        pred_loss_fn: prediction loss (e.g., HuberLoss)
        alpha: auxiliary loss weight (Time-MoE uses ~0.01–0.02)

    Returns:
        Scalar tensor (total loss)
    """

    # PT: Loss de predição (Huber, MSE, MAE, etc.)
    # EN: Prediction loss (Huber, MSE, MAE, etc.)
    pred_loss = pred_loss_fn(preds, targets)

    # PT: Auxiliary loss de balanceamento de carga
    # EN: Auxiliary load balancing loss
    balance_loss = aux_loss(
        probs_clean=probs_clean,
        topk_idx=topk_idx,
    )

    # PT: Loss total
    # EN: Total loss
    total_loss = pred_loss + alpha * balance_loss

    return total_loss
