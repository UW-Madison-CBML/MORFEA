"""
Loss Functions — Ver12
======================
Standard implementations:

    - MS-SSIM              : pytorch_msssim (Wang et al. 2003)
    - L1/MSE               : torch.nn.functional
    - Multi-scale temporal : within-sequence, sampled without replacement

Changes from Ver11:
    - Pair sampling: with replacement -> without replacement
      Guarantees n_pairs distinct (Δp, Δn) combinations per step.
      All 16 possible pairs = {1,2,3,4} x {8,12,16,24}

Combined loss:
    L_total = L_recon + temporal_weight * L_temporal

Multi-scale temporal loss:
    anchor   = z_t
    positive = z_{t+Δp}    Δp in {1, 2, 3, 4}
    negative = z_{t+Δn}    Δn in {8, 12, 16, 24}

    Loss: F.softplus(sim_neg - sim_pos).mean()
        - soft margin: always has gradient
        - same-embryo only: no embryo identity learning
        - random offsets: prevents memorising fixed rules
        - no-duplicate sampling: maximises diversity per step
"""

import random
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

# All possible (Δp, Δn) combinations
_DELTA_P   = [1, 2, 3, 4]
_DELTA_N   = [8, 12, 16, 24]
_ALL_PAIRS = [(dp, dn) for dp in _DELTA_P for dn in _DELTA_N]  # 16 pairs


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    x_rec: torch.Tensor,
    x_true: torch.Tensor,
    pixel_weight: float = 0.5,
    ms_ssim_weight: float = 0.5,
    loss_type: str = "l1",
) -> tuple[torch.Tensor, dict]:
    """
    Combined pixel + MS-SSIM reconstruction loss.
    Args:
        x_rec, x_true : (B, T, C, H, W)
        pixel_weight   : weight for L1 or MSE
        ms_ssim_weight : weight for MS-SSIM
        loss_type      : "l1" | "mse"
    """
    B, T, C, H, W = x_rec.shape
    r = x_rec.reshape(B * T, C, H, W)
    t = x_true.reshape(B * T, C, H, W)

    pixel = F.l1_loss(r, t) if loss_type == "l1" else F.mse_loss(r, t)

    smaller_side = min(H, W)
    if smaller_side < 161:
        scale = 161.0 / float(smaller_side)
        new_h = int(round(H * scale))
        new_w = int(round(W * scale))
        r_ms = F.interpolate(r, size=(new_h, new_w), mode="bilinear",
                             align_corners=False)
        t_ms = F.interpolate(t, size=(new_h, new_w), mode="bilinear",
                             align_corners=False)
    else:
        r_ms, t_ms = r, t

    ms      = ms_ssim(r_ms, t_ms, data_range=1.0, size_average=True)
    ms_loss = 1.0 - ms
    loss    = pixel_weight * pixel + ms_ssim_weight * ms_loss

    return loss, {
        f"{loss_type}": pixel.item(),
        "ms_ssim": ms.item(),
        "rec_loss": loss.item(),
    }


# ---------------------------------------------------------------------------
# Multi-scale temporal loss
# ---------------------------------------------------------------------------

def multiscale_temporal_loss(
    z_seq: torch.Tensor,
    n_pairs: int = 4,
) -> tuple[torch.Tensor, dict]:
    """
    Multi-scale within-sequence temporal loss.

    Samples n_pairs distinct (Δp, Δn) pairs without replacement from
    all 16 combinations in {1,2,3,4} x {8,12,16,24}.

    For each pair:
        anchor   = z_t
        positive = z_{t+Δp}   (near future, same embryo)
        negative = z_{t+Δn}   (far future,  same embryo)

        loss = F.softplus(sim_neg - sim_pos).mean()

    Pairs where T - max(Δp, Δn) - 1 <= 0 are skipped.

    Args:
        z_seq   : (B, T, D)
        n_pairs : number of distinct pairs to sample (max 16)
    """
    B, T, D = z_seq.shape

    z = F.normalize(z_seq, dim=-1)  # (B, T, D)

    # Sample without replacement — guarantees distinct pairs
    sampled = random.sample(_ALL_PAIRS, min(n_pairs, len(_ALL_PAIRS)))

    total_loss  = 0.0
    valid_pairs = 0
    first_sims  = None

    for dp, dn in sampled:
        n_steps = T - max(dp, dn) - 1
        if n_steps <= 0:
            continue

        anchor   = z[:, :n_steps, :]
        positive = z[:, dp:dp + n_steps, :]
        negative = z[:, dn:dn + n_steps, :]

        sim_pos = (anchor * positive).sum(-1)  # (B, n_steps)
        sim_neg = (anchor * negative).sum(-1)

        total_loss += F.softplus(sim_neg - sim_pos).mean()
        valid_pairs += 1

        if first_sims is None:
            first_sims = (sim_pos.detach(), sim_neg.detach())

    if valid_pairs == 0:
        return torch.tensor(0.0, device=z_seq.device), {
            "temporal_loss": 0.0,
            "temporal_pos_sim": 0.0,
            "temporal_neg_sim": 0.0,
        }

    loss = total_loss / valid_pairs

    with torch.no_grad():
        pos_sim = first_sims[0].mean().item() if first_sims else 0.0
        neg_sim = first_sims[1].mean().item() if first_sims else 0.0

    return loss, {
        "temporal_loss": loss.item(),
        "temporal_pos_sim": pos_sim,
        "temporal_neg_sim": neg_sim,
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict,
    x_true: torch.Tensor,
    pixel_weight: float    = 0.5,
    ms_ssim_weight: float  = 0.5,
    temporal_weight: float = 0.1,
    temporal_pairs: int    = 4,
    loss_type: str         = "l1",
) -> tuple[torch.Tensor, dict]:
    """
    L_total = L_recon + temporal_weight * L_temporal

    Args:
        outputs         : model.forward() dict
        x_true          : (B, T, C, H, W)
        pixel_weight    : L1/MSE weight
        ms_ssim_weight  : MS-SSIM weight
        temporal_weight : multi-scale temporal loss weight (0 to ablate)
        temporal_pairs  : number of distinct (Δp, Δn) pairs per step
        loss_type       : "l1" | "mse"
    """
    rec, rec_m = reconstruction_loss(
        outputs["reconstruction"], x_true,
        pixel_weight=pixel_weight,
        ms_ssim_weight=ms_ssim_weight,
        loss_type=loss_type,
    )

    z = outputs["z_seq"]
    can_temporal = (
        temporal_weight > 0
        and z.shape[1] > max(_DELTA_N) + 1
    )

    if can_temporal:
        tmp, tmp_m = multiscale_temporal_loss(z, n_pairs=temporal_pairs)
    else:
        tmp   = torch.tensor(0.0, device=x_true.device)
        tmp_m = {
            "temporal_loss": 0.0,
            "temporal_pos_sim": 0.0,
            "temporal_neg_sim": 0.0,
        }

    total = rec + temporal_weight * tmp

    return total, {
        "total_loss": total.item(),
        **rec_m,
        **tmp_m,
    }
