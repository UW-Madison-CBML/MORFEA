"""
Loss Functions — Ver10
======================
Standard implementations:

    - MS-SSIM        : pytorch_msssim (Wang et al. 2003)
    - L1/MSE         : torch.nn.functional
    - Rolled InfoNCE : within-batch contrastive with shifted positive

Combined loss:
    L_total = L_recon + infonce_weight * L_infonce

Rolled InfoNCE design (credit: collaborator suggestion):
    anchor   = z_seq[:, :T-1]              first T-1 frames
    positive = z_seq[:, 1:]                next frame (enforces smoothness)
    negative = torch.roll(z_seq, 1, dim=0) same timesteps from another embryo

    Why this works:
    - positive is z_{t+1}: enforces temporal smoothness within trajectory
    - negative is rolled batch: uses OTHER embryos as negatives
    - prevents all embryos collapsing to same latent region
    - does NOT suffer from LSTM trivial solution (near frames always closer
      than rolled negatives from different embryos)
    - batch_size >= 2 required (roll needs at least one other embryo)
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


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
    # pytorch-msssim requires min side > 160; upsample small images
    smaller_side = min(H, W)
    target_min = 161.0
    if smaller_side < target_min:
        scale = target_min / float(smaller_side)
        new_h, new_w = int(round(H * scale)), int(round(W * scale))
        r_ms = F.interpolate(r, size=(new_h, new_w), mode="bilinear", align_corners=False)
        t_ms = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
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
# Rolled InfoNCE
# ---------------------------------------------------------------------------

def rolled_infonce(
    z_seq: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    """
    Rolled InfoNCE contrastive loss.

    For each timestep t in [0, T-2]:
        anchor   = z_t               (current frame)
        positive = z_{t+1}           (next frame, same embryo — smoothness)
        negative = z_t from rolled   (same timestep, different embryo)

    The rolled negative uses torch.roll(z_seq, 1, dim=0):
        embryo 0's negative = embryo B-1's latent
        embryo 1's negative = embryo 0's latent
        etc.

    This enforces two things simultaneously:
        1. Temporal smoothness: z_t close to z_{t+1}
        2. Embryo separation: z_t far from other embryos' latents

    Unlike CPC, this does NOT suffer from the LSTM trivial solution.
    The LSTM makes z_t close to z_{t+1} (positive), but does NOT make
    z_t close to a different embryo's z_t (negative).

    Args:
        z_seq       : (B, T, D)
        temperature : InfoNCE temperature

    Requirements:
        B >= 2 (need at least one other embryo as negative)
        T >= 2
    """
    B, T, D = z_seq.shape
    assert B >= 2, "Rolled InfoNCE requires batch_size >= 2"
    assert T >= 2, "Rolled InfoNCE requires T >= 2"

    z = F.normalize(z_seq, dim=-1)  # (B, T, D)

    anchor   = z[:, :T - 1, :]                          # (B, T-1, D)
    positive = z[:, 1:, :]                               # (B, T-1, D)
    negative = torch.roll(z, shifts=1, dims=0)[:, :T - 1, :]  # (B, T-1, D)

    # Flatten time dimension: each (embryo, timestep) is one sample
    a = anchor.reshape(-1, D)    # (B*(T-1), D)
    p = positive.reshape(-1, D)
    n = negative.reshape(-1, D)

    # InfoNCE with 1 positive and 1 negative
    # logits[:, 0] = sim(anchor, positive)
    # logits[:, 1] = sim(anchor, negative)
    logits = torch.stack([
        (a * p).sum(-1) / temperature,
        (a * n).sum(-1) / temperature,
    ], dim=-1)  # (B*(T-1), 2)

    # label=0: positive should rank first
    labels = torch.zeros(B * (T - 1), dtype=torch.long, device=z_seq.device)
    loss   = F.cross_entropy(logits, labels)

    with torch.no_grad():
        pos_sim = (a * p).sum(-1).mean().item()
        neg_sim = (a * n).sum(-1).mean().item()

    return loss, {
        "infonce_loss": loss.item(),
        "infonce_pos_sim": pos_sim,
        "infonce_neg_sim": neg_sim,
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict,
    x_true: torch.Tensor,
    pixel_weight: float    = 0.5,
    ms_ssim_weight: float  = 0.5,
    infonce_weight: float  = 0.1,
    infonce_temp: float    = 0.07,
    loss_type: str         = "l1",
) -> tuple[torch.Tensor, dict]:
    """
    L_total = L_recon + infonce_weight * L_infonce

    Args:
        outputs        : model.forward() dict
        x_true         : (B, T, C, H, W)
        pixel_weight   : L1/MSE weight
        ms_ssim_weight : MS-SSIM weight
        infonce_weight : rolled InfoNCE weight (0 to ablate)
        infonce_temp   : InfoNCE temperature
        loss_type      : "l1" | "mse"
    """
    rec, rec_m = reconstruction_loss(
        outputs["reconstruction"], x_true,
        pixel_weight=pixel_weight,
        ms_ssim_weight=ms_ssim_weight,
        loss_type=loss_type,
    )

    z = outputs["z_seq"]
    can_infonce = (
        infonce_weight > 0
        and z.shape[0] >= 2
        and z.shape[1] >= 2
    )
    if can_infonce:
        nce, nce_m = rolled_infonce(z, temperature=infonce_temp)
    else:
        nce   = torch.tensor(0.0, device=x_true.device)
        nce_m = {
            "infonce_loss": 0.0,
            "infonce_pos_sim": 0.0,
            "infonce_neg_sim": 0.0,
        }

    total = rec + infonce_weight * nce

    return total, {
        "total_loss": total.item(),
        **rec_m,
        **nce_m,
    }
