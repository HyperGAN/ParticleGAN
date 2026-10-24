"""Wavefront Laws: Sobolev H^s gradient filtering, manifold priors, and ghost-free splat carrier.

This module implements:
- Law 1:
  - sobolev_filter_2d: Real 2D FFT Bessel-potential smoothing operator.
  - SobolevGradFilter: Straight-through autograd filter applying Sobolev smoothing to backward gradients.
  - WavefrontManifoldPrior: Patch-seam ratio, biharmonic curvature, and radial PSD excess regularizers.
  - SobolevWavefrontManipulator: Composable PriorManipulator integrating Law 1 into GANTrainer.
- Law 2:
  - GhostFreeSimplexMixer: Base+detail de-mixing operator with exact zero negative ghost leakage.
  - GhostFreeDeMixingManipulator: PriorManipulator applying GhostFreeSimplexMixer to mode representations.
  - DifferentiableSplatCarrierBasis: Domain-agnostic 2D Gaussian splat spatial carrier basis (2D parameters).
  - SplatCarrierManipulator: Composable PriorManipulator mapping latent codes to rendered splat carrier.
  - SimplexWavefrontGenerator: Multi-scale conv residual pyramid generator built on spatial carriers.
"""
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from .priors import PriorManipulator


__all__ = [
    "sobolev_filter_2d",
    "SobolevGradFilter",
    "WavefrontManifoldPrior",
    "SobolevWavefrontManipulator",
    "GhostFreeSimplexMixer",
    "GhostFreeDeMixingManipulator",
    "DifferentiableSplatCarrierBasis",
    "SplatCarrierManipulator",
    "ResidualConvBlock",
    "SimplexWavefrontGenerator",
]


def sobolev_filter_2d(
    grad: torch.Tensor,
    *,
    lam: float = 1.0,
    order: float = 1.0,
    nyquist: float = math.pi,
) -> torch.Tensor:
    """Preconditions 2D spatial gradients via Riemannian Sobolev H^s Bessel potential.

    Applies the frequency-domain operator:
        \\widetilde{g} = \\mathcal{F}^{-1}\\left\\{ (1 + \\lambda \\|\\mathbf{\\omega}\\|^2)^{-s} \\mathcal{F}\\{g\\} \\right\\}

    Args:
        grad: Float tensor of shape (..., H, W) with ndim >= 2.
        lam: Non-negative filtering scale parameter \\lambda.
        order: Non-negative Sobolev order s (e.g. 1.0 for H^1).
        nyquist: Angular frequency coordinate corresponding to Nyquist frequency
                 (default math.pi for standard angular frequency; 1.0 for unit-normalized).

    Returns:
        Tensor of identical shape, device, and dtype as grad.
    """
    if not isinstance(grad, torch.Tensor):
        raise TypeError(f"sobolev_filter_2d expects a torch.Tensor, got {type(grad)}")
    if not grad.is_floating_point():
        raise TypeError(f"sobolev_filter_2d requires a floating-point tensor, got {grad.dtype}")
    if grad.ndim < 2:
        raise ValueError(f"sobolev_filter_2d requires tensor with ndim >= 2, got {grad.ndim}")
    if lam <= 0.0 or order <= 0.0:
        return grad

    H, W = grad.shape[-2:]
    orig_dtype = grad.dtype
    # torch.fft.rfft2 on CPU does not support float16 / bfloat16; upcast to float32
    compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    x = grad.to(compute_dtype)

    scale_y = 2.0 * float(nyquist)
    scale_x = 2.0 * float(nyquist)
    fy = fft.fftfreq(H, device=grad.device, dtype=compute_dtype).view(H, 1) * scale_y
    fx = fft.rfftfreq(W, device=grad.device, dtype=compute_dtype).view(1, W // 2 + 1) * scale_x
    omega_sq = fy ** 2 + fx ** 2
    bessel_filter = (1.0 + float(lam) * omega_sq).pow(-float(order))

    F_grad = fft.rfft2(x, dim=(-2, -1))
    filtered_F = F_grad * bessel_filter
    out = fft.irfft2(filtered_F, s=(H, W), dim=(-2, -1))
    return out.to(orig_dtype)


class _SobolevGradFilterFunction(torch.autograd.Function):
    """Autograd function passing forward bit-identically while filtering backward gradients."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lam: float, order: float, nyquist: float) -> torch.Tensor:
        ctx.lam = lam
        ctx.order = order
        ctx.nyquist = nyquist
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], None, None, None]:
        if grad_output is None or ctx.lam <= 0.0 or ctx.order <= 0.0:
            return grad_output, None, None, None
        filtered_grad = sobolev_filter_2d(
            grad_output,
            lam=ctx.lam,
            order=ctx.order,
            nyquist=ctx.nyquist,
        )
        return filtered_grad, None, None, None


class SobolevGradFilter(nn.Module):
    """Sobolev H^s gradient preconditioning filter.

    Leaves forward activations strictly bit-identical (x -> x) while preconditioning
    backward gradients nabla_x L via Bessel-potential smoothing in the frequency domain.
    """

    def __init__(
        self,
        *,
        lam: float = 1.0,
        order: float = 1.0,
        nyquist: float = math.pi,
    ) -> None:
        super().__init__()
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError(f"lam must be finite and nonnegative, got {lam}")
        if not math.isfinite(order) or order < 0.0:
            raise ValueError(f"order must be finite and nonnegative, got {order}")
        if not math.isfinite(nyquist) or nyquist <= 0.0:
            raise ValueError(f"nyquist must be finite and positive, got {nyquist}")
        self.lam = float(lam)
        self.order = float(order)
        self.nyquist = float(nyquist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lam <= 0.0 or self.order <= 0.0 or not x.requires_grad:
            return x
        return _SobolevGradFilterFunction.apply(x, self.lam, self.order, self.nyquist)

    def extra_repr(self) -> str:
        return f"lam={self.lam}, order={self.order}, nyquist={self.nyquist}"


def _pad_linear_2d(img: torch.Tensor) -> torch.Tensor:
    """Pad a 4D tensor with linear extrapolation (zero second derivative at boundaries)."""
    top = 2.0 * img[:, :, 0:1, :] - img[:, :, 1:2, :]
    bot = 2.0 * img[:, :, -1:, :] - img[:, :, -2:-1, :]
    out = torch.cat([top, img, bot], dim=2)
    left = 2.0 * out[:, :, :, 0:1] - out[:, :, :, 1:2]
    right = 2.0 * out[:, :, :, -1:] - out[:, :, :, -2:-1]
    return torch.cat([left, out, right], dim=3)


class WavefrontManifoldPrior(nn.Module):
    """Optical Wavefront Manifold Prior regularizing spatial, seam, and spectral coherence.

    Penalizes:
    1. Patch-seam gradient ratio along ViT patch boundaries (primary and optional secondary patch size):
       R_seam = E_seam[g] / max(E_interior[g], eps)
       L_seam = relu(R_seam - seam_threshold)^2
    2. Biharmonic curvature penalty:
       L_biharmonic = E[sqrt(|Delta x|^2 + delta^2) - delta]
       via discrete 5-point or 9-point Laplacian convolution with reflect/linear/replicate padding.
    3. Radial 1/f^alpha Power Spectral Density (PSD) excess penalty:
       Evaluates 2D rFFT radial power falloff, penalizing excess log-power ratios above
       P_ref * (r/r0)^(-alpha) * 10^(margin_db / 10), gated by AC power.
    """

    def __init__(
        self,
        *,
        weight_seam: float = 1.0,
        weight_biharmonic: float = 0.1,
        weight_psd: float = 0.01,
        patch_size: int = 4,
        secondary_patch_size: Optional[int] = None,
        seam_threshold: float = 1.05,
        laplacian_kernel: str = "5point",
        alpha: float = 2.0,
        margin_db: float = 1.0,
        padding_mode: str = "reflect",
        delta: float = 0.01,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not math.isfinite(weight_seam) or weight_seam < 0.0:
            raise ValueError(f"weight_seam must be nonnegative, got {weight_seam}")
        if not math.isfinite(weight_biharmonic) or weight_biharmonic < 0.0:
            raise ValueError(f"weight_biharmonic must be nonnegative, got {weight_biharmonic}")
        if not math.isfinite(weight_psd) or weight_psd < 0.0:
            raise ValueError(f"weight_psd must be nonnegative, got {weight_psd}")
        if not isinstance(patch_size, int) or patch_size < 2:
            raise ValueError(f"patch_size must be an integer >= 2, got {patch_size}")
        if secondary_patch_size is not None and (not isinstance(secondary_patch_size, int) or secondary_patch_size < 2):
            raise ValueError(f"secondary_patch_size must be None or an integer >= 2, got {secondary_patch_size}")
        if not math.isfinite(seam_threshold) or seam_threshold <= 0.0:
            raise ValueError(f"seam_threshold must be positive, got {seam_threshold}")
        if laplacian_kernel not in ("5point", "9point"):
            raise ValueError(f"laplacian_kernel must be '5point' or '9point', got {laplacian_kernel}")
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not math.isfinite(margin_db) or margin_db < 0.0:
            raise ValueError(f"margin_db must be nonnegative, got {margin_db}")
        if padding_mode not in ("reflect", "replicate", "zeros", "linear"):
            raise ValueError(f"padding_mode must be 'reflect', 'replicate', 'zeros', or 'linear', got {padding_mode}")
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")

        self.weight_seam = float(weight_seam)
        self.weight_biharmonic = float(weight_biharmonic)
        self.weight_psd = float(weight_psd)
        self.patch_size = int(patch_size)
        self.secondary_patch_size = int(secondary_patch_size) if secondary_patch_size is not None else None
        self.seam_threshold = float(seam_threshold)
        self.laplacian_kernel = laplacian_kernel
        self.alpha = float(alpha)
        self.margin_db = float(margin_db)
        self.padding_mode = padding_mode
        self.delta = float(delta)
        self.eps = float(eps)

    @property
    def has_active_penalties(self) -> bool:
        return self.weight_seam > 0.0 or self.weight_biharmonic > 0.0 or self.weight_psd > 0.0

    def compute_patch_seam_ratio(self, img: torch.Tensor) -> torch.Tensor:
        """Computes the patch seam gradient ratio R_seam = E_seam[g] / max(E_interior[g], eps)."""
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img must be a torch.Tensor, got {type(img)}")
        if img.ndim != 4:
            raise ValueError(f"img must be 4D (B, C, H, W), got {img.shape}")
        B, C, H, W = img.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Image dimensions ({H}, {W}) must be >= patch_size ({self.patch_size})")

        delta_sq = self.delta ** 2
        dh = img[:, :, 1:, :] - img[:, :, :-1, :]
        dw = img[:, :, :, 1:] - img[:, :, :, :-1]
        # Align backward differences at pixel coordinates (y, x) using top and left padding
        gy = F.pad((dh.square() + delta_sq).sqrt() - self.delta, (0, 0, 1, 0))
        gx = F.pad((dw.square() + delta_sq).sqrt() - self.delta, (1, 0, 0, 0))
        grad_mag = (gx + gy).mean(dim=1)  # (B, H, W)

        y_coords = torch.arange(H, device=img.device).unsqueeze(1).expand(H, W)
        x_coords = torch.arange(W, device=img.device).unsqueeze(0).expand(H, W)

        seam_y = (y_coords % self.patch_size == 0)
        seam_x = (x_coords % self.patch_size == 0)
        if self.secondary_patch_size is not None and self.secondary_patch_size > 0:
            seam_y = seam_y | (y_coords % self.secondary_patch_size == 0)
            seam_x = seam_x | (x_coords % self.secondary_patch_size == 0)

        seam_mask_2d = seam_y | seam_x
        border_mask = (y_coords == 0) | (y_coords == H - 1) | (x_coords == 0) | (x_coords == W - 1)
        seam_mask = seam_mask_2d & (~border_mask)
        interior_mask = (~seam_mask_2d) & (~border_mask)

        mean_seam = grad_mag[:, seam_mask].mean()
        mean_interior = grad_mag[:, interior_mask].mean()
        return mean_seam / mean_interior.clamp_min(self.eps)

    # Alias for flexibility
    compute_seam_ratio = compute_patch_seam_ratio

    def forward(
        self,
        img: torch.Tensor,
        *,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"img must be a torch.Tensor, got {type(img)}")
        if img.ndim != 4:
            raise ValueError(f"img must be 4D (B, C, H, W), got {img.shape}")
        if not img.is_floating_point():
            raise TypeError(f"img must be floating point, got {img.dtype}")

        B, C, H, W = img.shape
        device = img.device
        dtype = img.dtype
        losses: Dict[str, torch.Tensor] = {}

        # 1. Patch-seam gradient penalty
        if self.weight_seam > 0.0:
            R_seam = self.compute_patch_seam_ratio(img)
            loss_seam = torch.relu(R_seam - self.seam_threshold).square()
            losses["loss_seam"] = loss_seam
            losses["R_seam"] = R_seam
        else:
            loss_seam = img.new_zeros(())
            losses["loss_seam"] = loss_seam
            losses["R_seam"] = img.new_zeros(())

        # 2. Biharmonic curvature penalty
        if self.weight_biharmonic > 0.0:
            if self.laplacian_kernel == "9point":
                k = torch.tensor([
                    [1.0, 4.0, 1.0],
                    [4.0, -20.0, 4.0],
                    [1.0, 4.0, 1.0],
                ], device=device, dtype=dtype) / 6.0
            else:
                k = torch.tensor([
                    [0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0],
                ], device=device, dtype=dtype)
            weight = k.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
            if self.padding_mode in ("linear", "reflect"):
                img_padded = _pad_linear_2d(img)
            elif self.padding_mode == "replicate":
                img_padded = F.pad(img, (1, 1, 1, 1), mode="replicate")
            else:
                img_padded = F.pad(img, (1, 1, 1, 1), mode="constant")
            lap = F.conv2d(img_padded, weight, groups=C)
            delta_sq = self.delta ** 2
            loss_biharmonic = ((lap.square() + delta_sq).sqrt() - self.delta).mean()
            losses["loss_biharmonic"] = loss_biharmonic
        else:
            loss_biharmonic = img.new_zeros(())
            losses["loss_biharmonic"] = loss_biharmonic

        # 3. Radial power spectral density (PSD) excess penalty
        if self.weight_psd > 0.0:
            compute_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
            img_c = img.to(compute_dtype)

            fy = fft.fftfreq(H, device=device).unsqueeze(1) * H
            fx = fft.rfftfreq(W, device=device).unsqueeze(0) * W
            f_rad = torch.sqrt(fy ** 2 + fx ** 2)

            max_r = min(H, W) // 2
            r_idx = f_rad.round().long().clamp(0, max_r)
            r_flat = r_idx.view(-1)
            num_bins = max_r + 1

            counts = torch.bincount(r_flat, minlength=num_bins).to(dtype=compute_dtype).clamp_min(1.0)
            F_x = fft.rfft2(img_c, norm="ortho")
            psd = (F_x.real ** 2 + F_x.imag ** 2).mean(dim=1)  # (B, H, W//2 + 1)

            psd_flat = psd.reshape(B, -1)
            zero_init = torch.zeros(B, num_bins, device=device, dtype=compute_dtype)
            radial_sum = zero_init.scatter_add(1, r_flat.unsqueeze(0).expand(B, -1), psd_flat)
            radial_psd = radial_sum / counts.unsqueeze(0)

            eps = 1e-8
            ref_power = radial_psd[:, 1:2].clamp_min(1e-7)
            r_vals = torch.arange(1, num_bins, device=device, dtype=compute_dtype)
            target_power = ref_power.detach() / (r_vals / r_vals[0]).unsqueeze(0).pow(self.alpha)

            margin_nats = self.margin_db * (math.log(10.0) / 10.0)
            log_ratio = torch.log(radial_psd[:, 1:] + eps) - torch.log(target_power + eps)
            ac_power = psd.sum(dim=(-2, -1)) - psd[:, 0, 0]
            gate = torch.tanh(ac_power / 1e-3).view(B, 1)
            loss_psd = (gate * torch.relu(log_ratio - margin_nats).square()).mean().to(dtype)
            losses["loss_psd"] = loss_psd
        else:
            loss_psd = img.new_zeros(())
            losses["loss_psd"] = loss_psd

        total_loss = (
            self.weight_seam * loss_seam
            + self.weight_biharmonic * loss_biharmonic
            + self.weight_psd * loss_psd
        )
        losses["total_loss"] = total_loss
        losses["loss"] = total_loss

        if return_dict:
            return losses
        return total_loss


class SobolevWavefrontManipulator(PriorManipulator):
    """PriorManipulator connecting Sobolev gradient preconditioning and Wavefront prior.

    Connects:
    1. SobolevGradFilter in `postprocess_generated(generated)` when grad_lambda > 0.
    2. WavefrontManifoldPrior in `regularize(raw_rows, ..., generated=generated)`.
    """
    row_local_gradients: bool = True

    def __init__(
        self,
        *,
        grad_lambda: float = 1.0,
        grad_order: float = 1.0,
        grad_nyquist: float = math.pi,
        weight: float = 1.0,
        weight_seam: float = 1.0,
        weight_biharmonic: float = 0.1,
        weight_psd: float = 0.01,
        patch_size: int = 4,
        secondary_patch_size: Optional[int] = None,
        seam_threshold: float = 1.05,
        laplacian_kernel: str = "5point",
        alpha: float = 2.0,
        margin_db: float = 1.0,
        padding_mode: str = "reflect",
        delta: float = 0.01,
    ) -> None:
        super().__init__()
        self.grad_lambda = float(grad_lambda)
        self.grad_order = float(grad_order)
        self.grad_nyquist = float(grad_nyquist)
        self.weight = float(weight)

        if self.grad_lambda > 0.0 and self.grad_order > 0.0:
            self.grad_filter = SobolevGradFilter(
                lam=self.grad_lambda,
                order=self.grad_order,
                nyquist=self.grad_nyquist,
            )
        else:
            self.grad_filter = None

        self.wavefront_prior = WavefrontManifoldPrior(
            weight_seam=weight_seam,
            weight_biharmonic=weight_biharmonic,
            weight_psd=weight_psd,
            patch_size=patch_size,
            secondary_patch_size=secondary_patch_size,
            seam_threshold=seam_threshold,
            laplacian_kernel=laplacian_kernel,
            alpha=alpha,
            margin_db=margin_db,
            padding_mode=padding_mode,
            delta=delta,
        )

    def postprocess_generated(self, generated: torch.Tensor) -> torch.Tensor:
        """Applies SobolevGradFilter to fake_g during the G-step."""
        if self.grad_lambda > 0.0 and self.grad_filter is not None:
            return self.grad_filter(generated)
        return generated

    def regularize(
        self,
        raw_rows: torch.Tensor,
        *,
        sampled_codes: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluates WavefrontManifoldPrior on fake_g and scales by self.weight."""
        if generated is None or self.weight <= 0.0 or not self.wavefront_prior.has_active_penalties:
            return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)
        loss = self.wavefront_prior(generated)
        return (self.weight * loss).to(device=raw_rows.device, dtype=raw_rows.dtype)


class GhostFreeSimplexMixer(nn.Module):
    """Ghost-free base+detail simplex de-mixing operator.

    Inputs:
        u_base: (K, C, H, W) coarse base representations with shared spatial support.
        u_full: (K, C, H, W) full mode representations containing localized details.
        weights: (B, K) simplex routing weights satisfying w_k >= 0, sum_k w_k = 1.

    Returns:
        u: (B, C, H, W) blended spatial carrier with exact zero negative leakage.
    """

    def __init__(
        self,
        *,
        gamma_base: float = 1.0,
        gamma_detail: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(gamma_base) or gamma_base < 0.0:
            raise ValueError(f"gamma_base must be finite nonnegative float, got {gamma_base}")
        if not math.isfinite(gamma_detail) or gamma_detail < 0.0:
            raise ValueError(f"gamma_detail must be finite nonnegative float, got {gamma_detail}")
        self.gamma_base = float(gamma_base)
        self.gamma_detail = float(gamma_detail)

    def compute_basis(self, u_base: torch.Tensor, u_full: torch.Tensor) -> torch.Tensor:
        """Computes de-mixed basis fields B_k."""
        if u_base.shape != u_full.shape:
            raise ValueError(
                f"u_base shape {tuple(u_base.shape)} must match u_full shape {tuple(u_full.shape)}"
            )
        u_base_mean = u_base.mean(dim=0, keepdim=True)
        return u_base_mean + self.gamma_base * (u_base - u_base_mean) + self.gamma_detail * (u_full - u_base)

    def forward(
        self,
        u_base: torch.Tensor,
        u_full: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        B_k = self.compute_basis(u_base, u_full)
        if weights.ndim != 2:
            raise ValueError(f"weights must be 2D tensor of shape (B, K), got ndim={weights.ndim}")
        K = u_base.shape[0]
        if weights.shape[1] != K:
            raise ValueError(
                f"weights mode dimension {weights.shape[1]} must match carrier modes K={K}"
            )
        return torch.einsum("bk, k... -> b...", weights, B_k)


class GhostFreeDeMixingManipulator(PriorManipulator):
    """Prior manipulator applying GhostFreeSimplexMixer to arbitrary mode representations."""

    def __init__(
        self,
        num_modes: int,
        representation_shape: Sequence[int],
        *,
        gamma_base: float = 1.0,
        gamma_detail: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.representation_shape = tuple(representation_shape)
        prod_dim = 1
        for dim in self.representation_shape:
            prod_dim *= dim
        self.prod_dim = prod_dim

        self.mixer = GhostFreeSimplexMixer(gamma_base=gamma_base, gamma_detail=gamma_detail)

        if learnable:
            self.base_repr = nn.Parameter(torch.randn(num_modes, prod_dim) * 0.1)
            self.detail_repr = nn.Parameter(torch.zeros(num_modes, prod_dim))
        else:
            self.register_buffer("base_repr", torch.zeros(num_modes, prod_dim))
            self.register_buffer("detail_repr", torch.zeros(num_modes, prod_dim))

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        if codes.shape[-1] < self.num_modes:
            raise ValueError(
                f"codes dimension {codes.shape[-1]} smaller than num_modes {self.num_modes}"
            )
        raw_w = codes[:, : self.num_modes]
        row_sums = raw_w.sum(dim=-1)
        if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3) and (raw_w >= -1e-5).all():
            w = raw_w.clamp_min(0.0)
            w = w / w.sum(dim=-1, keepdim=True)
        else:
            w = torch.softmax(raw_w, dim=-1)

        u_base = self.base_repr.view(self.num_modes, *self.representation_shape)
        u_full = u_base + self.detail_repr.view(self.num_modes, *self.representation_shape)

        blended = self.mixer(u_base, u_full, w)
        if codes.shape[-1] > self.num_modes:
            res = codes[:, self.num_modes :]
            try:
                blended.residuals = res
            except Exception:
                pass
        return blended


class DifferentiableSplatCarrierBasis(nn.Module):
    """Domain-agnostic learnable 2D Gaussian splat spatial carrier basis.

    All learnable parameters are strictly 2D Parameter matrices (ndim == 2)
    assigned to role="router" for native DualNorm optimization.
    """

    def __init__(
        self,
        num_modes: int = 18,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        num_base_splats: int = 6,
        num_detail_splats: int = 4,
    ) -> None:
        super().__init__()
        if type(num_modes) is not int or num_modes < 2:
            raise ValueError(f"num_modes must be integer >= 2, got {num_modes}")
        if type(channels) is not int or channels < 1:
            raise ValueError(f"channels must be positive integer, got {channels}")
        if type(height) is not int or height < 4 or type(width) is not int or width < 4:
            raise ValueError(f"height and width must be integers >= 4, got ({height}, {width})")
        if type(num_base_splats) is not int or num_base_splats < 1:
            raise ValueError(f"num_base_splats must be positive integer, got {num_base_splats}")
        if type(num_detail_splats) is not int or num_detail_splats < 0:
            raise ValueError(f"num_detail_splats must be nonnegative integer, got {num_detail_splats}")

        self.num_modes = num_modes
        self.channels = channels
        self.height = height
        self.width = width
        self.num_base_splats = num_base_splats
        self.num_detail_splats = num_detail_splats

        # 1. base_geometry: Parameter[M_base, 5]
        # Initialized radially with broad spatial envelopes (log_sigma = -1.0)
        bg = torch.zeros(num_base_splats, 5)
        for m in range(num_base_splats):
            angle = 2.0 * math.pi * m / float(num_base_splats)
            bg[m, 0] = 0.3 * math.sin(angle)
            bg[m, 1] = 0.3 * math.cos(angle)
            bg[m, 2] = math.log(0.35)
            bg[m, 3] = math.log(0.35)
            bg[m, 4] = 0.0
        self.base_geometry = nn.Parameter(bg)

        # 2. base_appearance: Parameter[K, M_base * C]
        self.base_appearance = nn.Parameter(
            torch.randn(num_modes, num_base_splats * channels) * (0.2 / math.sqrt(num_base_splats))
        )

        # 3. base_background: Parameter[K, 3 * C]
        # Initialized with zero slopes and slight positive offset
        bg_init = torch.zeros(num_modes, 3 * channels)
        bg_init[:, 0::3] = 0.05  # c0
        self.base_background = nn.Parameter(bg_init)

        # 4. detail_params: Parameter[K, M_detail * (5 + C)]
        dp = torch.zeros(num_modes, num_detail_splats * (5 + channels))
        for k in range(num_modes):
            for m in range(num_detail_splats):
                off = m * (5 + channels)
                # Scattered initial centers
                dp[k, off + 0] = 0.4 * math.sin(2.0 * math.pi * (m + k * 0.3) / max(1, num_detail_splats))
                dp[k, off + 1] = 0.4 * math.cos(2.0 * math.pi * (m + k * 0.3) / max(1, num_detail_splats))
                # Localized standard deviation
                dp[k, off + 2] = math.log(0.15)
                dp[k, off + 3] = math.log(0.15)
                dp[k, off + 4] = 0.0
                # Color offsets initialized small
                dp[k, off + 5 : off + 5 + channels] = torch.randn(channels) * 0.02
        self.detail_params = nn.Parameter(dp)

        # Coordinate grid [-1, 1]^2
        # CRITICAL: Must be .contiguous().clone() so policy.finish_step() buffer copy_ succeeds
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("grid_y", grid_y.contiguous().clone())
        self.register_buffer("grid_x", grid_x.contiguous().clone())

    def _render_gaussians(
        self,
        mu_y: torch.Tensor,
        mu_x: torch.Tensor,
        log_sigma_y: torch.Tensor,
        log_sigma_x: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable 2D Gaussian envelope rendering."""
        shape_prefix = mu_y.shape
        gy = self.grid_y.view(*([1] * len(shape_prefix)), self.height, self.width)
        gx = self.grid_x.view(*([1] * len(shape_prefix)), self.height, self.width)

        my = mu_y.view(*shape_prefix, 1, 1)
        mx = mu_x.view(*shape_prefix, 1, 1)
        lsy = log_sigma_y.view(*shape_prefix, 1, 1).clamp(-4.0, 2.0)
        lsx = log_sigma_x.view(*shape_prefix, 1, 1).clamp(-4.0, 2.0)
        th = theta.view(*shape_prefix, 1, 1)

        dy = gy - my
        dx = gx - mx
        cos_t = torch.cos(th)
        sin_t = torch.sin(th)

        y_rot = cos_t * dy + sin_t * dx
        x_rot = -sin_t * dy + cos_t * dx

        sy = torch.exp(lsy)
        sx = torch.exp(lsx)

        dist_sq = (y_rot / sy) ** 2 + (x_rot / sx) ** 2
        return torch.exp(-0.5 * dist_sq)

    def render(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Renders U^base and U^full across all K modes."""
        # 1. Base shared spatial envelopes: (M_base, H, W)
        E_base = self._render_gaussians(
            self.base_geometry[:, 0],
            self.base_geometry[:, 1],
            self.base_geometry[:, 2],
            self.base_geometry[:, 3],
            self.base_geometry[:, 4],
        )

        # 2. Base appearance combination: (K, M_base, C)
        A_base = self.base_appearance.view(self.num_modes, self.num_base_splats, self.channels)
        U_base_splats = torch.einsum("kmc, mhw -> kchw", A_base, E_base)

        # 3. Base planar background harmonics: (K, C, 3)
        bg = self.base_background.view(self.num_modes, self.channels, 3)
        c0 = bg[:, :, 0:1, None]
        sy = bg[:, :, 1:2, None]
        sx = bg[:, :, 2:3, None]
        U_bg = c0 + sy * self.grid_y.unsqueeze(0).unsqueeze(0) + sx * self.grid_x.unsqueeze(0).unsqueeze(0)

        U_base = U_base_splats + U_bg

        # 4. Detail anisotropic mode splats: (K, M_detail, 5 + C)
        if self.num_detail_splats > 0:
            dp = self.detail_params.view(self.num_modes, self.num_detail_splats, 5 + self.channels)
            E_detail = self._render_gaussians(
                dp[:, :, 0],
                dp[:, :, 1],
                dp[:, :, 2],
                dp[:, :, 3],
                dp[:, :, 4],
            )  # (K, M_detail, H, W)
            Delta_c = dp[:, :, 5 : 5 + self.channels]  # (K, M_detail, C)
            D = torch.einsum("kmc, kmhw -> kchw", Delta_c, E_detail)
            U_full = U_base + D
        else:
            U_full = U_base

        return U_base, U_full


class SplatCarrierManipulator(PriorManipulator):
    """Domain-agnostic learnable 2D Gaussian splat spatial carrier manipulator.

    Chains DifferentiableSplatCarrierBasis with GhostFreeSimplexMixer.
    All learnable splat parameters are strictly 2D Parameter matrices (ndim == 2)
    assigned to role="router" in Recipe.make_optimizers.
    """

    def __init__(
        self,
        num_modes: int = 18,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        num_base_splats: int = 6,
        num_detail_splats: int = 4,
        *,
        basis: Optional[DifferentiableSplatCarrierBasis] = None,
        gamma_base: float = 1.0,
        gamma_detail: float = 1.0,
        concat_residuals: bool = False,
    ) -> None:
        super().__init__()
        if basis is not None:
            self.basis = basis
            self.num_modes = basis.num_modes
            self.channels = basis.channels
            self.height = basis.height
            self.width = basis.width
        else:
            self.num_modes = num_modes
            self.channels = channels
            self.height = height
            self.width = width
            self.basis = DifferentiableSplatCarrierBasis(
                num_modes=num_modes,
                channels=channels,
                height=height,
                width=width,
                num_base_splats=num_base_splats,
                num_detail_splats=num_detail_splats,
            )
        self.concat_residuals = bool(concat_residuals)
        self.mixer = GhostFreeSimplexMixer(
            gamma_base=gamma_base,
            gamma_detail=gamma_detail,
        )

    def transform(
        self,
        codes: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        *,
        base_prior: nn.Module,
        generator: Optional[torch.Generator] = None,
        training: bool = True,
    ) -> torch.Tensor:
        if codes.shape[-1] < self.num_modes:
            raise ValueError(
                f"codes dimension {codes.shape[-1]} smaller than num_modes {self.num_modes}"
            )

        raw_w = codes[:, : self.num_modes]
        row_sums = raw_w.sum(dim=-1)
        if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3) and (raw_w >= -1e-5).all():
            w = raw_w.clamp_min(0.0)
            w = w / w.sum(dim=-1, keepdim=True)
        else:
            w = torch.softmax(raw_w, dim=-1)

        u_base, u_full = self.basis.render()
        carrier = self.mixer(u_base, u_full, w)

        if self.concat_residuals and codes.shape[-1] > self.num_modes:
            residuals = codes[:, self.num_modes :]
            res_spatial = residuals.view(codes.shape[0], -1, 1, 1).expand(
                -1, -1, self.height, self.width
            )
            out = torch.cat([carrier, res_spatial], dim=1)
        else:
            out = carrier

        try:
            out.weights = w.detach()
            out.u_base = u_base.detach()
        except Exception:
            pass

        return out


class ResidualConvBlock(nn.Module):
    """Residual convolution block with GroupNorm and SiLU activation."""

    def __init__(self, channels: int, num_groups: int = 4) -> None:
        super().__init__()
        groups = min(num_groups, channels)
        while channels % groups != 0:
            groups -= 1
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.norm1(self.conv1(x)))
        res = self.norm2(self.conv2(res))
        return self.act2(x + res)


class SimplexWavefrontGenerator(nn.Module):
    """Generator combining spatial carrier with multi-scale conv residual pyramid.

    Inputs:
        carrier: (B, C, H, W) spatial carrier or (B, K) simplex routing weights.

    Returns:
        image: (B, C, H, W) bounded synthesized output in (-1, 1).
    """

    def __init__(
        self,
        channels: int = 3,
        hidden_dim: int = 32,
        num_blocks: int = 2,
        residual_gain: float = 0.5,
        num_groups: int = 4,
        in_channels: Optional[int] = None,
        carrier_basis: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.carrier_basis = carrier_basis
        if carrier_basis is not None:
            self.channels = getattr(carrier_basis, "channels", channels)
        else:
            self.channels = channels
        self.in_channels = in_channels if in_channels is not None else self.channels
        self.hidden_dim = hidden_dim
        self.residual_gain = float(residual_gain)
        self.mixer = GhostFreeSimplexMixer()

        # 1. Initial feature projection
        self.input_conv = nn.Conv2d(self.in_channels, hidden_dim, 3, padding=1)

        # 2. Multi-scale residual pyramid blocks
        self.blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, num_groups=num_groups)
            for _ in range(num_blocks)
        ])

        # 3. Dilated context branch (dilation=2)
        groups = min(num_groups, hidden_dim)
        while hidden_dim % groups != 0:
            groups -= 1
        self.dilated_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2)
        self.dilated_norm = nn.GroupNorm(groups, hidden_dim)
        self.dilated_act = nn.SiLU()

        # 4. Zero-initialized final residual conv
        self.final_conv = nn.Conv2d(hidden_dim, self.channels, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def render_carrier(self, weights: torch.Tensor, *, raw: bool = False) -> torch.Tensor:
        """Renders carrier from weights using configured carrier_basis."""
        if self.carrier_basis is None:
            raise RuntimeError("render_carrier requires carrier_basis to be configured")
        u_base, u_full = self.carrier_basis.render()
        carrier = self.mixer(u_base, u_full, weights)
        return carrier if raw else torch.tanh(carrier)

    def get_final_conv(self) -> nn.Conv2d:
        """Returns the final convolutional projection layer."""
        return self.final_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            carrier = self.render_carrier(x, raw=True)
        elif x.ndim == 4:
            carrier = x
        else:
            raise ValueError(
                f"SimplexWavefrontGenerator expects 2D (B, K) or 4D (B, C, H, W) tensor, got shape {tuple(x.shape)}"
            )

        feat = self.input_conv(carrier)
        for block in self.blocks:
            feat = block(feat)

        coarse = self.dilated_act(self.dilated_norm(self.dilated_conv(feat)))
        feat = feat + coarse

        res = self.final_conv(feat)
        bounded_res = self.residual_gain * torch.tanh(res)

        base_carrier = carrier[:, : self.channels]
        out = torch.tanh(base_carrier + bounded_res)
        return out
