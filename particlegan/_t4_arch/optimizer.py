"""SVD-free Depth-Equilibrated & Coherence-Gated Trust Optimizer (Theory 4B).

Solves Deep Layerwise Jacobian Exponential Skew (``10^4x``--``10^8x`` across 16
unnormalized layers or zero-initialized residual branches) without ``torch.linalg.svd``
or ``optimizer_smoothing`` dead-zone floors:

1. **Underflow-Safe Scale-Normalized First & Second Moments**:
   Tracks ``(m, v_rms)`` normalized dynamically by ``max(amax(v_rms), amax(|g|), 1e-38)``
   so ``10^-12`` early-layer gradients in 16-layer skip-free MLPs and ``10^-6``
   ``tanh`` rail gradients never underflow in FP32/BF16.
2. **Scale-Normalized 5-Step Quintic Newton-Schulz Orthogonalization (Zero SVD)**:
   Equalizes all active singular directions of 2D dense weights and 4D per-offset
   convolution kernels via 5 pure GEMM/BMM iterations on ``m_hat / ||m_hat||_F``
   (BF16 on large CUDA matrices ``min(m, n) >= 32``, FP32 otherwise).
3. **Momentum Coherence Gate & Zero-Branch-Safe Layerwise Trust Radius**:
   Scales each layer step by directional coherence ``||m_hat||_F / ||v_hat||_F in [0, 1]``
   (damping oscillatory equilibrium noise to zero without any step clock), composite
   depth factor ``min(1, (3 / max(3, L)) * wakeup_boost)``, and trust radius
   ``clamp(rms(W_l), (1 / sqrt(L)) * w_orth_l, 2.5 * w_orth_l)``.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer

from ..grad_regularizers import CriticStepRecord
from ..optim.dualnorm import NormalizedOptimizer

_T4_ROLES = frozenset({"generator", "encoder", "router", "noise", "critic", "prior", "table"})
_QUINTIC_A = 3.4445
_QUINTIC_B = -4.7750
_QUINTIC_C = 2.0315


def quintic_orthogonalize_2d(mat: torch.Tensor, *, steps: int = 5) -> torch.Tensor:
    """Scale-normalized 5-step Quintic Newton-Schulz polar factor for a 2D matrix."""
    orig_dtype = mat.dtype
    use_bf16 = mat.is_cuda and min(mat.shape[0], mat.shape[1]) >= 32
    x = mat.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    max_abs = x.abs().amax().clamp_min(1e-38)
    x = x / max_abs
    norm = x.norm().clamp_min(1e-12)
    x = x / norm
    if use_bf16:
        x = x.to(torch.bfloat16)
    for _ in range(steps):
        gram = x @ x.T
        x = _QUINTIC_A * x + (_QUINTIC_B * gram + _QUINTIC_C * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(orig_dtype)


def quintic_orthogonalize_batched(mats: torch.Tensor, *, steps: int = 5) -> torch.Tensor:
    """Scale-normalized batched 5-step Quintic Newton-Schulz for [B, M, N] conv slices."""
    orig_dtype = mats.dtype
    use_bf16 = mats.is_cuda and min(mats.shape[1], mats.shape[2]) >= 16
    x = mats.float()
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(1, 2)
    max_abs = x.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-38)
    x = x / max_abs
    norms = x.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
    x = x / norms
    if use_bf16:
        x = x.to(torch.bfloat16)
    for _ in range(steps):
        gram = torch.bmm(x, x.transpose(1, 2))
        gram2 = torch.bmm(gram, gram)
        x = _QUINTIC_A * x + torch.bmm(_QUINTIC_B * gram + _QUINTIC_C * gram2, x)
    if transposed:
        x = x.transpose(1, 2)
    return x.to(orig_dtype)


class T4EquilibratedOptimizer(NormalizedOptimizer):
    """Recipe-owned Theory 4B SVD-free depth-equilibrated trust optimizer (``t4_equil``)."""

    add_param_group = Optimizer.add_param_group

    def bind_origin_sample(self, sample: Any) -> None:
        del sample

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 0.015,
        betas: tuple[float, float] = (0.80, 0.97),
        eps: float = 1e-8,
        amsgrad: bool = False,
        **unused_options: Any,
    ) -> None:
        del unused_options
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
            raise ValueError("invalid t4_equil learning rate")
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not (0.0 <= float(b) < 1.0) for b in betas)
        ):
            raise ValueError("invalid t4_equil betas")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("invalid t4_equil eps")

        materialized = list(params)
        if materialized and not isinstance(materialized[0], dict):
            materialized = [{"params": materialized, "role": "generator"}]

        prepared_groups: List[Dict[str, Any]] = []
        for raw in materialized:
            grp = dict(raw)
            grp_params = list(grp.get("params", ()))
            if not grp_params:
                continue
            role = grp.get("role", grp.get("forge_role", "generator"))
            if role not in _T4_ROLES:
                raise ValueError(f"unsupported t4_equil role: {role!r}")
            grp["params"] = grp_params
            grp["role"] = role
            grp["algorithm"] = "t4_prior" if role in ("prior", "table") else "t4_equil"
            grp["sampled_rows_required"] = role in ("prior", "table")
            prepared_groups.append(grp)

        defaults = {
            "lr": float(lr),
            "betas": (float(betas[0]), float(betas[1])),
            "eps": float(eps),
            "amsgrad": bool(amsgrad),
            "role": "generator",
            "algorithm": "t4_equil",
            "sampled_rows_required": False,
        }
        Optimizer.__init__(self, prepared_groups, defaults)

        has_prior = any(g["role"] in ("prior", "table") for g in self.param_groups)
        self.family = "t4_equil"
        self.prior_mechanisms = {
            "a2": {
                "requested": False,
                "enabled": False,
                "implemented": False,
                "owner": "t4_equil",
                "has_prior_group": has_prior,
            }
        }
        self.latent_damping = None
        self.latent_history = None
        self.direct_response = None
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}
        self._row_scales: Dict[torch.Tensor, torch.Tensor] = {}
        self._t4_wakeup_boost: float = 1.0
        self._t4_g_scale: float = 1.0
        self._t4_prior_scale: float = 1.0

    def set_sampled_rows(self, parameter: torch.Tensor, rows: torch.Tensor) -> None:
        if not isinstance(rows, torch.Tensor):
            rows = torch.as_tensor(rows, dtype=torch.long, device=parameter.device)
        rows = torch.unique(rows.to(dtype=torch.long, device=parameter.device), sorted=True)
        if rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(parameter)).any())):
            raise ValueError("sampled rows out of bounds for prior table")
        self._sampled_rows[parameter] = rows

    def sampled_rows_for(self, parameter: torch.Tensor) -> Optional[torch.Tensor]:
        return self._sampled_rows.get(parameter)

    def set_row_scales(self, parameter: torch.Tensor, scales: torch.Tensor) -> None:
        self._row_scales[parameter] = scales.detach().to(device=parameter.device, dtype=torch.float32)

    def clear_sampled_rows(self) -> None:
        self._sampled_rows.clear()
        self._row_scales.clear()

    def abort_pending(self) -> None:
        self.clear_sampled_rows()
        self._t4_wakeup_boost = 1.0
        self._t4_g_scale = 1.0
        self._t4_prior_scale = 1.0

    def bind_t4_modulation(self, *, g_scale: float = 1.0, wakeup_boost: float = 1.0, prior_scale: float = 1.0) -> None:
        self._t4_g_scale = float(g_scale)
        self._t4_wakeup_boost = float(wakeup_boost)
        self._t4_prior_scale = float(prior_scale)

    def _player(self, role: str) -> str:
        if role in ("prior", "table"):
            return "prior"
        if role == "critic":
            return "critic"
        return "generator"

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        try:
            n_mat: Dict[str, int] = {}
            total_gen_mats = 0
            for grp in self.param_groups:
                rl = grp["role"]
                pl = self._player(rl)
                for p in grp["params"]:
                    if p.ndim >= 2:
                        n_mat[rl] = n_mat.get(rl, 0) + 1
                        if pl == "generator":
                            total_gen_mats += 1

            rec = getattr(self, "_t2_record", getattr(self, "_empf_record", getattr(self, "record", None)))
            empf_scale = float(getattr(rec, "_step_scale", 1.0)) if rec is not None else 1.0

            for grp in self.param_groups:
                role = grp["role"]
                player = self._player(role)
                b1, b2 = grp["betas"]
                L_raw = max(1, n_mat.get(role, 1))
                L_count = 4 if (player == "generator" and L_raw == 2 and total_gen_mats >= 4) else L_raw
                # Shallow 1-layer fixtures (e.g. direct coordinate clouds) keep snappy momentum;
                # multi-layer networks use multi-step directional coherence smoothing.
                if player == "prior":
                    b1_eff = max(float(b1), 0.80)
                elif L_count <= 1:
                    b1_eff = min(float(b1), 0.20)
                else:
                    b1_eff = max(float(b1), 0.75)
                b2_eff = min(max(float(b2), 0.94), 0.99)
                c_zero = 1.0 / math.sqrt(float(L_count))

                if player == "generator":
                    depth_scale = min(1.0, (3.0 / float(max(3, L_count))) * self._t4_wakeup_boost)
                    base_lr = float(grp["lr"]) * empf_scale * self._t4_g_scale * depth_scale
                elif player == "critic":
                    depth_scale = 3.0 / float(max(3, L_count))
                    base_lr = float(grp["lr"]) * empf_scale * depth_scale
                else:
                    base_lr = float(grp["lr"]) * empf_scale * self._t4_prior_scale

                pending_dense: Dict[tuple[int, int], List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
                pending_conv: Dict[tuple[int, int, int, int], List[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}

                for p in grp["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    st = self.state[p]
                    step_t = int(st.get("step", 0)) + 1
                    st["step"] = step_t

                    if player == "prior":
                        if "m" not in st:
                            st["m"] = torch.zeros_like(p)
                            st["v_row"] = torch.zeros((len(p), 1), device=p.device, dtype=torch.float32)
                            st["row_step"] = torch.zeros((len(p), 1), device=p.device, dtype=torch.float32)
                        rows = self._sampled_rows.get(p)
                        if rows is None:
                            rows = torch.arange(len(p), device=p.device, dtype=torch.long)
                        if rows.numel() == 0:
                            continue
                        gr = grad[rows].float()
                        fin_row = torch.isfinite(gr).all(dim=-1, keepdim=True)
                        gr_safe = torch.where(fin_row, gr, torch.zeros_like(gr))
                        m_r = st["m"][rows].float()
                        v_r = st["v_row"][rows]
                        g_norm = gr_safe.norm(dim=-1, keepdim=True)
                        scale = torch.maximum(v_r, g_norm).clamp_min(1e-38)
                        m_new = torch.where(fin_row, m_r.mul(b1_eff).add(gr_safe, alpha=1.0 - b1_eff), m_r)
                        v_upd = (b2_eff * (v_r / scale).pow(2) + (1.0 - b2_eff) * (g_norm / scale).pow(2)).sqrt().mul_(scale)
                        v_new = torch.where(fin_row, v_upd, v_r)
                        st["m"][rows] = m_new.to(p.dtype)
                        st["v_row"][rows] = v_new
                        rs = torch.where(fin_row, st["row_step"][rows] + 1.0, st["row_step"][rows])
                        st["row_step"][rows] = rs
                        rs_safe = rs.clamp_min(1.0)
                        m_hat = m_new / (1.0 - b1_eff ** rs_safe)
                        v_hat = v_new / (1.0 - b2_eff ** rs_safe).sqrt()
                        denom = torch.maximum(v_hat, scale * 1e-12).clamp_min(1e-38)
                        row_coh = (m_hat.norm(dim=-1, keepdim=True) / denom).clamp(0.25, 1.0)
                        step_vec = torch.where(
                            fin_row,
                            (m_hat / denom) * row_coh * math.sqrt(float(max(1, p.shape[1]))),
                            torch.zeros_like(m_hat),
                        )
                        r_scales = self._row_scales.get(p)
                        if r_scales is not None and len(r_scales) == len(rows):
                            step_vec = step_vec * r_scales.unsqueeze(1)
                        p.index_add_(0, rows, step_vec.to(p.dtype), alpha=-base_lr)
                        continue

                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                        st["v_rms"] = torch.zeros_like(p)
                    m = st["m"]
                    v_rms = st["v_rms"]
                    gf = grad.float()
                    finite = torch.isfinite(gf).all()
                    gf_safe = torch.where(finite, gf, torch.zeros_like(gf))
                    g_max = gf_safe.abs().amax().clamp_min(1e-38)
                    scale = torch.maximum(v_rms.float().amax(), g_max).clamp_min(1e-38)
                    m_next = m.float().mul(b1_eff).add(gf_safe, alpha=1.0 - b1_eff)
                    v_scaled = b2_eff * (v_rms.float() / scale).pow(2) + (1.0 - b2_eff) * (gf_safe / scale).pow(2)
                    v_next = v_scaled.sqrt().mul_(scale)
                    m.copy_(torch.where(finite, m_next, m.float()).to(p.dtype))
                    v_rms.copy_(torch.where(finite, v_next, v_rms.float()).to(p.dtype))

                    bc1 = 1.0 - b1_eff ** step_t
                    bc2_sqrt = math.sqrt(1.0 - b2_eff ** step_t)
                    m_scaled = (m.float() / scale) / bc1
                    v_scaled_hat = (v_rms.float() / scale) / bc2_sqrt
                    m_norm = m_scaled.norm()
                    v_norm = v_scaled_hat.norm()
                    active = finite & (v_norm > 1e-30) & (m_norm > 1e-30)
                    coherence = torch.where(
                        active,
                        (m_norm / v_norm.clamp_min(1e-30)).clamp(0.0, 1.0),
                        torch.zeros_like(m_norm),
                    )

                    if p.ndim == 4 and (p.shape[2] > 1 or p.shape[3] > 1):
                        c0, c1, kh, kw = p.shape
                        k_area = kh * kw
                        w_op = 1.0 / math.sqrt(float(math.sqrt(float(c0 * c1)) * float(k_area)))
                        w_rms = p.float().pow(2).mean().sqrt()
                        r_trust = w_rms.clamp(c_zero * w_op, 2.5 * w_op)
                        step_mult = (-base_lr) * coherence * r_trust
                        pending_conv.setdefault((c0, c1, kh, kw), []).append((p, m_scaled, step_mult))
                    elif p.ndim >= 2:
                        mat = m_scaled.reshape(p.shape[0], -1)
                        w_op = 1.0 / math.sqrt(float(max(1, mat.shape[1])))
                        w_rms = p.float().pow(2).mean().sqrt()
                        r_trust = w_rms.clamp(c_zero * w_op, 2.5 * w_op)
                        step_mult = (-base_lr) * coherence * r_trust
                        pending_dense.setdefault((mat.shape[0], mat.shape[1]), []).append((p, mat, step_mult))
                    else:
                        denom = torch.maximum(v_scaled_hat, torch.full_like(v_scaled_hat, 1e-12))
                        u = m_scaled / denom
                        u_rms = u.pow(2).mean().sqrt().clamp_min(1.0)
                        u = (u / u_rms) * ((-base_lr) * coherence)
                        p.add_(u.to(p.dtype))

                for _, bucket in pending_dense.items():
                    if len(bucket) == 1:
                        p_i, mat_i, mult_i = bucket[0]
                        orth_i = quintic_orthogonalize_2d(mat_i, steps=5).float()
                        orth_rms = orth_i.pow(2).mean().sqrt().clamp_min(1e-8)
                        p_i.add_((orth_i * (mult_i / orth_rms)).reshape_as(p_i).to(p_i.dtype))
                    else:
                        mats = torch.stack([b_item[1] for b_item in bucket], dim=0)
                        orths = quintic_orthogonalize_batched(mats, steps=5).float()
                        orth_rms_all = orths.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
                        for idx_b, (p_i, _, mult_i) in enumerate(bucket):
                            p_i.add_((orths[idx_b] * (mult_i / orth_rms_all[idx_b])).reshape_as(p_i).to(p_i.dtype))

                for (c0, c1, kh, kw), bucket in pending_conv.items():
                    k_area = kh * kw
                    slices_list = [b_item[1].permute(2, 3, 0, 1).reshape(k_area, c0, c1) for b_item in bucket]
                    all_slices = torch.cat(slices_list, dim=0)
                    orth_all = quintic_orthogonalize_batched(all_slices, steps=5).float()
                    orth_layers = orth_all.view(len(bucket), k_area, c0, c1)
                    for idx_b, (p_i, _, mult_i) in enumerate(bucket):
                        orth_i = orth_layers[idx_b].permute(1, 2, 0).reshape(c0, c1, kh, kw)
                        orth_rms = orth_i.pow(2).mean().sqrt().clamp_min(1e-8)
                        p_i.add_((orth_i * (mult_i / orth_rms)).to(p_i.dtype))
        finally:
            self.abort_pending()

        if hasattr(self, "record") and self.record is not None:
            self.record.record_step(self)
        return loss

    def state_dict(self) -> Dict[str, Any]:
        result = Optimizer.state_dict(self)
        result["t4_equil"] = {
            "schema": 1,
            "family": self.family,
            "sampled_rows": [
                self._sampled_rows.get(grp["params"][0]) if grp["algorithm"] == "t4_prior" else None
                for grp in self.param_groups
            ],
        }
        if hasattr(self, "record") and self.record is not None:
            result["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return result

    def validate_state_dict(self, saved: Any) -> None:
        expected_keys = {"state", "param_groups", "t4_equil"}
        if hasattr(self, "record") and self.record is not None:
            expected_keys.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            raise ValueError("invalid T4EquilibratedOptimizer checkpoint schema")
        meta = saved["t4_equil"]
        if (
            not isinstance(meta, dict)
            or set(meta) != {"schema", "family", "sampled_rows"}
            or type(meta["schema"]) is not int
            or meta["schema"] != 1
            or meta["family"] != self.family
        ):
            raise ValueError("invalid T4EquilibratedOptimizer metadata")

        groups, values = saved["param_groups"], saved["state"]
        if (
            not isinstance(groups, list)
            or len(groups) != len(self.param_groups)
            or not isinstance(values, dict)
            or not isinstance(meta["sampled_rows"], list)
            or len(meta["sampled_rows"]) != len(groups)
        ):
            raise ValueError("invalid T4EquilibratedOptimizer parameter groups")

        seen = set()
        for idx, (actual, grp) in enumerate(zip(self.param_groups, groups)):
            if not isinstance(grp, dict) or set(grp) != set(actual) or not isinstance(grp.get("params"), list):
                raise ValueError("invalid T4EquilibratedOptimizer group fields")
            if len(grp["params"]) != len(actual["params"]):
                raise ValueError("checkpoint T4EquilibratedOptimizer group sizes differ")
            for key in set(actual) - {"params", "lr"}:
                if grp[key] != actual[key]:
                    raise ValueError(f"checkpoint T4EquilibratedOptimizer group {key} differs")
            if isinstance(grp["lr"], bool) or not math.isfinite(grp["lr"]) or grp["lr"] < 0:
                raise ValueError("invalid T4EquilibratedOptimizer group lr")
            rows = meta["sampled_rows"][idx]
            if rows is not None:
                if (
                    actual["algorithm"] != "t4_prior"
                    or not isinstance(rows, torch.Tensor)
                    or rows.dtype != torch.long
                    or rows.ndim != 1
                    or (rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(actual["params"][0])).any())))
                    or len(torch.unique(rows)) != len(rows)
                ):
                    raise ValueError("invalid T4EquilibratedOptimizer sampled rows")

            for pid, param in zip(grp["params"], actual["params"]):
                if type(pid) is not int or pid in seen:
                    raise ValueError("invalid T4EquilibratedOptimizer parameter ID")
                seen.add(pid)
                st = values.get(pid, {})
                if not isinstance(st, dict):
                    raise ValueError("invalid T4EquilibratedOptimizer parameter state")
                if not st:
                    continue
                if actual["algorithm"] == "t4_prior":
                    if set(st) not in ({"step"}, {"step", "m", "v_row", "row_step"}):
                        raise ValueError("invalid T4EquilibratedOptimizer prior state keys")
                    if "m" in st:
                        m_t, vr_t, rs_t = st["m"], st["v_row"], st["row_step"]
                        if not isinstance(m_t, torch.Tensor) or m_t.shape != param.shape or m_t.dtype != param.dtype or not bool(torch.isfinite(m_t).all()):
                            raise ValueError("invalid T4 prior m tensor")
                        for t_chk, name_chk in ((vr_t, "v_row"), (rs_t, "row_step")):
                            if (
                                not isinstance(t_chk, torch.Tensor)
                                or t_chk.shape != (len(param), 1)
                                or t_chk.dtype != torch.float32
                                or not bool(torch.isfinite(t_chk).all())
                                or bool((t_chk < 0).any())
                            ):
                                raise ValueError(f"invalid T4 prior {name_chk} tensor")
                else:
                    if set(st) not in ({"step"}, {"step", "m", "v_rms"}):
                        raise ValueError("invalid T4EquilibratedOptimizer dense state keys")
                    if "m" in st:
                        for name_chk in ("m", "v_rms"):
                            t_chk = st[name_chk]
                            if (
                                not isinstance(t_chk, torch.Tensor)
                                or t_chk.shape != param.shape
                                or t_chk.dtype != param.dtype
                                or not bool(torch.isfinite(t_chk).all())
                            ):
                                raise ValueError(f"invalid T4 {name_chk} tensor")
                        if bool((st["v_rms"] < 0).any()):
                            raise ValueError("negative T4 v_rms history")

                step_val = st["step"]
                if isinstance(step_val, torch.Tensor):
                    if step_val.numel() != 1 or not bool(torch.isfinite(step_val).all()) or bool(step_val < 0) or bool(step_val != step_val.floor()):
                        raise ValueError("invalid T4 step tensor")
                elif type(step_val) is not int or step_val < 0:
                    raise ValueError("invalid T4 step integer")

        if not values.keys() <= seen:
            raise ValueError("unknown T4EquilibratedOptimizer parameter history")

        if hasattr(self, "record") and self.record is not None:
            reg = saved["regularizer"]
            if (
                not isinstance(reg, dict)
                or set(reg) != {"optimizer_family", "record", "ema", "guard"}
                or reg["optimizer_family"] != self.family
                or reg["ema"] is not None
                or reg["guard"] is not None
            ):
                raise ValueError("invalid T4 critic regularizer state")
            rec_state = reg["record"]
            if hasattr(self.record, "base_state"):
                rec_state = self.record.base_state(rec_state)
            if not isinstance(rec_state, dict) or set(rec_state) != set(CriticStepRecord.KEYS):
                raise ValueError("invalid T4 critic record")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.validate_state_dict(state)
        copied = deepcopy(state)
        meta = copied.pop("t4_equil")
        regularizer = copied.pop("regularizer", None)
        Optimizer.load_state_dict(self, copied)
        # Ensure all floating/auxiliary tensors in parameter state reside on the live parameter device
        for grp in self.param_groups:
            for p in grp["params"]:
                st = self.state.get(p)
                if st:
                    if "step" in st and isinstance(st["step"], torch.Tensor):
                        st["step"] = int(st["step"].item())
                    for k, v in list(st.items()):
                        if isinstance(v, torch.Tensor) and v.device != p.device:
                            st[k] = v.to(device=p.device)
        self.abort_pending()
        for grp, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None:
                self.set_sampled_rows(grp["params"][0], rows)
        if regularizer is not None and hasattr(self, "record") and self.record is not None:
            self.record.load_state_dict(regularizer["record"])
            if hasattr(self.record, "restore_device_state") and self.param_groups and self.param_groups[0]["params"]:
                self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "T4EquilibratedOptimizer",
    "quintic_orthogonalize_2d",
    "quintic_orthogonalize_batched",
]
