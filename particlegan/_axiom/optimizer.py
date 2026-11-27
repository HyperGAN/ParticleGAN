"""Axiom L1 Optimizer: Law 4.1-4.5 Role-Agnostic Equilibrated Dynamics."""

from __future__ import annotations

from copy import copy, deepcopy
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from .math_ops import (
    fp32_tikhonov_ns_2d,
    fp32_tikhonov_ns_batched,
    mantissa_exponent_normalize,
    me_unit_rows,
    schatten6_decay_batched,
    smoothstep01,
    solve_nonascent_multipliers,
    stable_rank_power,
    zero_pinv_project_nonascent,
)
from .state import (
    SCHEMA_VERSION,
    clear_active_prior_rows,
    eval_probe_forward,
    get_tensor_record,
    set_active_prior_rows,
)

_ALLOWED_ROLES = frozenset(
    {"generator", "encoder", "router", "noise", "critic", "prior", "table"}
)
_IGNORED_OPTIONS = frozenset(
    {
        "weight_decay",
        "amsgrad",
        "maximize",
        "differentiable",
        "capturable",
        "foreach",
        "fused",
    }
)
_FP32_STATE_KEYS: Tuple[str, ...] = ("v_pow", "m_unit", "m_scale", "m_prior")
_SKETCH_DIM: int = 67
_LOG2_DAMP_FLOOR: float = -4.0
_PYTH_CAP: float = 1.5
_WEIGHT_LOG2_FLOOR: float = math.log2(1e-12)


def collect_conv_layouts(
    modules: Iterable[Optional[nn.Module]],
) -> Dict[int, Tuple[bool, int]]:
    """Map Conv2d/ConvTranspose2d weight ``id(w)`` to ``(is_transpose, groups)``."""
    layouts: Dict[int, Tuple[bool, int]] = {}
    for root in modules:
        if not isinstance(root, nn.Module):
            continue
        for mod in root.modules():
            if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
                is_t = isinstance(mod, nn.ConvTranspose2d)
                layouts[id(mod.weight)] = (is_t, int(mod.groups))
    return layouts


def _flat_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor.view(-1)
    if tensor.ndim == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
        return tensor.permute(0, 2, 3, 1).view(-1)
    return tensor.reshape(-1)


def _is_fin_num(
    val: Any,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    strict_lo: bool = False,
) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    fv = float(val)
    if not math.isfinite(fv):
        return False
    if lo is not None and (fv <= lo if strict_lo else fv < lo):
        return False
    return hi is None or fv <= hi


def _update_l2_ref(
    st: Dict[str, Any],
    lt: float,
    g_eq: float,
    d_eq: float,
    has_weight: bool,
) -> float:
    """Huber-capped (+2 octave) cosine-gated log-norm reference tracker."""
    if not bool(st.get("log2_init", False)):
        if has_weight:
            st["log2_init"] = True
            st["log2_ref"] = float(lt)
        return float(lt)
    rp = float(st.get("log2_ref", 0.0))
    if lt > rp:
        rn = rp + 0.25 * min(2.0, lt - rp)
    else:
        d_eff = 0.25 * (1.0 - g_eq) + d_eq * g_eq
        rn = rp - d_eff * min(2.0, rp - lt)
    st["log2_ref"] = rn
    return rn


def _apply_momentum(
    st: Dict[str, Any], param: torch.Tensor, gu: torch.Tensor, b1: float
) -> torch.Tensor:
    """Degree-0 unit-gradient FP32 momentum (active only when ``beta1 > 0``)."""
    if b1 <= 0.0:
        return gu
    mu = st.setdefault("m_unit", torch.zeros_like(param, dtype=torch.float32))
    ms = st.setdefault(
        "m_scale", torch.zeros((), device=param.device, dtype=torch.float32)
    )
    gu_new, _, _ = mantissa_exponent_normalize(b1 * mu + (1.0 - b1) * gu)
    mu.copy_(gu_new)
    ms.fill_(1.0)
    return gu_new


def _collect_reachable_params(loss: torch.Tensor, param_ids: set[int]) -> set[int]:
    """Identify leaf parameters reachable from ``loss.grad_fn`` without autograd.grad."""
    root = getattr(loss, "grad_fn", None)
    if root is None:
        return set()
    touched: set[int] = set()
    stack = [root]
    seen = {id(root)}
    while stack:
        node = stack.pop()
        var = getattr(node, "variable", None)
        if isinstance(var, torch.Tensor) and id(var) in param_ids:
            touched.add(id(var))
        for nxt, _ in getattr(node, "next_functions", ()):
            if nxt is not None and id(nxt) not in seen:
                seen.add(id(nxt))
                stack.append(nxt)
    return touched


def _streaming_param_dot(
    xs: Sequence[Optional[torch.Tensor]],
    ys: Sequence[Optional[torch.Tensor]],
    z0: torch.Tensor,
) -> torch.Tensor:
    acc = z0
    for x_p, y_p in zip(xs, ys):
        if x_p is not None and y_p is not None:
            acc = acc + (x_p.double() * y_p.double()).sum()
    return acc


def _is_active_scalar_loss(val: Any) -> bool:
    return isinstance(val, torch.Tensor) and val.ndim == 0 and bool(val.requires_grad)


def _blend_dot_products(
    disps: List[torch.Tensor],
    normals: List[List[Optional[torch.Tensor]]],
) -> Tuple[List[float], List[float], float, float, float]:
    z0 = torch.zeros((), dtype=torch.float64, device=disps[0].device)
    d2_t = _streaming_param_dot(disps, disps, z0)
    n2_0 = _streaming_param_dot(normals[0], normals[0], z0)
    ad_0 = _streaming_param_dot(normals[0], disps, z0)
    if len(normals) == 1:
        n2_v, ad_v, d2_f = torch.stack((n2_0, ad_0, d2_t)).tolist()
        return [ad_v], [n2_v], 0.0, d2_f, ad_v
    n2_1 = _streaming_param_dot(normals[1], normals[1], z0)
    ad_1 = _streaming_param_dot(normals[1], disps, z0)
    a01_t = _streaming_param_dot(normals[0], normals[1], z0)
    v = torch.stack((n2_0, n2_1, ad_0, ad_1, a01_t, d2_t)).tolist()
    return [v[2], v[3]], [v[0], v[1]], v[4], v[5], max(v[2], v[3])


def _apply_blend_step(
    all_p: List[torch.Tensor],
    origs: List[torch.Tensor],
    disps: List[torch.Tensor],
    normals: List[List[Optional[torch.Tensor]]],
    alphas: Tuple[float, ...],
) -> None:
    active = [(a, grp) for a, grp in zip(alphas, normals) if a != 0.0]
    for idx, (p, o, d_p) in enumerate(zip(all_p, origs, disps)):
        acc = d_p.double()
        for alpha, grp in active:
            a_p = grp[idx]
            if a_p is not None:
                acc = acc - alpha * a_p.double()
        p.copy_(o + acc.to(p.dtype))


def _blend_post_metrics(
    ad: List[float],
    a_n2: List[float],
    a01: float,
    d2: float,
    alphas: Tuple[float, ...],
) -> Tuple[float, float]:
    if len(ad) == 1:
        a0 = alphas[0]
        after = ad[0] - a0 * a_n2[0]
        n2 = d2 - 2.0 * a0 * ad[0] + a0 * a0 * a_n2[0]
        return after, max(0.0, n2)
    a0, a1 = alphas[0], alphas[1]
    after = max(
        ad[0] - a0 * a_n2[0] - a1 * a01,
        ad[1] - a0 * a01 - a1 * a_n2[1],
    )
    n2 = (
        d2
        - 2.0 * (a0 * ad[0] + a1 * ad[1])
        + a0 * a0 * a_n2[0]
        + a1 * a1 * a_n2[1]
        + 2.0 * a0 * a1 * a01
    )
    return after, max(0.0, n2)


class AxiomOptimizer(Optimizer):
    """Law 4.1-4.5 role-agnostic, BN- & scale-invariant equilibrated optimizer (schema 2)."""

    family = recipe_optimizer_family = "axiom"
    _SUPPORTS_ZEROSYNC_V4_TENSORS: bool = True

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 0.014,
        betas: Tuple[float, float] = (0.0, 0.999),
        eps: float = 1e-8,
        smoothing: float = 0.05,
        direction_blend: bool = False,
        critic: Optional[nn.Module] = None,
        record: Any = None,
        conv_layouts: Optional[Dict[int, Tuple[bool, int]]] = None,
        **options: Any,
    ) -> None:
        self._validate_init_args(lr, betas, eps, smoothing, direction_blend, options)
        self.family = self.recipe_optimizer_family = "axiom"
        self.smoothing = float(smoothing)
        self.direction_blend = bool(direction_blend)
        self.critic = critic
        self.record = self._axiom_record = record
        self.ema_critic = self.anchor = self.guard = None
        self.latent_damping = self.latent_history = None
        self.direct_response = self.direct_history = None
        self._conv_layouts = dict(conv_layouts) if conv_layouts else {}
        self._sampled_rows: Dict[torch.Tensor, torch.Tensor] = {}
        self._sampled_rows_unique: Dict[torch.Tensor, torch.Tensor] = {}
        self._row_scales: Dict[torch.Tensor, torch.Tensor] = {}
        self._gen_param_ids: set[int] = set()
        self._metrics_cache: Dict[int, Tuple[float, float, int, int, int]] = {}
        self._mod_g_scale: float | torch.Tensor = 1.0
        self._mod_wakeup_boost: float | torch.Tensor = 1.0
        self._mod_prior_scale = self.last_step_gain = self._ema_func_gain = 1.0
        self._ema_step_cos_val = self._log2_damp_val = self._crit_skew_ema = 0.0
        self._ema_step_cos_t: Optional[torch.Tensor] = None
        self._log2_damp_t: Optional[torch.Tensor] = None
        self._prev_step_sketch: Optional[torch.Tensor] = None
        self._fast_cache: Optional[Dict[str, Any]] = None
        self._protected: Optional[List[List[Optional[torch.Tensor]]]] = None
        self.last_probe_consumed = False
        self.constraint_geometry_stats = {
            "steps": 0,
            "projected_steps": 0,
            "max_derivative_before": 0.0,
            "max_derivative_after": 0.0,
        }
        self.direction_blend_stats = {
            "conflict_steps": 0,
            "blended_steps": 0,
            "pareto_stalls": 0,
            "retained_norm_ratio_sum": 0.0,
            "max_retained_norm_ratio": 0.0,
        }
        self.ns_2d_activations = self.fast_path_steps = 0
        self.trust_backtracks_total = self.probe_consumptions = 0

        default_role = "critic" if critic is not None else "generator"
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(b) for b in betas),
            "eps": float(eps),
            "role": default_role,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "differentiable": False,
            "capturable": False,
            "foreach": None,
            "fused": None,
            **options,
        }
        super().__init__(params, defaults)
        self._configure_param_groups(default_role)
        if isinstance(critic, nn.Module):
            self.bind_module_layouts((critic,))

    @staticmethod
    def _validate_init_args(
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        smoothing: float,
        direction_blend: bool,
        options: Dict[str, Any],
    ) -> None:
        valid_scalars = (
            _is_fin_num(lr, 0.0)
            and _is_fin_num(eps, 0.0, strict_lo=True)
            and _is_fin_num(smoothing, 0.0)
        )
        if not valid_scalars:
            raise ValueError("invalid AxiomOptimizer scalar hyperparameter")
        if type(direction_blend) is not bool or len(betas) != 2:
            raise ValueError("invalid AxiomOptimizer betas or direction_blend")
        if any(not _is_fin_num(b, 0.0) or float(b) >= 1.0 for b in betas):
            raise ValueError("invalid AxiomOptimizer betas")
        for key, val in options.items():
            if key not in _IGNORED_OPTIONS:
                raise TypeError(f"unsupported AxiomOptimizer option: {key}")
            if val not in (None, False, 0, 0.0):
                raise ValueError(f"AxiomOptimizer requires disabled {key}")

    def _configure_param_groups(self, default_role: str) -> None:
        for grp in self.param_groups:
            role = grp.get("role", grp.get("forge_role", default_role))
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"unknown optimizer parameter role: {role!r}")
            grp["role"] = role
            if role in ("prior", "table"):
                if len(grp["params"]) != 1 or grp["params"][0].ndim != 2:
                    raise ValueError("prior/table group must hold one 2D latent table")
                grp["algorithm"] = "axiom_prior"
                grp.setdefault("sampled_rows_required", role == "prior")
            else:
                grp["algorithm"] = "axiom"

    def bind_module_layouts(self, modules: Iterable[Optional[nn.Module]]) -> None:
        """Register Conv2d/ConvTranspose2d layouts for exact 1x1 == Linear parity."""
        self._conv_layouts.update(collect_conv_layouts(modules))
        self._metrics_cache.clear()

    def _active_record(self) -> Any:
        if self.record is not None:
            return self.record
        if self._axiom_record is not None:
            return self._axiom_record
        for grp in self.param_groups:
            for p in grp["params"]:
                rec = get_tensor_record(p)
                if rec is not None:
                    return rec
        return None

    def _parameters(self) -> List[torch.Tensor]:
        return [p for grp in self.param_groups for p in grp["params"]]

    def set_generator_probe_parameters(
        self, parameters: Iterable[torch.Tensor]
    ) -> None:
        self._gen_param_ids = {id(p) for p in parameters}

    def bind_origin_sample(self, sample: Optional[torch.Tensor]) -> None:
        del sample

    def _flush_scalars(self) -> None:
        if self._ema_step_cos_t is None and self._log2_damp_t is None:
            return
        tensors = [
            t.double()
            for t in (self._ema_step_cos_t, self._log2_damp_t)
            if t is not None
        ]
        vals = (
            torch.stack(tensors).tolist()
            if len(tensors) > 1
            else [float(tensors[0].item())]
        )
        idx = 0
        if self._ema_step_cos_t is not None:
            self._ema_step_cos_val, self._ema_step_cos_t = float(vals[idx]), None
            idx += 1
        if self._log2_damp_t is not None:
            self._log2_damp_val, self._log2_damp_t = float(vals[idx]), None

    def _flush_fast_cache(self) -> None:
        if self._fast_cache is None:
            return
        refs = self._fast_cache["log2_ref"].detach().cpu().tolist()
        inits = self._fast_cache["log2_init"].detach().cpu().tolist()
        for p, rv, iv in zip(self._fast_cache["params"], refs, inits):
            self.state[p]["log2_ref"] = float(rv)
            self.state[p]["log2_init"] = bool(iv)

    @property
    def ema_step_cos(self) -> float:
        self._flush_scalars()
        return self._ema_step_cos_val

    @property
    def log2_damp(self) -> float:
        self._flush_scalars()
        return self._log2_damp_val

    def _param_metrics(self, p: torch.Tensor) -> Tuple[float, float, int, int, int]:
        cached = self._metrics_cache.get(id(p))
        if cached is not None:
            return cached
        if p.ndim == 4:
            out = self._conv_param_metrics(p)
        elif p.ndim >= 2:
            rd = int(p.shape[0])
            cd = int(p.numel() // max(1, rd))
            md = min(rd, cd)
            fac = math.sqrt(max(1.0, rd / max(1, cd)))
            out = (fac, 0.5 * math.log2(max(1, md)), rd, cd, md)
        else:
            # S^0 (0 tangential DOF) uses exact half-bracket bisection, S^{n-1} uses 1.0.
            out = (0.5 if p.numel() == 1 else 1.0, 0.0, 1, 1, 1)
        self._metrics_cache[id(p)] = out
        return out

    def _conv_param_metrics(
        self, p: torch.Tensor
    ) -> Tuple[float, float, int, int, int]:
        c0, c1, kh, kw = (int(s) for s in p.shape)
        is_t, gr = self._conv_layouts.get(id(p), (False, 1))
        oc, ic = (c1 * gr, c0) if is_t else (c0, c1 * gr)
        if kh == 1 and kw == 1 and gr == 1 and not is_t:
            md = min(oc, ic)
            fac = math.sqrt(max(1.0, oc / max(1, ic)))
            return fac, 0.5 * math.log2(max(1, md)), oc, ic, md
        area = max(1, (kh * kw) // 4 if is_t else kh * kw)
        fan = max(1, min(oc // max(1, gr), ic // max(1, gr)))
        fac = math.sqrt(oc / max(1, ic)) * math.sqrt(gr / area) * (fan**0.25)
        return fac, 0.5 * math.log2(max(1, min(oc, ic) * kh * kw)), 1, 1, 1

    def set_sampled_rows(self, param: torch.Tensor, rows: torch.Tensor) -> None:
        """Register active prior table rows via weak registry (zero attr mutation)."""
        rows_t = (
            torch.as_tensor(rows).reshape(-1).to(dtype=torch.long, device=param.device)
        )
        if rows_t.device.type == "cpu" and rows_t.numel():
            if bool((rows_t < 0).any()) or bool((rows_t >= len(param)).any()):
                raise ValueError("sampled rows out of bounds for prior table")
        prev = self._sampled_rows.get(param)
        self._sampled_rows[param] = (
            rows_t if prev is None else torch.cat((prev, rows_t))
        )
        self._sampled_rows_unique.pop(param, None)
        set_active_prior_rows(param, self._sampled_rows[param])
        rec = self._active_record()
        if rec is not None:
            rec._active_prior_table = param
            rec._active_batch_rows = rows_t.detach()

    def sampled_rows_for(self, param: torch.Tensor) -> Optional[torch.Tensor]:
        if param not in self._sampled_rows:
            return None
        uniq = self._sampled_rows_unique.get(param)
        if uniq is None:
            uniq = torch.unique(self._sampled_rows[param], sorted=True)
            self._sampled_rows_unique[param] = uniq
        return uniq.detach().clone()

    def clear_sampled_rows(self) -> None:
        for p in self._sampled_rows:
            clear_active_prior_rows(p)
        self._sampled_rows.clear()
        self._sampled_rows_unique.clear()
        self._row_scales.clear()
        rec = self._active_record()
        if rec is not None:
            rec._active_prior_table = rec._active_batch_rows = None

    def set_row_scales(
        self, param: torch.Tensor, scales: Optional[torch.Tensor]
    ) -> None:
        if scales is None:
            self._row_scales.pop(param, None)
        else:
            self._row_scales[param] = scales.detach().to(
                device=param.device, dtype=torch.float32
            )

    def bind_axiom_modulation(
        self,
        *,
        g_scale: float | torch.Tensor = 1.0,
        wakeup_boost: float | torch.Tensor = 1.0,
        prior_scale: float = 1.0,
    ) -> None:
        self._mod_g_scale = (
            g_scale.detach().float().reshape(())
            if isinstance(g_scale, torch.Tensor)
            else float(g_scale)
        )
        self._mod_wakeup_boost = (
            wakeup_boost.detach().float().reshape(())
            if isinstance(wakeup_boost, torch.Tensor)
            else float(wakeup_boost)
        )
        self._mod_prior_scale = float(prior_scale)

    bind_v4_modulation = bind_axiom_modulation

    def _mask_protected_piece(
        self,
        src: Optional[torch.Tensor],
        param: torch.Tensor,
        *,
        is_prior_sampled: bool,
        clone: bool,
    ) -> Optional[torch.Tensor]:
        if src is None:
            return None
        if is_prior_sampled:
            rows = self.sampled_rows_for(param)
            if rows is None:
                raise ValueError("direction_blend requires explicit sampled rows")
            piece = torch.zeros_like(src)
            piece[rows] = src[rows]
        else:
            piece = src.clone() if clone else src
        if not bool(torch.isfinite(piece).all()):
            raise ValueError("nonfinite direction_blend protected gradient")
        return piece

    def _collect_protected_normal(
        self,
        *,
        overrides: Optional[Dict[int, Optional[torch.Tensor]]] = None,
        clone: bool = False,
    ) -> List[Optional[torch.Tensor]]:
        normal: List[Optional[torch.Tensor]] = []
        for grp in self.param_groups:
            is_ps = grp["algorithm"] == "axiom_prior" and bool(
                grp.get("sampled_rows_required", True)
            )
            for p in grp["params"]:
                if overrides is not None and id(p) in overrides:
                    src = overrides[id(p)]
                    use_clone = False
                else:
                    src = None if p.grad is None else p.grad.detach()
                    use_clone = clone
                normal.append(
                    self._mask_protected_piece(
                        src, p, is_prior_sampled=is_ps, clone=use_clone
                    )
                )
        return normal

    def bind_protected_from_grads(self) -> None:
        """Bind current ``p.grad`` as exact 1-loss protected normal."""
        if not self.direction_blend:
            return
        if self._protected is not None:
            raise ValueError("Axiom direction_blend losses already pending")
        self._protected = [self._collect_protected_normal(clone=False)]

    def bind_protected_losses(
        self, losses: Sequence[torch.Tensor], *, protected_evaluator: Any = None
    ) -> None:
        """Bind 1 or 2 scalar protected losses via ``.backward(retain_graph=True)``."""
        del protected_evaluator
        if not self.direction_blend:
            return
        if self._protected is not None:
            raise ValueError("Axiom direction_blend losses already pending")
        lt = tuple(losses)
        if not 1 <= len(lt) <= 2 or not all(_is_active_scalar_loss(x) for x in lt):
            raise ValueError("one or two active scalar protected losses required")
        params = self._parameters()
        trainable = [p for p in params if p.requires_grad]
        saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
        normals: List[List[Optional[torch.Tensor]]] = []
        try:
            for lk in lt:
                for p in params:
                    p.grad = None
                if trainable:
                    lk.backward(retain_graph=True, inputs=trainable)
                normals.append(self._collect_protected_normal(clone=True))
        finally:
            for p, sg in zip(params, saved):
                p.grad = sg
        self._protected = normals

    def backward_with_protected(
        self,
        loss_gan: torch.Tensor,
        aux_loss: Optional[torch.Tensor] = None,
    ) -> None:
        """Single-backward exact 1-loss protection without cloning untouched G grads."""
        if not _is_active_scalar_loss(loss_gan):
            raise ValueError("backward_with_protected requires active scalar loss_gan")
        if self.direction_blend and self._protected is not None:
            raise ValueError("Axiom direction_blend losses already pending")
        has_aux = _is_active_scalar_loss(aux_loss)
        loss_gan.backward()
        if not self.direction_blend:
            if has_aux:
                aux_loss.backward()  # type: ignore[union-attr]
            return
        params = self._parameters()
        touched = (
            _collect_reachable_params(aux_loss, {id(p) for p in params})  # type: ignore[arg-type]
            if has_aux
            else set()
        )
        saved_gan = {
            id(p): (None if p.grad is None else p.grad.detach().clone())
            for p in params
            if id(p) in touched
        }
        if has_aux:
            aux_loss.backward()  # type: ignore[union-attr]
        self._protected = [self._collect_protected_normal(overrides=saved_gan)]

    def abort_pending(self) -> None:
        self.clear_sampled_rows()
        self._mod_g_scale = self._mod_wakeup_boost = self._mod_prior_scale = 1.0
        self._protected = None

    def _build_step_sketch(
        self, net_pairs: List[Tuple[torch.Tensor, torch.Tensor]], dev: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows: List[torch.Tensor] = []
        for _, g in net_pairs:
            if g.numel() <= 0:
                continue
            fl = _flat_view(g.detach())
            n = int(fl.numel())
            if n >= _SKETCH_DIM:
                stride = max(1, n // _SKETCH_DIM)
                if stride > 1 and stride % 2 == 0:
                    stride -= 1
                rows.append(fl[::stride][:_SKETCH_DIM].float())
            else:
                pad = fl.new_zeros(_SKETCH_DIM - n, dtype=torch.float32)
                rows.append(torch.cat((fl.float(), pad), dim=0))
        if not rows:
            z = torch.zeros(_SKETCH_DIM, dtype=torch.float32, device=dev)
            return z, torch.zeros((), dtype=torch.bool, device=dev)
        unit_rows, _, _ = me_unit_rows(torch.stack(rows, dim=0).to(dev))
        acc = unit_rows.sum(dim=0)
        sk_n = acc.norm()
        sk_ok = torch.isfinite(sk_n) & (sk_n > 0.0)
        return torch.where(
            sk_ok, acc / sk_n.clamp_min(1e-30), torch.zeros_like(acc)
        ), sk_ok

    def _update_step_cosine(
        self,
        net_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        dev: torch.device,
        ema_fg: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        rho_p = (
            self._ema_step_cos_t
            if (self._ema_step_cos_t is not None and self._ema_step_cos_t.device == dev)
            else torch.full((), self._ema_step_cos_val, dtype=torch.float64, device=dev)
        )
        damp_p = (
            self._log2_damp_t
            if (self._log2_damp_t is not None and self._log2_damp_t.device == dev)
            else torch.full((), self._log2_damp_val, dtype=torch.float64, device=dev)
        )
        had_hist = self._prev_step_sketch is not None
        sk_now, sk_ok = self._build_step_sketch(net_pairs, dev)
        prev_sk = self._prev_step_sketch
        if prev_sk is not None:
            prev_dev = prev_sk.to(device=dev, dtype=torch.float32)
            cos_cur = (sk_now * prev_dev).sum().double().clamp(-1.0, 1.0)
            rho_n = torch.where(sk_ok, 0.90 * rho_p + 0.10 * cos_cur, rho_p)
            self._prev_step_sketch = torch.where(sk_ok, sk_now, prev_dev)
        else:
            rho_n = rho_p
            if net_pairs:
                self._prev_step_sketch = sk_now

        g_down = smoothstep01((-0.45 - rho_n) / 0.25)
        g_up = smoothstep01((rho_n + 0.35) / 0.35)
        g_osc = smoothstep01((-0.35 - rho_n) / 0.35)
        g_eq = smoothstep01((0.55 - rho_n) / 0.30)
        log2_d = (damp_p - 0.12 * g_down + 0.25 * g_up).clamp(_LOG2_DAMP_FLOOR, 0.0)
        self._ema_step_cos_t, self._log2_damp_t = rho_n, log2_d
        stiff = torch.maximum(
            rho_n.new_full((), smoothstep01((0.18 - ema_fg) / 0.07)),
            smoothstep01((0.60 - ema_fg) / 0.30) * g_down,
        )
        return torch.exp2(log2_d), float(self.smoothing) * g_osc, g_eq, stiff, had_hist

    @staticmethod
    def _critic_layer_skew(
        crit_m4: List[Tuple[Any, ...]], ref_dev: torch.device
    ) -> Optional[float]:
        can_batch = all(
            it[0].dtype == torch.float32
            and it[1].dtype == torch.float32
            and it[0].device == ref_dev
            for it in crit_m4
        )
        if can_batch:
            g_n = torch.stack(torch._foreach_norm([it[1] for it in crit_m4])).double()
            p_n = torch.stack(torch._foreach_norm([it[0] for it in crit_m4])).double()
            ok = (
                torch.isfinite(g_n)
                & (g_n > 1e-17)
                & (g_n < 1e17)
                & torch.isfinite(p_n)
                & (p_n < 1e17)
            )
            l2_p = torch.where(
                p_n > 1e-12, torch.log2(p_n.clamp_min(1e-12)), torch.zeros_like(p_n)
            )
            hl2 = torch.tensor(
                [float(it[4]) for it in crit_m4], dtype=torch.float64, device=ref_dev
            )
            lt = torch.log2(torch.where(ok, g_n, torch.ones_like(g_n))) + l2_p - hl2
            all_ok, sk_f = torch.stack(
                (ok.all().double(), lt.amax() - lt.amin())
            ).tolist()
            if all_ok > 0.5:
                return float(sk_f)
        l2_vals: List[float] = []
        for p, g, _, _, hl2, *_ in crit_m4:
            _, l2g, okg = mantissa_exponent_normalize(g)
            _, l2p, okp = mantissa_exponent_normalize(p.detach())
            if not bool(okg.item()):
                continue
            lp_v = float(l2p.item())
            lp_f = lp_v if (bool(okp.item()) and lp_v > _WEIGHT_LOG2_FLOOR) else 0.0
            l2_vals.append(float(l2g.item()) + lp_f - float(hl2))
        return (max(l2_vals) - min(l2_vals)) if len(l2_vals) >= 3 else None

    def _detect_critic_pathology(
        self, crit_m4: List[Tuple[Any, ...]], ref_dev: torch.device
    ) -> bool:
        """Compound Law 4.2 witness: layer skew > 4x AND min r_stable < 1.05."""
        if len(crit_m4) < 3:
            return False
        skew_now = self._critic_layer_skew(crit_m4, ref_dev)
        if skew_now is None:
            return False
        self._crit_skew_ema = max(
            skew_now, 0.97 * self._crit_skew_ema + 0.03 * skew_now
        )
        if self._crit_skew_ema <= 2.0:
            return False
        r_stabs: List[torch.Tensor] = []
        for p, g, _, _, _, rd, cd, *_ in crit_m4:
            st = self.state[p]
            r_s, v_next = stable_rank_power(
                g.reshape(rd, cd), iters=2, init_v=st.get("v_pow"), return_v=True
            )
            st["v_pow"] = v_next.detach()
            r_stabs.append(r_s.to(ref_dev))
        return min(torch.stack(r_stabs).tolist()) < 1.05

    @staticmethod
    def _critic_spec_factors(crit_2d: List[Tuple[Any, ...]]) -> List[torch.Tensor]:
        """Compute ``gamma_spec(G) = min(sqrt(m), 2*sqrt(r_stable(G)))`` per critic matrix."""
        buckets: Dict[
            Tuple[int, int, torch.device], List[Tuple[int, torch.Tensor, int]]
        ] = {}
        for idx, (_, g, _, _, _, rd, cd, md, _, _) in enumerate(crit_2d):
            buckets.setdefault((rd, cd, g.device), []).append(
                (idx, g.reshape(rd, cd), md)
            )
        out: List[Optional[torch.Tensor]] = [None] * len(crit_2d)
        for items in buckets.values():
            mats = torch.stack([it[1] for it in items], dim=0)
            r_stabs = stable_rank_power(mats, iters=2)
            md = items[0][2]
            cap = r_stabs.new_full(r_stabs.shape, math.sqrt(float(md)))
            gammas = torch.minimum(cap, 2.0 * r_stabs.sqrt()).double()
            for j, (orig_idx, _, _) in enumerate(items):
                out[orig_idx] = gammas[j]
        return [t for t in out if t is not None]

    def _two_obj_pyth(self) -> Dict[int, float]:
        if self._protected is None or len(self._protected) != 2:
            return {}
        n0, n1 = self._protected[0], self._protected[1]
        role_acc: Dict[
            str,
            Tuple[
                List[int], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]
            ],
        ] = {}
        idx = 0
        for grp in self.param_groups:
            pids, g0s, g1s = role_acc.setdefault(grp["role"], ([], [], []))
            for p in grp["params"]:
                pids.append(id(p))
                if p.grad is not None and p.numel() > 0:
                    g0s.append(n0[idx])
                    g1s.append(n1[idx])
                idx += 1
        result: Dict[int, float] = {}
        for pids, g0s, g1s in role_acc.values():
            ref = next((t for t in (*g0s, *g1s) if t is not None), None)
            if ref is None:
                continue
            z0 = torch.zeros((), dtype=torch.float64, device=ref.device)
            dots = (
                _streaming_param_dot(g0s, g0s, z0),
                _streaming_param_dot(g1s, g1s, z0),
                _streaming_param_dot(g0s, g1s, z0),
            )
            a2, b2, dt = torch.stack(dots).tolist()
            if a2 > 1e-24 and b2 > 1e-24 and math.isfinite(a2 + b2 + dt):
                den = math.sqrt(max(a2 + b2 + 2.0 * dt, 1e-36))
                boost = min(_PYTH_CAP, max(1.0, (math.sqrt(a2) + math.sqrt(b2)) / den))
                for pid in pids:
                    result[pid] = boost
        return result

    def _prep_prior(
        self, p: torch.Tensor, grad: torch.Tensor, grp: Dict[str, Any], b1: float
    ) -> Optional[Tuple[Any, ...]]:
        st, r_sc = self.state[p], self._row_scales.get(p)
        req = bool(grp.get("sampled_rows_required", True))
        if req and r_sc is None:
            raw = self._sampled_rows[p]
            if raw.numel() == 0:
                return None
            ridx, _ = torch.sort(raw)
            is_f = torch.ones(int(ridx.shape[0]), device=p.device, dtype=torch.bool)
            if ridx.shape[0] > 1:
                is_f[1:] = ridx[1:] != ridx[:-1]
            rmask = is_f.unsqueeze(1)
        else:
            ridx = (
                self.sampled_rows_for(p)
                if req
                else torch.arange(len(p), device=p.device, dtype=torch.long)
            )
            if ridx is None or ridx.numel() == 0:
                return None
            rmask = torch.ones(
                (int(ridx.shape[0]), 1), device=p.device, dtype=torch.bool
            )
        gr = grad[ridx].float()
        fin = torch.isfinite(gr).all(dim=-1, keepdim=True)
        dr = torch.where(fin, gr, torch.zeros_like(gr))
        if b1 > 0.0:
            mp = st.setdefault("m_prior", torch.zeros_like(p, dtype=torch.float32))
            mpv = mp[ridx]
            dr = torch.where(fin, mpv.mul(b1).add(dr, alpha=1.0 - b1), mpv)
            mp.index_add_(0, ridx, (dr - mpv) * rmask.float())
        u_row, l2, ok0 = me_unit_rows(dr)
        ok = rmask & ok0
        nok = ok.float().sum()
        mean_l2 = (l2 * ok.float()).sum() / nok.clamp_min(1.0)
        return p, ridx, r_sc, u_row, l2, ok, nok.double(), mean_l2.double()

    def _finish_prior(
        self,
        pr: Tuple[Any, ...],
        n_ok: float,
        mean_l2: float,
        *,
        base_lr: float,
        s_rel_pr: float,
        s_peak_pr: float,
        g_eq: float,
        d_eq: float,
        had_hist: bool,
    ) -> None:
        if base_lr <= 0.0:
            return
        p, ridx, r_sc, u_row, l2, ok, _, _ = pr
        st = self.state[p]
        ref_n = (
            _update_l2_ref(st, mean_l2, g_eq, d_eq, True)
            if n_ok > 0.0
            else float(st.get("log2_ref", 0.0))
        )
        if had_hist:
            rat = torch.exp2((ref_n - l2).clamp(-20.0, 20.0))
            rs = float(s_rel_pr) * rat + float(s_peak_pr) * (rat - 1.0).clamp_min(0.0)
            ur = torch.where(
                ok, u_row * torch.rsqrt(1.0 + rs.square()), torch.zeros_like(u_row)
            )
        else:
            ur = torch.where(ok, u_row, torch.zeros_like(u_row))
        if r_sc is not None and len(r_sc) == len(ridx):
            ur = ur * r_sc.unsqueeze(1)
        p.index_add_(0, ridx, ur.to(p.dtype), alpha=-base_lr)

    def _ensure_fast_cache(
        self, items: List[Tuple[Any, ...]], dev: torch.device
    ) -> Dict[str, Any]:
        params = [it[0] for it in items]
        key = tuple(id(p) for p in params)
        lr_key = tuple(it[2] for it in items)
        fac_key = tuple(it[3] for it in items)
        c = self._fast_cache
        needs_new = (
            c is None
            or c["key"] != key
            or c["device"] != dev
            or c["fac_key"] != fac_key
        )
        if needs_new:
            self._flush_fast_cache()
            c = {
                "key": key,
                "device": dev,
                "params": params,
                "lr_key": lr_key,
                "fac_key": fac_key,
                "lrs": torch.tensor(lr_key, dtype=torch.float64, device=dev),
                "factors": torch.tensor(fac_key, dtype=torch.float64, device=dev),
                "half_l2": torch.tensor(
                    [it[4] for it in items], dtype=torch.float64, device=dev
                ),
                "is_gen": torch.tensor(
                    [it[5] for it in items], dtype=torch.bool, device=dev
                ),
                "log2_ref": torch.tensor(
                    [float(self.state[p].get("log2_ref", 0.0)) for p in params],
                    dtype=torch.float64,
                    device=dev,
                ),
                "log2_init": torch.tensor(
                    [bool(self.state[p].get("log2_init", False)) for p in params],
                    dtype=torch.bool,
                    device=dev,
                ),
            }
            self._fast_cache = c
        elif c["lr_key"] != lr_key:
            c["lr_key"] = lr_key
            c["lrs"] = torch.tensor(lr_key, dtype=torch.float64, device=dev)
        return c

    @staticmethod
    def _compute_fast_damp(
        sn: torch.Tensor,
        p_norms: torch.Tensor,
        ok: torch.Tensor,
        c: Dict[str, Any],
        *,
        s_osc: torch.Tensor,
        s_base: torch.Tensor,
        s_peak: torch.Tensor,
        g_eq: torch.Tensor,
        d_eq: float,
        had_hist: bool,
    ) -> torch.Tensor:
        has_w = p_norms > 1e-12
        log2_p = torch.where(
            has_w, torch.log2(p_norms.clamp_min(1e-12)), torch.zeros_like(p_norms)
        )
        lt = torch.log2(sn) + log2_p - c["half_l2"]
        rp, ip = c["log2_ref"], c["log2_init"]
        d_eff = 0.25 * (1.0 - g_eq) + d_eq * g_eq
        up_step = rp + 0.25 * (lt - rp).clamp(0.0, 2.0)
        dn_step = rp - d_eff * (rp - lt).clamp(0.0, 2.0)
        rn = torch.where(ip, torch.where(lt > rp, up_step, dn_step), lt)
        c["log2_ref"].copy_(torch.where(ok & (ip | has_w), rn, rp))
        c["log2_init"].copy_(ip | (ok & has_w))
        if not had_hist:
            return torch.ones_like(sn)
        rat = torch.exp2((rn - lt).clamp(-20.0, 20.0))
        s_tikh = (
            g_eq * (s_base * rat + s_peak * (rat - 1.0).clamp_min(0.0)) + s_osc * rat
        )
        return torch.where(
            ip | has_w, torch.rsqrt(1.0 + s_tikh.square()), torch.ones_like(rat)
        )

    @staticmethod
    def _apply_partial_foreach(
        items: List[Tuple[Any, ...]],
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        sc: torch.Tensor,
        keep: List[int],
    ) -> List[Tuple[Any, ...]]:
        keep_p = [params[i] for i in keep]
        keep_g = [grads[i] for i in keep]
        keep_s = [sc[i] for i in keep]
        torch._foreach_addcmul_(keep_p, keep_g, keep_s)
        keep_set = set(keep)
        return [it for i, it in enumerate(items) if i not in keep_set]

    def _run_foreach_fast(
        self,
        items: List[Tuple[Any, ...]],
        *,
        c_osc: torch.Tensor,
        s_osc: torch.Tensor,
        s_base: torch.Tensor,
        s_peak: torch.Tensor,
        g_eq: torch.Tensor,
        d_eq: float,
        had_hist: bool,
        g_mult: float | torch.Tensor,
        d_mult: float | torch.Tensor,
    ) -> List[Tuple[Any, ...]]:
        params, grads = zip(*((it[0], it[1]) for it in items))
        p_list, g_list = list(params), list(grads)
        dev = p_list[0].device
        c = self._ensure_fast_cache(items, dev)
        g_norms = torch.stack(torch._foreach_norm(g_list)).double()
        p_norms = torch.stack(torch._foreach_norm(p_list)).double()
        ok = (
            torch.isfinite(g_norms)
            & (g_norms > 1e-17)
            & (g_norms < 1e17)
            & torch.isfinite(p_norms)
            & (p_norms < 1e17)
        )
        all_ok = bool(ok.all().item())
        keep = None if all_ok else [i for i, b in enumerate(ok.tolist()) if b]
        if keep is not None and not keep:
            return list(items)

        gm_t = (
            g_mult.to(dev, dtype=torch.float64)
            if isinstance(g_mult, torch.Tensor)
            else float(g_mult)
        )
        dm_t = (
            d_mult.to(dev, dtype=torch.float64)
            if isinstance(d_mult, torch.Tensor)
            else float(d_mult)
        )
        lrs = torch.where(c["is_gen"], c["lrs"] * gm_t, c["lrs"] * dm_t)
        sn = g_norms if all_ok else torch.where(ok, g_norms, torch.ones_like(g_norms))
        damp = self._compute_fast_damp(
            sn,
            p_norms,
            ok,
            c,
            s_osc=s_osc,
            s_base=s_base,
            s_peak=s_peak,
            g_eq=g_eq,
            d_eq=d_eq,
            had_hist=had_hist,
        )
        one = sn.new_ones(())
        spec_t = torch.stack([one if it[6] is None else it[6] for it in items])
        sc = (-(lrs * c["factors"] * spec_t * c_osc * damp) / sn).float()
        self.fast_path_steps += 1
        if keep is None:
            torch._foreach_addcmul_(p_list, g_list, list(sc.unbind(0)))
            return []
        return self._apply_partial_foreach(items, p_list, g_list, sc, keep)

    @staticmethod
    def _pack_fp32_norms(
        net_all: List[Tuple[int, Tuple[Any, ...]]],
        ref_dev: torch.device,
        s_vec: List[torch.Tensor],
    ) -> Tuple[List[int], int]:
        fp32_idx: List[int] = []
        fp32_p: List[torch.Tensor] = []
        fp32_g: List[torch.Tensor] = []
        for idx, (_, it) in enumerate(net_all):
            p, g = it[0], it[1]
            if (
                p.dtype == torch.float32
                and g.dtype == torch.float32
                and p.device == ref_dev
                and g.numel() > 0
            ):
                fp32_idx.append(idx)
                fp32_p.append(p)
                fp32_g.append(g)
        off = len(s_vec)
        if fp32_g:
            s_vec.extend(n.double() for n in torch._foreach_norm(fp32_g))
            s_vec.extend(n.double() for n in torch._foreach_norm(fp32_p))
        return fp32_idx, off

    @staticmethod
    def _resolve_item_norm(
        p: torch.Tensor,
        g: torch.Tensor,
        norm_pair: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[torch.Tensor, float, float, bool]]:
        if norm_pair is not None:
            gn, pn = norm_pair
            if (
                math.isfinite(gn)
                and 1e-17 < gn < 1e17
                and math.isfinite(pn)
                and pn < 1e17
            ):
                has_w = pn > 1e-12
                return g / gn, math.log2(gn), (math.log2(pn) if has_w else 0.0), has_w
        gu, l2g_t, okg = mantissa_exponent_normalize(g)
        if not bool(okg.item()):
            return None
        _, l2p_t, okp = mantissa_exponent_normalize(p.detach())
        lp_v = float(l2p_t.item())
        has_w = bool(okp.item()) and lp_v > _WEIGHT_LOG2_FLOOR
        return gu, float(l2g_t.item()), (lp_v if has_w else 0.0), has_w

    @staticmethod
    def _apply_updates(
        upd_p: List[torch.Tensor], upd_d: List[torch.Tensor], ref_dev: torch.device
    ) -> None:
        if not upd_p:
            return
        if all(p.device == ref_dev and p.dtype == upd_p[0].dtype for p in upd_p):
            torch._foreach_add_(upd_p, upd_d)
        else:
            for p_i, d_i in zip(upd_p, upd_d):
                p_i.add_(d_i)

    def _run_slow_and_ns(
        self,
        prior_it: List[Tuple[Any, ...]],
        slow_it: List[Tuple[Any, ...]],
        d2_it: List[Tuple[Any, ...]],
        *,
        c_osc: torch.Tensor,
        s_osc: torch.Tensor,
        g_eq: torch.Tensor,
        stiff_t: torch.Tensor,
        d_eq: float,
        sm_r: float,
        had_hist: bool,
        g_mult: float | torch.Tensor,
        d_cpl: float | torch.Tensor,
        ref_dev: torch.device,
    ) -> float:
        pre_pr = [
            (lr_p, self._prep_prior(p, g, grp, b1_p))
            for p, g, lr_p, b1_p, grp in prior_it
        ]
        gm_t = (
            g_mult.to(ref_dev, dtype=torch.float64).reshape(())
            if isinstance(g_mult, torch.Tensor)
            else c_osc.new_full((), float(g_mult))
        )
        dm_t = (
            d_cpl.to(ref_dev, dtype=torch.float64).reshape(())
            if isinstance(d_cpl, torch.Tensor)
            else c_osc.new_full((), float(d_cpl))
        )
        s_vec: List[torch.Tensor] = [
            c_osc.double(),
            s_osc.double(),
            g_eq.double(),
            stiff_t.double(),
            gm_t,
            dm_t,
        ]
        pr_off = len(s_vec)
        for _, pr in pre_pr:
            if pr is not None:
                s_vec.extend((pr[6], pr[7]))
        net_all = [(0, x) for x in slow_it] + [(1, x) for x in d2_it]
        fp32_idx, norm_off = self._pack_fp32_norms(net_all, ref_dev, s_vec)
        sv = torch.stack(s_vec).tolist()
        cosc_f, sosc_f, geq_f, stiff_f, gm_f, dm_f = [float(x) for x in sv[:6]]
        s_base_f = sm_r * 0.05 * (1.0 - stiff_f)
        s_peak_f = sm_r * 0.20 * (1.0 - stiff_f)
        s_osc_f = sosc_f * (1.0 - stiff_f)
        s_rel_pr = s_osc_f + (0.0 if self._mod_prior_scale < 1.0 else geq_f * s_base_f)

        pi_k = pr_off
        for lr_p, pr in pre_pr:
            if pr is None:
                continue
            self._finish_prior(
                pr,
                float(sv[pi_k]),
                float(sv[pi_k + 1]),
                base_lr=lr_p,
                s_rel_pr=s_rel_pr,
                s_peak_pr=1.25 * geq_f * s_peak_f,
                g_eq=geq_f,
                d_eq=d_eq,
                had_hist=had_hist,
            )
            pi_k += 2

        nf = len(fp32_idx)
        norm_map = {
            k: (float(sv[norm_off + j]), float(sv[norm_off + nf + j]))
            for j, k in enumerate(fp32_idx)
        }
        upd_p, upd_d, ns_b = self._prepare_net_updates(
            net_all,
            norm_map,
            cosc_f=cosc_f,
            s_osc_f=s_osc_f,
            s_base_f=s_base_f,
            s_peak_f=s_peak_f,
            geq_f=geq_f,
            d_eq=d_eq,
            had_hist=had_hist,
            gm_f=gm_f,
            dm_f=dm_f,
        )
        self._run_ns_buckets(ns_b, upd_p, upd_d)
        self._apply_updates(upd_p, upd_d, ref_dev)
        return stiff_f

    def _prepare_net_updates(
        self,
        net_all: List[Tuple[int, Tuple[Any, ...]]],
        norm_map: Dict[int, Tuple[float, float]],
        *,
        cosc_f: float,
        s_osc_f: float,
        s_base_f: float,
        s_peak_f: float,
        geq_f: float,
        d_eq: float,
        had_hist: bool,
        gm_f: float,
        dm_f: float,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Dict[Tuple[int, int], List[Tuple[Any, ...]]],
    ]:
        upd_p: List[torch.Tensor] = []
        upd_d: List[torch.Tensor] = []
        ns_b: Dict[Tuple[int, int], List[Tuple[Any, ...]]] = {}
        for idx, (kind, it) in enumerate(net_all):
            p, g = it[0], it[1]
            resolved = self._resolve_item_norm(p, g, norm_map.get(idx))
            if resolved is None:
                continue
            gu, l2_g, l2_p, has_w = resolved
            st = self.state[p]
            if kind == 0:
                _, _, lr_p, fac, hl2, b1, is_gr, spec_t = it
                fac_eff = fac * (float(spec_t.item()) if spec_t is not None else 1.0)
                rd = cd = 1
            else:
                _, _, lr_p, fac_eff, hl2, rd, cd, b1, is_gr = it
            gu = _apply_momentum(st, p, gu, b1)
            lt = l2_g + l2_p - float(hl2)
            rn = _update_l2_ref(st, lt, geq_f, d_eq, has_w)
            rat = 2.0 ** max(-20.0, min(20.0, rn - lt))
            active_smooth = had_hist and (has_w or bool(st.get("log2_init", False)))
            s_tikh = (
                geq_f * (s_base_f * rat + s_peak_f * max(0.0, rat - 1.0))
                + s_osc_f * rat
                if active_smooth
                else 0.0
            )
            rate = (lr_p * gm_f) if is_gr else (lr_p * dm_f)
            if kind == 0:
                scale = -(rate * fac_eff) * cosc_f / math.sqrt(1.0 + s_tikh * s_tikh)
                upd_p.append(p)
                upd_d.append((gu * scale).to(p.dtype))
            else:
                nt = rd > cd
                mc = gu.reshape(rd, cd).T if nt else gu.reshape(rd, cd)
                ns_b.setdefault((int(mc.shape[0]), int(mc.shape[1])), []).append(
                    (p, mc, nt, rate * fac_eff * cosc_f, s_tikh)
                )
        return upd_p, upd_d, ns_b

    def _run_ns_buckets(
        self,
        ns_b: Dict[Tuple[int, int], List[Tuple[Any, ...]]],
        upd_p: List[torch.Tensor],
        upd_d: List[torch.Tensor],
    ) -> None:
        for bkt in ns_b.values():
            self.ns_2d_activations += len(bkt)
            mats = torch.stack([b[1] for b in bkt], dim=0)
            rs_vals = [b[4] for b in bkt]
            if all(v <= 0.0 for v in rs_vals):
                rs_arg: float | torch.Tensor = 0.0
            elif all(v == rs_vals[0] for v in rs_vals):
                rs_arg = rs_vals[0]
            else:
                rs_arg = torch.tensor(rs_vals, dtype=torch.float32, device=mats.device)
            orths = fp32_tikhonov_ns_batched(
                mats, steps=5, rel_smooth=rs_arg, pre_normalized=True
            ).float()
            for idx, (p_i, _, nt, mult, _) in enumerate(bkt):
                mat_i = orths[idx].T if nt else orths[idx]
                upd_p.append(p_i)
                upd_d.append((mat_i * (-mult)).reshape_as(p_i).to(p_i.dtype))

    @staticmethod
    def _apply_schatten6(
        s6_b: Dict[Tuple[int, int], List[Tuple[Any, ...]]],
        d_cpl: float | torch.Tensor,
        ref_dev: torch.device,
    ) -> None:
        if not s6_b:
            return
        dm_t = (
            d_cpl.to(ref_dev, dtype=torch.float32).reshape(())
            if isinstance(d_cpl, torch.Tensor)
            else torch.full((), float(d_cpl), dtype=torch.float32, device=ref_dev)
        )
        for bkt in s6_b.values():
            ws = torch.stack(
                [
                    (p.float().reshape(rd, cd).T if nt else p.float().reshape(rd, cd))
                    / fac
                    for p, fac, nt, rd, cd, _ in bkt
                ],
                dim=0,
            )
            rates = torch.tensor(
                [r_s for *_, r_s in bkt], dtype=torch.float32, device=ws.device
            ) * dm_t.to(ws.device)
            ds6 = schatten6_decay_batched(ws, rates, lam=0.25, tau=2.0)
            for i, (p, fac, nt, _, _, _) in enumerate(bkt):
                upd = (ds6[i].T if nt else ds6[i]) * fac
                p.add_(upd.reshape_as(p).to(p.dtype))

    @staticmethod
    def _probe_rel_rms(
        y: Optional[torch.Tensor], y_pre: torch.Tensor, coord_sc: float
    ) -> float:
        if y is None or y.shape != y_pre.shape:
            return float("inf")
        bp = int(y_pre.shape[0])
        diff_rms = (y.double() - y_pre.double()).reshape(bp, -1).square().mean().sqrt()
        fin, rms = torch.stack((torch.isfinite(y).all().double(), diff_rms)).tolist()
        return (float(rms) / coord_sc) if fin > 0.5 else float("inf")

    @staticmethod
    def _flush_trust_probe(
        rec: Any, yp: torch.Tensor, y_pre: torch.Tensor, pre_gain: float | torch.Tensor
    ) -> Tuple[float, float, float]:
        bp, dev = int(y_pre.shape[0]), yp.device
        ek, et = (
            rec.take_pending_ema_tensors()
            if hasattr(rec, "take_pending_ema_tensors")
            else ([], [])
        )
        fin0 = torch.isfinite(yp).all().double()
        z_zero = yp.new_zeros((), dtype=torch.float64)
        diff0 = torch.where(torch.isfinite(yp), yp.double() - y_pre.double(), z_zero)
        rms0 = diff0.reshape(bp, -1).square().mean().sqrt()
        has_pg = isinstance(pre_gain, torch.Tensor)
        can_pack = bool(et) and all(t.device == dev for t in et)
        pack = [fin0, rms0]
        if has_pg:
            pack.append(pre_gain.to(device=dev, dtype=torch.float64).reshape(()))
        if can_pack:
            pack.extend(t.double() for t in et)
        pv = torch.stack(pack).tolist()
        pg_f = float(pv[2]) if has_pg else float(pre_gain)
        if et:
            vals = (
                [float(v) for v in pv[2 + int(has_pg) :]]
                if can_pack
                else [float(t.item()) for t in et]
            )
            rec.apply_flushed_ema_values(ek, vals)
        return float(pv[0]), float(pv[1]), pg_f

    def _backtrack_trust_alpha(
        self,
        g_mod: nn.Module,
        pa: Tuple[Any, ...],
        pk: Dict[str, Any],
        rec: Any,
        deltas: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        y_pre: torch.Tensor,
        coord_sc: float,
        d_obs: float,
        d_min: float,
        d_max: float,
    ) -> Tuple[float, float]:
        alpha = 1.0
        if d_min > 0.0 and math.isfinite(d_obs) and 1e-12 < d_obs < d_min:
            alpha = min(16.0, max(1.0, d_min / (d_obs + d_min / 16.0)))
            for p, o, d in deltas:
                p.copy_(o + d * alpha)
            d_obs = self._probe_rel_rms(
                eval_probe_forward(g_mod, pa, pk, rec), y_pre, coord_sc
            )
        for _ in range(4):
            if d_obs <= d_max:
                break
            self.trust_backtracks_total += 1
            if math.isfinite(d_obs) and d_obs > 1e-30:
                alpha = max(1e-6, min(0.5 * alpha, 0.9 * alpha * (d_max / d_obs)))
            else:
                alpha = max(1e-6, 0.1 * alpha)
            for p, o, d in deltas:
                p.copy_(o + d * alpha)
            d_obs = self._probe_rel_rms(
                eval_probe_forward(g_mod, pa, pk, rec), y_pre, coord_sc
            )
        if not math.isfinite(d_obs) or d_obs > d_max:
            alpha = max(1e-6, 0.5 * alpha)
            for p, o, d in deltas:
                p.copy_(o + d * alpha)
        return alpha, d_obs

    def _update_func_gain(
        self,
        rec: Any,
        *,
        alpha: float,
        d_obs: float,
        d_max: float,
        pg_f: float,
        s_therm: float,
        stiff_g: float,
    ) -> None:
        if alpha < 1.0:
            tgt = max(0.05, min(1.0, pg_f * alpha))
            self._ema_func_gain = 0.70 * self._ema_func_gain + 0.30 * tgt
        elif (
            s_therm >= 0.5 and math.isfinite(d_obs) and (alpha > 1.0 or d_obs >= 0.012)
        ):
            fr = (d_max / max(d_obs, 1e-12)) * pg_f
            if fr > 1.5 and min(1.0, fr) > self._ema_func_gain:
                tgt = max(0.05, min(1.0, fr))
                self._ema_func_gain = 0.995 * self._ema_func_gain + 0.005 * tgt
        rec.g_func_gain = float((1.0 - stiff_g) * self._ema_func_gain + stiff_g)

    def _apply_verified_trust(
        self,
        rec: Any,
        tracked: List[Tuple[torch.Tensor, torch.Tensor]],
        *,
        step_scale: float,
        s_therm: float,
        pre_gain: float | torch.Tensor,
        is_simp: bool,
        stiff_g: float,
    ) -> Tuple[float, bool]:
        cache, rec._g_probe_cache = rec._g_probe_cache, None
        if cache is None:
            return 1.0, False
        g_mod, pa, pk, y_pre = cache
        deltas = [(p, o, (p - o).detach()) for p, o in tracked]
        yp = eval_probe_forward(g_mod, pa, pk, rec)
        if yp is None or yp.shape != y_pre.shape:
            return 1.0, False
        fin0, rms0, pg_f = self._flush_trust_probe(rec, yp, y_pre, pre_gain)
        per_dim = math.sqrt(max(1, int(y_pre[0].numel())))
        coord_sc = max(1e-30, float(getattr(rec, "ema_data_scale", 1.0)) / per_dim)
        push_res = float(getattr(rec, "ema_push_res", 0.0))
        d_max = (0.045 + 0.16 * min(1.5, push_res)) * step_scale
        d_min = (
            0.012 * smoothstep01((push_res - 0.18) / 0.25) * step_scale
            if (not is_simp and push_res > 0.18 and s_therm > 0.10)
            else 0.0
        )
        d_obs0 = (rms0 / coord_sc) if fin0 > 0.5 else float("inf")
        alpha, d_obs = self._backtrack_trust_alpha(
            g_mod, pa, pk, rec, deltas, y_pre, coord_sc, d_obs0, d_min, d_max
        )
        self._update_func_gain(
            rec,
            alpha=alpha,
            d_obs=d_obs,
            d_max=d_max,
            pg_f=pg_f,
            s_therm=s_therm,
            stiff_g=stiff_g,
        )
        return alpha, True

    def _apply_direction_blend(
        self, all_p: List[torch.Tensor], origs: List[torch.Tensor]
    ) -> None:
        normals = self._protected
        if normals is None or not all_p:
            return
        disps = [p - o for p, o in zip(all_p, origs)]
        ad, a_n2, a01_f, d2_f, bmax = _blend_dot_products(disps, normals)
        stats, blend = self.constraint_geometry_stats, self.direction_blend_stats
        proj, after_max = False, bmax
        if bmax > 0.0:
            blend["conflict_steps"] += 1
            poss = True
            if len(normals) == 2 and a_n2[0] > 0.0 and a_n2[1] > 0.0:
                rho = max(-1.0, min(1.0, a01_f / math.sqrt(a_n2[0] * a_n2[1])))
                poss = math.sqrt(max(0.0, 0.5 * (1.0 + rho))) > 1e-12
            if poss:
                alphas, is_zero, proj = solve_nonascent_multipliers(
                    ad, a_n2, d2_f, a01_f
                )
                blend["blended_steps"] += 1
            else:
                alphas, is_zero, proj = (0.0,) * len(normals), True, True
                blend["pareto_stalls"] += 1
            if is_zero:
                torch._foreach_copy_(all_p, origs)
                after_max, app_n2 = 0.0, 0.0
            else:
                _apply_blend_step(all_p, origs, disps, normals, alphas)
                after_max, app_n2 = _blend_post_metrics(ad, a_n2, a01_f, d2_f, alphas)
            dn = math.sqrt(max(0.0, d2_f))
            ratio = math.sqrt(app_n2) / dn if dn > 0.0 else 0.0
            blend["retained_norm_ratio_sum"] += ratio
            blend["max_retained_norm_ratio"] = max(
                blend["max_retained_norm_ratio"], ratio
            )
        stats["steps"] += 1
        stats["projected_steps"] += int(proj)
        stats["max_derivative_before"] = max(stats["max_derivative_before"], bmax)
        stats["max_derivative_after"] = max(stats["max_derivative_after"], after_max)

    def _classify_net_param(
        self,
        p: torch.Tensor,
        g: torch.Tensor,
        lr_p: float,
        b1: float,
        is_gr: bool,
        gen_probe: bool,
        d2_it: List[Any],
        crit_2d: List[Any],
        fast_it: List[Any],
        slow_it: List[Any],
    ) -> None:
        fac, hl2, rd, cd, md = self._param_metrics(p)
        if p.ndim in (2, 4) and md >= 2:
            if is_gr:
                d2_it.append((p, g, lr_p, fac, hl2, rd, cd, b1, True))
            else:
                crit_2d.append((p, g, lr_p, fac, hl2, rd, cd, md, b1, False))
            return
        can_fast = (
            not gen_probe
            and b1 == 0.0
            and p.dtype == torch.float32
            and g.dtype == torch.float32
        )
        if can_fast:
            fast_it.append((p, g, lr_p, fac, hl2, is_gr, None))
        else:
            slow_it.append((p, g, lr_p, fac, hl2, b1, is_gr, None))

    def _partition_step_items(
        self,
        *,
        step_scale: float,
        s_cool: float,
        pyth: Dict[int, float],
        gen_probe: bool,
    ) -> Tuple[
        List[Any],
        List[Any],
        List[Any],
        List[Any],
        List[Any],
        List[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        fast_it, d2_it, slow_it, prior_it, crit_2d, tracked = [], [], [], [], [], []
        for grp in self.param_groups:
            is_pr = grp["algorithm"] == "axiom_prior"
            is_gr = grp["role"] not in ("critic", "prior", "table")
            b1 = float(grp["betas"][0])
            b1_p = max(b1, 0.75) if (is_pr and self._mod_prior_scale < 1.0) else b1
            blr = (
                float(grp["lr"])
                * step_scale
                * s_cool
                * (self._mod_prior_scale if is_pr else 1.0)
            )
            for p in grp["params"]:
                g = p.grad
                if g is None:
                    continue
                self.state[p]["step"] = int(self.state[p].get("step", 0)) + 1
                lr_p = blr * pyth.get(id(p), 1.0)
                if is_pr:
                    prior_it.append((p, g, lr_p, b1_p, grp))
                elif lr_p > 0.0:
                    if gen_probe and (
                        not self._gen_param_ids or id(p) in self._gen_param_ids
                    ):
                        tracked.append((p, p.detach().clone()))
                    self._classify_net_param(
                        p,
                        g,
                        lr_p,
                        b1,
                        is_gr,
                        gen_probe,
                        d2_it,
                        crit_2d,
                        fast_it,
                        slow_it,
                    )
        return fast_it, d2_it, slow_it, prior_it, crit_2d, tracked

    def _route_critic_2d(
        self,
        crit_2d: List[Tuple[Any, ...]],
        d2_it: List[Any],
        fast_it: List[Any],
        slow_it: List[Any],
        s_therm: float,
        ref_dev: torch.device,
    ) -> Dict[Tuple[int, int], List[Tuple[Any, ...]]]:
        s6_b: Dict[Tuple[int, int], List[Tuple[Any, ...]]] = {}
        if not crit_2d:
            return s6_b
        crit_m4 = [it for it in crit_2d if it[7] >= 4]
        if self._detect_critic_pathology(crit_m4, ref_dev):
            for p, g, lr_p, fac, hl2, rd, cd, _, b1, _ in crit_2d:
                d2_it.append((p, g, lr_p, fac, hl2, rd, cd, b1, False))
            return s6_b
        for (p, g, lr_p, fac, hl2, rd, cd, _, b1, _), gamma_t in zip(
            crit_2d, self._critic_spec_factors(crit_2d)
        ):
            if b1 == 0.0 and p.dtype == torch.float32 and g.dtype == torch.float32:
                fast_it.append((p, g, lr_p, fac, hl2, False, gamma_t))
            else:
                slow_it.append((p, g, lr_p, fac, hl2, b1, False, gamma_t))
            if s_therm > 0.0:
                nt = rd > cd
                s6_b.setdefault((cd, rd) if nt else (rd, cd), []).append(
                    (p, fac, nt, rd, cd, lr_p * s_therm)
                )
        return s6_b

    def _critic_coupling(
        self, is_crit: bool, rec: Any, ref_dev: torch.device
    ) -> float | torch.Tensor:
        if not is_crit or rec is None:
            return 1.0
        g_fg = float(getattr(rec, "g_func_gain", 1.0))
        if g_fg >= 0.75:
            return 1.0
        dg = max(0.12, g_fg)
        if hasattr(rec, "dev_ema"):
            em = rec.dev_ema("ema_margin", ref_dev)
            mi = rec.dev_ema("margin_initialized", ref_dev)
            gate = smoothstep01((em - 0.28) / 0.22) * smoothstep01((1.40 - em) / 0.30)
            mg = torch.where(mi > 0.5, gate, em.new_zeros(()))
            return (1.0 - mg) + mg * dg
        if getattr(rec, "margin_initialized", False):
            em_f = float(getattr(rec, "ema_margin", 0.0))
            mg_f = smoothstep01((em_f - 0.28) / 0.22) * smoothstep01(
                (1.40 - em_f) / 0.30
            )
            return (1.0 - mg_f) + mg_f * dg
        return 1.0

    def _collect_validated_pairs(
        self,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.device]]:
        net_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        ref_dev: Optional[torch.device] = None
        for grp in self.param_groups:
            is_pr = grp["algorithm"] == "axiom_prior"
            req = bool(grp.get("sampled_rows_required", True))
            for p in grp["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse or g.is_complex():
                    raise ValueError("AxiomOptimizer requires dense real gradients")
                if is_pr and req and p not in self._sampled_rows:
                    raise ValueError(
                        "row-normalized prior updates require actual sampled rows"
                    )
                ref_dev = p.device if ref_dev is None else ref_dev
                if not is_pr:
                    net_pairs.append((p, g))
        return net_pairs, ref_dev

    def _can_use_gen_probe(
        self, rec: Any, is_crit: bool, is_simp: bool, is_cf: bool
    ) -> bool:
        if is_crit or rec is None or is_simp or is_cf:
            return False
        if self._protected is not None and len(self._protected) >= 2:
            return False
        return getattr(rec, "_g_probe_cache", None) is not None and bool(
            getattr(rec, "_enable_secant_probe", True)
        )

    def _build_step_context(
        self, rec: Any, is_crit: bool, ref_dev: torch.device
    ) -> Dict[str, Any]:
        step_scale = float(getattr(rec, "step_scale", 1.0)) if rec is not None else 1.0
        s_therm = float(getattr(rec, "thermal_scale", 1.0)) if rec is not None else 1.0
        is_simp = (
            bool(getattr(rec, "_simplicial_step", False)) if rec is not None else False
        )
        is_cf = (
            bool(getattr(rec, "cond_initialized", False)) if rec is not None else False
        )
        ema_fg = float(self._ema_func_gain)
        pg_b = 1.0 if ema_fg >= 0.75 else ema_fg
        wb = self._mod_wakeup_boost
        pre_g = (
            (pg_b * wb.to(ref_dev)).clamp_max(1.0)
            if isinstance(wb, torch.Tensor)
            else min(1.0, pg_b * float(wb))
        )
        gs = self._mod_g_scale
        g_mult = (gs.to(ref_dev) if isinstance(gs, torch.Tensor) else gs) * pre_g
        return {
            "step_scale": step_scale,
            "s_therm": s_therm,
            "is_simp": is_simp,
            "is_cf": is_cf,
            "ema_fg": ema_fg,
            "pre_g": pre_g,
            "g_mult": g_mult,
            "d_cpl": self._critic_coupling(is_crit, rec, ref_dev),
            "gen_probe": self._can_use_gen_probe(rec, is_crit, is_simp, is_cf),
            "s_cool": 1.0
            if s_therm >= 0.9999
            else s_therm / math.hypot(s_therm, 0.0061),
            "d_eq": 0.0005 if s_therm >= 0.5 else 0.0,
        }

    def _finalize_step_post(
        self,
        rec: Any,
        is_crit: bool,
        ctx: Dict[str, Any],
        tracked: List[Tuple[torch.Tensor, torch.Tensor]],
        stiff_f: float,
        all_p: Optional[List[torch.Tensor]],
        origs: Optional[List[torch.Tensor]],
    ) -> None:
        if ctx["gen_probe"] and tracked and rec is not None:
            alpha_g, p_ran = self._apply_verified_trust(
                rec,
                tracked,
                step_scale=ctx["step_scale"],
                s_therm=ctx["s_therm"],
                pre_gain=ctx["pre_g"],
                is_simp=ctx["is_simp"],
                stiff_g=stiff_f,
            )
        else:
            alpha_g, p_ran = 1.0, False
            if rec is not None and not is_crit:
                rec._g_probe_cache = None
        if self.direction_blend and all_p is not None and origs is not None:
            self._apply_direction_blend(all_p, origs)
        self.last_step_gain, self.last_probe_consumed = float(alpha_g), bool(p_ran)
        self.probe_consumptions += int(p_ran)

    def _execute_step_body(
        self,
        net_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        ref_dev: torch.device,
        all_p: Optional[List[torch.Tensor]],
        origs: Optional[List[torch.Tensor]],
    ) -> None:
        rec, is_crit = self._active_record(), self.critic is not None
        ctx = self._build_step_context(rec, is_crit, ref_dev)
        pyth = self._two_obj_pyth() if (ctx["is_cf"] and self.direction_blend) else {}
        sm_r = float(self.smoothing) / 0.05

        fast_it, d2_it, slow_it, prior_it, crit_2d, tracked = (
            self._partition_step_items(
                step_scale=ctx["step_scale"],
                s_cool=ctx["s_cool"],
                pyth=pyth,
                gen_probe=ctx["gen_probe"],
            )
        )
        s6_b = self._route_critic_2d(
            crit_2d, d2_it, fast_it, slow_it, ctx["s_therm"], ref_dev
        )
        c_osc, s_osc_raw, g_eq, stiff_t, had_hist = self._update_step_cosine(
            net_pairs, ref_dev, ctx["ema_fg"]
        )
        if fast_it:
            spilled = self._run_foreach_fast(
                fast_it,
                c_osc=c_osc,
                s_osc=s_osc_raw * (1.0 - stiff_t),
                s_base=(sm_r * 0.05) * (1.0 - stiff_t),
                s_peak=(sm_r * 0.20) * (1.0 - stiff_t),
                g_eq=g_eq,
                d_eq=ctx["d_eq"],
                had_hist=had_hist,
                g_mult=ctx["g_mult"],
                d_mult=ctx["d_cpl"],
            )
            if spilled:
                self._flush_fast_cache()
                self._fast_cache = None
                for p, g, lr_p, fac, hl2, is_gr, spec_t in spilled:
                    slow_it.append((p, g, lr_p, fac, hl2, 0.0, is_gr, spec_t))
        stiff_f = 0.0
        if prior_it or slow_it or d2_it:
            stiff_f = self._run_slow_and_ns(
                prior_it,
                slow_it,
                d2_it,
                c_osc=c_osc,
                s_osc=s_osc_raw,
                g_eq=g_eq,
                stiff_t=stiff_t,
                d_eq=ctx["d_eq"],
                sm_r=sm_r,
                had_hist=had_hist,
                g_mult=ctx["g_mult"],
                d_cpl=ctx["d_cpl"],
                ref_dev=ref_dev,
            )
        self._apply_schatten6(s6_b, ctx["d_cpl"], ref_dev)
        self._finalize_step_post(rec, is_crit, ctx, tracked, stiff_f, all_p, origs)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.direction_blend and (closure is not None or self._protected is None):
            self.abort_pending()
            raise ValueError(
                "direction_blend step requires explicit protected-loss backward"
            )
        net_pairs, ref_dev = self._collect_validated_pairs()
        all_p = (
            self._parameters()
            if (self.direction_blend and self._protected is not None)
            else None
        )
        origs = [p.detach().clone() for p in all_p] if all_p is not None else None
        try:
            if ref_dev is None:
                self.last_step_gain, self.last_probe_consumed = 1.0, False
                rec = self._active_record()
                if rec is not None and self.critic is None:
                    rec._g_probe_cache = None
                return loss
            self._execute_step_body(net_pairs, ref_dev, all_p, origs)
        finally:
            self.abort_pending()
        if self.record is not None:
            self.record.record_step(self)
        return loss

    def __getstate__(self) -> Dict[str, Any]:
        self._flush_scalars()
        self._flush_fast_cache()
        return {**self.__dict__, "_fast_cache": None, "_metrics_cache": {}}

    def state_dict(self) -> Dict[str, Any]:
        self._flush_scalars()
        self._flush_fast_cache()
        res = Optimizer.state_dict(self)
        sampled_rows: List[Optional[torch.Tensor]] = []
        for g in self.param_groups:
            rows = (
                self.sampled_rows_for(g["params"][0])
                if g["algorithm"] == "axiom_prior"
                else None
            )
            sampled_rows.append(None if rows is None else rows.cpu())
        sketch = (
            None
            if self._prev_step_sketch is None
            else [float(x) for x in self._prev_step_sketch.detach().cpu().tolist()]
        )
        meta: Dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "family": self.family,
            "last_step_gain": float(self.last_step_gain),
            "ema_func_gain": float(self._ema_func_gain),
            "ema_step_cos": float(self._ema_step_cos_val),
            "log2_damp": float(self._log2_damp_val),
            "crit_skew_ema": float(self._crit_skew_ema),
            "direction_blend": bool(self.direction_blend),
            "prev_step_sketch": sketch,
            "sampled_rows": sampled_rows,
        }
        if self.direction_blend:
            pending = (
                None
                if self._protected is None
                else [
                    [None if t is None else t.detach().cpu().clone() for t in grp]
                    for grp in self._protected
                ]
            )
            meta["constraint_geometry"] = {
                "stats": deepcopy(self.constraint_geometry_stats),
                "blend_stats": deepcopy(self.direction_blend_stats),
                "pending": pending,
            }
        res["axiom"] = meta
        if self.record is not None:
            res["regularizer"] = {
                "optimizer_family": self.family,
                "record": self.record.state_dict(),
                "ema": None,
                "guard": None,
            }
        return res

    def _validate_pending_list(self, pending: Any) -> None:
        params = self._parameters()
        if not isinstance(pending, list) or not 1 <= len(pending) <= 2:
            raise ValueError("invalid AxiomOptimizer pending protected normals")
        for grp in pending:
            if not isinstance(grp, list) or len(grp) != len(params):
                raise ValueError("invalid AxiomOptimizer pending normal list length")
            for t, p in zip(grp, params):
                if t is None:
                    continue
                ok = (
                    isinstance(t, torch.Tensor)
                    and tuple(t.shape) == tuple(p.shape)
                    and bool(torch.isfinite(t).all())
                )
                if not ok:
                    raise ValueError("invalid AxiomOptimizer pending normal tensor")

    def _validate_pending_cg(self, cg: Any) -> None:
        if not isinstance(cg, dict) or set(cg) != {"stats", "blend_stats", "pending"}:
            raise ValueError("invalid AxiomOptimizer direction_blend checkpoint")
        bad_stats = not isinstance(cg["stats"], dict) or set(cg["stats"]) != set(
            self.constraint_geometry_stats
        )
        bad_blend = not isinstance(cg["blend_stats"], dict) or set(
            cg["blend_stats"]
        ) != set(self.direction_blend_stats)
        if bad_stats or bad_blend:
            raise ValueError("invalid AxiomOptimizer direction_blend stats")
        if cg["pending"] is not None:
            self._validate_pending_list(cg["pending"])

    def _validate_meta_header(self, meta: Any) -> None:
        exp_m = {
            "schema",
            "family",
            "last_step_gain",
            "ema_func_gain",
            "ema_step_cos",
            "log2_damp",
            "crit_skew_ema",
            "prev_step_sketch",
            "direction_blend",
            "sampled_rows",
        }
        if self.direction_blend:
            exp_m.add("constraint_geometry")
        if not isinstance(meta, dict) or set(meta) != exp_m:
            raise ValueError("incompatible AxiomOptimizer metadata keys")
        header_ok = (
            type(meta["schema"]) is int
            and meta["schema"] == SCHEMA_VERSION
            and meta["family"] == self.family
            and type(meta["direction_blend"]) is bool
            and meta["direction_blend"] == self.direction_blend
        )
        if not header_ok:
            raise ValueError("incompatible AxiomOptimizer schema or header")
        sr = meta["sampled_rows"]
        if not isinstance(sr, list) or len(sr) != len(self.param_groups):
            raise ValueError("incompatible AxiomOptimizer sampled_rows")

    def _validate_axiom_meta(self, meta: Any) -> None:
        self._validate_meta_header(meta)
        bounds = (
            ("last_step_gain", 0.0, None, True),
            ("ema_func_gain", 0.0, 1.0, True),
            ("ema_step_cos", -1.0, 1.0, False),
            ("log2_damp", _LOG2_DAMP_FLOOR, 0.0, False),
            ("crit_skew_ema", 0.0, None, False),
        )
        if any(not _is_fin_num(meta[k], lo, hi, slo) for k, lo, hi, slo in bounds):
            raise ValueError("invalid AxiomOptimizer scalar metadata")
        sk = meta["prev_step_sketch"]
        if sk is not None:
            if (
                not isinstance(sk, list)
                or len(sk) != _SKETCH_DIM
                or any(not _is_fin_num(x) for x in sk)
            ):
                raise ValueError("invalid AxiomOptimizer prev_step_sketch")
        if self.direction_blend:
            self._validate_pending_cg(meta["constraint_geometry"])

    def _validate_param_groups(self, saved_groups: Any) -> Dict[Any, Tuple[int, ...]]:
        if not isinstance(saved_groups, list) or len(saved_groups) != len(
            self.param_groups
        ):
            raise ValueError("AxiomOptimizer param_groups mismatch")
        p_shapes: Dict[Any, Tuple[int, ...]] = {}
        for cur, sav in zip(self.param_groups, saved_groups):
            valid = (
                isinstance(sav, dict)
                and len(sav.get("params", ())) == len(cur["params"])
                and sav.get("role") == cur["role"]
                and sav.get("algorithm") == cur["algorithm"]
            )
            if not valid:
                raise ValueError("AxiomOptimizer group mismatch")
            for p, sid in zip(cur["params"], sav["params"]):
                p_shapes[sid] = tuple(p.shape)
        return p_shapes

    @staticmethod
    def _validate_fp32_state_tensors(
        sv: Dict[str, Any], expected_shape: Optional[Tuple[int, ...]]
    ) -> None:
        for tk in _FP32_STATE_KEYS:
            if tk not in sv:
                continue
            t = sv[tk]
            if not isinstance(t, torch.Tensor) or not bool(torch.isfinite(t).all()):
                raise ValueError(f"invalid AxiomOptimizer tensor state {tk!r}")
            if tk in ("m_unit", "m_prior") and expected_shape is not None:
                if tuple(t.shape) != expected_shape:
                    raise ValueError(f"shape mismatch for {tk!r}")

    @classmethod
    def _validate_single_param_state(
        cls, sv: Any, expected_shape: Optional[Tuple[int, ...]]
    ) -> None:
        if not isinstance(sv, dict) or "m_u" in sv:
            raise ValueError("invalid AxiomOptimizer parameter state")
        step_val = sv.get("step", 0)
        if type(step_val) is not int or step_val < 0:
            raise ValueError("invalid AxiomOptimizer parameter step")
        if "log2_ref" in sv and not _is_fin_num(sv["log2_ref"]):
            raise ValueError("invalid AxiomOptimizer log2_ref")
        if "log2_init" in sv and type(sv["log2_init"]) is not bool:
            raise ValueError("invalid AxiomOptimizer log2_init")
        cls._validate_fp32_state_tensors(sv, expected_shape)

    @classmethod
    def _validate_param_states(
        cls, saved_state: Any, p_shapes: Dict[Any, Tuple[int, ...]]
    ) -> None:
        if not isinstance(saved_state, dict):
            raise ValueError("AxiomOptimizer state must be a dict")
        for sid, sv in saved_state.items():
            cls._validate_single_param_state(sv, p_shapes.get(sid))

    def validate_state_dict(self, saved: Any) -> None:
        """Fail-closed schema-2 validation rejecting schema-1, missing keys, or bad state."""
        exp_top = {"state", "param_groups", "axiom"}
        if self.record is not None:
            exp_top.add("regularizer")
        if not isinstance(saved, dict) or set(saved) != exp_top:
            raise ValueError("invalid AxiomOptimizer checkpoint schema")
        self._validate_axiom_meta(saved["axiom"])
        p_shapes = self._validate_param_groups(saved["param_groups"])
        self._validate_param_states(saved["state"], p_shapes)
        if self.record is not None:
            reg = saved["regularizer"]
            valid_reg = (
                isinstance(reg, dict)
                and set(reg) == {"optimizer_family", "record", "ema", "guard"}
                and reg["optimizer_family"] == self.family
                and reg["ema"] is None
                and reg["guard"] is None
            )
            if not valid_reg:
                raise ValueError("invalid AxiomOptimizer regularizer checkpoint")
            copy(self.record).load_state_dict(reg["record"])

    def _restore_fp32_state_tensors(self, state_dict: Dict[str, Any]) -> None:
        for cg, sg in zip(self.param_groups, state_dict["param_groups"]):
            for p, sid in zip(cg["params"], sg["params"]):
                sv = state_dict["state"].get(sid)
                if not isinstance(sv, dict):
                    continue
                for tk in _FP32_STATE_KEYS:
                    t = sv.get(tk)
                    if isinstance(t, torch.Tensor):
                        self.state[p][tk] = (
                            t.detach().to(device=p.device, dtype=torch.float32).clone()
                        )

    def _restore_direction_blend(self, meta: Dict[str, Any]) -> None:
        if not self.direction_blend:
            self._protected = None
            return
        cg = meta["constraint_geometry"]
        self.constraint_geometry_stats = deepcopy(cg["stats"])
        self.direction_blend_stats = deepcopy(cg["blend_stats"])
        if cg["pending"] is None:
            self._protected = None
            return
        params = self._parameters()
        self._protected = [
            [
                None
                if t is None
                else t.detach().to(device=p.device, dtype=t.dtype).clone()
                for t, p in zip(grp, params)
            ]
            for grp in cg["pending"]
        ]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        meta = state_dict["axiom"]
        base_sd = {
            k: v for k, v in state_dict.items() if k not in ("axiom", "regularizer")
        }
        Optimizer.load_state_dict(self, base_sd)
        self._restore_fp32_state_tensors(state_dict)

        self.clear_sampled_rows()
        self._fast_cache = None
        self._metrics_cache.clear()
        self._ema_step_cos_t = self._log2_damp_t = None
        self.last_step_gain = float(meta["last_step_gain"])
        self._ema_func_gain = float(meta["ema_func_gain"])
        self._ema_step_cos_val = float(meta["ema_step_cos"])
        self._log2_damp_val = float(meta["log2_damp"])
        self._crit_skew_ema = float(meta["crit_skew_ema"])
        params = self._parameters()
        ref_dev = params[0].device if params else torch.device("cpu")
        sk = meta["prev_step_sketch"]
        self._prev_step_sketch = (
            None
            if sk is None
            else torch.tensor(sk, dtype=torch.float32, device=ref_dev)
        )
        self.ns_2d_activations = self.fast_path_steps = 0
        self.trust_backtracks_total = self.probe_consumptions = 0
        self.last_probe_consumed = False
        for grp, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None and grp["algorithm"] == "axiom_prior":
                self.set_sampled_rows(
                    grp["params"][0], rows.to(grp["params"][0].device)
                )
        self._restore_direction_blend(meta)
        if self.record is not None:
            self.record.load_state_dict(state_dict["regularizer"]["record"])
            self.record.restore_device_state(ref_dev)


__all__ = [
    "AxiomOptimizer",
    "collect_conv_layouts",
    "fp32_tikhonov_ns_2d",
    "fp32_tikhonov_ns_batched",
    "mantissa_exponent_normalize",
    "zero_pinv_project_nonascent",
]
