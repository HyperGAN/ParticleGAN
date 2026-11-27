"""Axiom L1 Optimizer (Law 4)."""

from __future__ import annotations

from copy import copy
from math import hypot, isfinite, log2, sqrt

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
    stream_dot,
    tensor_dot,
    zero_pinv_project_nonascent,
)
from .state import (
    SCHEMA_VERSION,
    clear_active_prior_rows,
    eval_probe_forward,
    get_tensor_record,
    set_active_prior_rows,
)

_F32, _F64, _BOOL, _LONG = torch.float32, torch.float64, torch.bool, torch.long
_VALID_ROLES = ("generator", "encoder", "router", "noise", "critic", "prior", "table")
_OPT_DEFAULTS = (
    ("weight_decay", 0.0),
    ("amsgrad", False),
    ("maximize", False),
    ("differentiable", False),
    ("capturable", False),
    ("foreach", None),
    ("fused", None),
)
_FP32_STATE_KEYS = ("v_pow", "m_unit", "m_scale", "m_prior")
_SCALAR_SPECS = (
    ("last_step_gain", 0.0, None, True),
    ("ema_func_gain", 0.0, 1.0, True),
    ("ema_step_cos", -1.0, 1.0, False),
    ("log2_damp", -10.0, 0.0, False),
    ("crit_skew_ema", 0.0, None, False),
)
_META_KEYS = ("schema", "family", "direction_blend", "prev_step_sketch", "sampled_rows") + tuple(
    spec[0] for spec in _SCALAR_SPECS
)
_SKETCH_DIM = 67
_WEIGHT_LOG2_FLOOR = log2(1e-12)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("invalid axiom optimizer argument or state")


def _require_exact_keys(mapping, expected: set[str]) -> None:
    _require(isinstance(mapping, dict) and set(mapping) == expected)


def _is_finite_scalar(val, low=None, high=None, strict_low: bool = False) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not isfinite(val):
        return False
    low_ok = low is None or (val > low if strict_low else val >= low)
    return low_ok and (high is None or val <= high)


def _is_finite_tensor(tensor) -> bool:
    return isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all().item())


def _as_scalar_tensor(val, device, dtype=_F64) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.to(device, dtype).reshape(())
    return torch.full((), float(val), device=device, dtype=dtype)


def _is_scalar_loss(val) -> bool:
    return isinstance(val, torch.Tensor) and val.ndim == 0 and bool(val.requires_grad)


def _all_normals_finite(normals) -> bool:
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for group in normals:
        for tensor in group:
            if tensor is not None and tensor.numel() > 0:
                by_device.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    return all(bool(torch.stack(flags).all().item()) for flags in by_device.values())


def _mask_prior_rows(src: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    hit = src.new_zeros(len(src), dtype=_F32)
    hit.index_add_(0, rows, rows.new_ones(len(rows), dtype=_F32))
    return torch.where(hit.gt(0.0).unsqueeze(1), src, 0.0)


def _update_log2_ref(state: dict, cur_log2: float, g_eq: float, decay: float, has_weight: bool) -> float:
    if not state.get("log2_init", False):
        if has_weight:
            state["log2_init"], state["log2_ref"] = True, cur_log2
        return cur_log2
    prev = state.get("log2_ref", 0.0)
    if cur_log2 > prev:
        nxt = prev + 0.25 * min(2.0, cur_log2 - prev)
    else:
        nxt = prev - (0.25 * (1.0 - g_eq) + decay * g_eq) * min(2.0, prev - cur_log2)
    state["log2_ref"] = nxt
    return nxt


def _apply_momentum(state: dict, param: torch.Tensor, g_unit: torch.Tensor, beta1: float) -> torch.Tensor:
    if beta1 <= 0.0:
        return g_unit
    m_unit = state.setdefault("m_unit", param.new_zeros(param.shape, dtype=_F32))
    m_scale = state.setdefault("m_scale", param.new_zeros((), dtype=_F32))
    blended, _, _ = mantissa_exponent_normalize(beta1 * m_unit + (1.0 - beta1) * g_unit)
    m_unit.copy_(blended)
    m_scale.fill_(1.0)
    return blended


def _reachable_param_ids(loss: torch.Tensor, param_ids: set[int]) -> set[int]:
    touched: set[int] = set()
    visited: set[int] = set()
    stack = [getattr(loss, "grad_fn", None)]
    while stack:
        node = stack.pop()
        if node is None or id(node) in visited:
            continue
        visited.add(id(node))
        var = getattr(node, "variable", None)
        if isinstance(var, torch.Tensor) and id(var) in param_ids:
            touched.add(id(var))
        stack.extend(nxt for nxt, _ in getattr(node, "next_functions", ()))
    return touched


def _blend_dots_one(params, origs, n0):
    zero = params[0].new_zeros((), dtype=_F64)
    n0_sq = n0_d = d_sq = zero
    for param, orig, a0 in zip(params, origs, n0):
        if param.numel() == 0:
            continue
        disp = param - orig
        d_sq = d_sq + tensor_dot(disp, disp)
        if a0 is not None:
            n0_sq = n0_sq + tensor_dot(a0, a0)
            n0_d = n0_d + tensor_dot(a0, disp)
    v_n0_sq, v_n0_d, v_d_sq = torch.stack((n0_sq, n0_d, d_sq)).tolist()
    return [v_n0_d], [v_n0_sq], 0.0, v_d_sq


def _blend_dots_two(params, origs, n0, n1):
    zero = params[0].new_zeros((), dtype=_F64)
    n0_sq = n1_sq = n0_d = n1_d = n01 = d_sq = zero
    for param, orig, a0, a1 in zip(params, origs, n0, n1):
        if param.numel() == 0:
            continue
        disp = param - orig
        d_sq = d_sq + tensor_dot(disp, disp)
        if a0 is not None:
            n0_sq = n0_sq + tensor_dot(a0, a0)
            n0_d = n0_d + tensor_dot(a0, disp)
        if a1 is not None:
            n1_sq = n1_sq + tensor_dot(a1, a1)
            n1_d = n1_d + tensor_dot(a1, disp)
        if a0 is not None and a1 is not None:
            n01 = n01 + tensor_dot(a0, a1)
    v = torch.stack((n0_sq, n1_sq, n0_d, n1_d, n01, d_sq)).tolist()
    return [v[2], v[3]], [v[0], v[1]], v[4], v[5]


def _blend_post_metrics(ad, an2, a01, d2, alphas):
    a0, a1 = (alphas[0], alphas[1]) if len(alphas) > 1 else (alphas[0], 0.0)
    n0, n1, d0, d1 = (an2[0], an2[1], ad[0], ad[1]) if len(ad) > 1 else (an2[0], 0.0, ad[0], 0.0)
    after = max(d0 - a0 * n0 - a1 * a01, d1 - a0 * a01 - a1 * n1) if len(ad) > 1 else (d0 - a0 * n0)
    norm_sq = max(0.0, d2 - 2.0 * (a0 * d0 + a1 * d1) + a0 * a0 * n0 + a1 * a1 * n1 + 2.0 * a0 * a1 * a01)
    return after, norm_sq


def _probe_rms_and_finite(y_cur: torch.Tensor, y_pre: torch.Tensor):
    finite = torch.isfinite(y_cur).all().double()
    diff = torch.where(torch.isfinite(y_cur), y_cur.double() - y_pre.double(), 0.0)
    return finite, diff.reshape(y_pre.shape[0], -1).square().mean().sqrt()


def _role_pythagorean_boost(g0s, g1s):
    ref = next((tensor for tensor in (*g0s, *g1s) if tensor is not None), None)
    if ref is None:
        return None
    zero = ref.new_zeros((), dtype=_F64)
    a2, b2, dot = torch.stack(
        (stream_dot(g0s, g0s, zero), stream_dot(g1s, g1s, zero), stream_dot(g0s, g1s, zero))
    ).tolist()
    if a2 > 1e-24 and b2 > 1e-24 and isfinite(a2 + b2 + dot):
        return min(1.5, max(1.0, (sqrt(a2) + sqrt(b2)) / sqrt(max(a2 + b2 + 2.0 * dot, 1e-36))))
    return None


def _copy_pending(pending, params=None):
    if pending is None:
        return None
    return [
        [
            None if tensor is None else tensor.detach().to(param.device if params else "cpu", tensor.dtype).clone()
            for tensor, param in zip(group, params or group)
        ]
        for group in pending
    ]


def _resolve_norms(param: torch.Tensor, grad: torch.Tensor, pair, need_unit: bool = True):
    if pair is not None and 1e-17 < pair[0] < 1e17 and 0.0 <= pair[1] < 1e17:
        g_norm, p_norm = pair
        has_weight = p_norm > 1e-12
        return (grad / g_norm if need_unit else g_norm), log2(g_norm), (log2(p_norm) if has_weight else 0.0), has_weight
    g_unit, log2_g, ok_g = mantissa_exponent_normalize(grad)
    if not bool(ok_g.item()):
        return None
    _, log2_p, ok_p = mantissa_exponent_normalize(param.detach())
    lp = float(log2_p.item())
    has_weight = bool(ok_p.item()) and lp > _WEIGHT_LOG2_FLOOR
    return g_unit, float(log2_g.item()), (lp if has_weight else 0.0), has_weight


def _copy_params(dsts, srcs) -> None:
    if not dsts:
        return
    dev, dt = dsts[0].device, dsts[0].dtype
    if all(p.device == dev and p.dtype == dt for p in dsts):
        torch._foreach_copy_(dsts, srcs)
    else:
        for dst, src in zip(dsts, srcs):
            dst.copy_(src)


def _apply_param_deltas(params, deltas, dev) -> None:
    if not params:
        return
    if all(p.device == dev and p.dtype == params[0].dtype for p in params):
        torch._foreach_add_(params, deltas)
    else:
        for param, delta in zip(params, deltas):
            param.add_(delta)


def collect_conv_layouts(modules) -> dict[int, tuple[bool, int]]:
    """Map Conv2d/ConvTranspose2d weight ids to ``(is_transposed, groups)``."""
    return {
        id(mod.weight): (isinstance(mod, nn.ConvTranspose2d), int(mod.groups))
        for root in modules
        if isinstance(root, nn.Module)
        for mod in root.modules()
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d))
    }


class AxiomOptimizer(Optimizer):
    """Law-4 scale-invariant, clock-free Axiom optimizer with zero-pinv direction blend."""

    family = recipe_optimizer_family = "axiom"
    _SUPPORTS_ZEROSYNC_V4_TENSORS: bool = True
    ema_critic = anchor = guard = latent_damping = latent_history = direct_response = direct_history = None

    def __init__(
        self,
        params,
        *,
        lr=0.014,
        betas=(0.0, 0.999),
        eps=1e-8,
        smoothing=0.05,
        direction_blend=False,
        critic=None,
        record=None,
        conv_layouts=None,
        **options,
    ):
        self._check_init(lr, betas, eps, smoothing, direction_blend, options)
        self.smoothing, self.direction_blend, self.critic = float(smoothing), direction_blend, critic
        self.record = self._axiom_record = record
        self._prev_step_sketch = self._protected = None
        self._conv_layouts = dict(conv_layouts) if conv_layouts else {}
        self._sampled_rows, self._row_scales, self._gen_probe_ids, self._metric_cache = {}, {}, set(), {}
        self._mod_g_scale = self._mod_wakeup_boost = self._mod_prior_scale = 1.0
        self.last_step_gain = self.ema_func_gain = 1.0
        self.ema_step_cos = self.log2_damp = self.crit_skew_ema = 0.0
        self.constraint_geometry_stats = dict(
            steps=0, projected_steps=0, max_derivative_before=0.0, max_derivative_after=0.0
        )
        self.direction_blend_stats = dict(
            conflict_steps=0, blended_steps=0, pareto_stalls=0, retained_norm_ratio_sum=0.0, max_retained_norm_ratio=0.0
        )
        self._reset_counts()
        default_role = "critic" if critic is not None else "generator"
        defaults = {
            **dict(_OPT_DEFAULTS),
            "lr": float(lr),
            "betas": tuple(float(b) for b in betas),
            "eps": float(eps),
            "role": default_role,
            **options,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["role"] = role = group.get("role", group.get("forge_role", default_role))
            _require(role in _VALID_ROLES)
            is_prior = role in ("prior", "table")
            if is_prior:
                _require(len(group["params"]) == 1 and group["params"][0].ndim == 2)
                group.setdefault("sampled_rows_required", role == "prior")
            group["algorithm"] = "axiom_prior" if is_prior else "axiom"
        if isinstance(critic, nn.Module):
            self.bind_module_layouts((critic,))

    def _reset_counts(self) -> None:
        self.ns_2d_activations = self.fast_path_steps = self.trust_backtracks_total = self.probe_consumptions = 0
        self.last_probe_consumed = False

    @staticmethod
    def _check_init(lr, betas, eps, smoothing, direction_blend, options) -> None:
        ok = (
            _is_finite_scalar(lr, 0.0)
            and _is_finite_scalar(eps, 0.0, strict_low=True)
            and _is_finite_scalar(smoothing, 0.0)
            and type(direction_blend) is bool
            and len(betas) == 2
        )
        _require(ok)
        for beta in betas:
            _require(_is_finite_scalar(beta, 0.0) and beta < 1.0)
        allowed = {key for key, _ in _OPT_DEFAULTS}
        for key, val in options.items():
            if key not in allowed:
                raise TypeError(f"unexpected optimizer option {key!r}")
            _require(val in (None, False, 0, 0.0))

    def bind_module_layouts(self, modules) -> None:
        """Register Conv2d/ConvTranspose2d layouts from ``modules`` for exact polar scaling."""
        self._conv_layouts.update(collect_conv_layouts(modules))
        self._metric_cache.clear()

    def _active_record(self):
        rec = self.record if self.record is not None else self._axiom_record
        if rec is not None:
            return rec
        return next((found for p in self._parameters() if (found := get_tensor_record(p)) is not None), None)

    def _parameters(self) -> list[torch.Tensor]:
        return [param for group in self.param_groups for param in group["params"]]

    def set_generator_probe_parameters(self, parameters) -> None:
        """Restrict pushforward trust-band tracking to the given parameter subset."""
        self._gen_probe_ids = {id(param) for param in parameters}

    def bind_origin_sample(self, sample) -> None:
        """Compatibility no-op retained for trainer protocol parity."""

    def _param_metrics(self, param: torch.Tensor):
        cached = self._metric_cache.get(id(param))
        if cached is not None:
            return cached
        if param.ndim < 2:
            return self._metric_cache.setdefault(id(param), (0.5 if param.numel() == 1 else 1.0, 0.0, 1, 1, 1))
        if param.ndim == 4:
            c0, c1, kh, kw = param.shape
            is_trans, groups = self._conv_layouts.get(id(param), (False, 1))
            out_ch, in_ch = (c1 * groups, c0) if is_trans else (c0, c1 * groups)
            if not (kh == kw == groups == 1 and not is_trans):
                area = max(1, (kh * kw) // 4 if is_trans else kh * kw)
                fan = max(1, min(out_ch // max(1, groups), in_ch // max(1, groups)))
                factor = sqrt(out_ch / max(1, in_ch) * groups / area) * (fan**0.25)
                half_log2 = 0.5 * log2(max(1, min(out_ch, in_ch) * kh * kw))
                return self._metric_cache.setdefault(id(param), (factor, half_log2, 1, 1, 1))
            rows, cols = out_ch, in_ch
        else:
            rows, cols = param.shape[0], param.numel() // max(1, param.shape[0])
        min_dim = min(rows, cols)
        entry = (sqrt(max(1.0, rows / max(1, cols))), 0.5 * log2(max(1, min_dim)), rows, cols, min_dim)
        return self._metric_cache.setdefault(id(param), entry)

    def set_sampled_rows(self, param: torch.Tensor, rows) -> None:
        """Record sampled row indices for a particle-prior table parameter."""
        row_t = torch.as_tensor(rows).reshape(-1).to(param.device, _LONG)
        if row_t.device.type == "cpu" and row_t.numel():
            _require(not ((row_t < 0).any().item() or (row_t >= len(param)).any().item()))
        prev = self._sampled_rows.get(param)
        self._sampled_rows[param] = row_t if prev is None else torch.cat((prev, row_t))
        set_active_prior_rows(param, self._sampled_rows[param])
        if (rec := self._active_record()) is not None:
            rec._active_prior_table, rec._active_batch_rows = param, row_t.detach()

    def sampled_rows_for(self, param: torch.Tensor):
        """Return sorted unique sampled row indices for ``param``, or ``None``."""
        if param not in self._sampled_rows:
            return None
        return torch.unique(self._sampled_rows[param], sorted=True).detach().clone()

    def clear_sampled_rows(self) -> None:
        """Clear pending sampled prior rows and per-row scales."""
        for param in self._sampled_rows:
            clear_active_prior_rows(param)
        self._sampled_rows.clear()
        self._row_scales.clear()
        if (rec := self._active_record()) is not None:
            rec._active_prior_table = rec._active_batch_rows = None

    def set_row_scales(self, param: torch.Tensor, scales) -> None:
        """Bind optional per-row step scales for sampled prior updates."""
        if scales is None:
            self._row_scales.pop(param, None)
        else:
            self._row_scales[param] = scales.detach().to(param.device, _F32)

    def bind_axiom_modulation(self, *, g_scale=1.0, wakeup_boost=1.0, prior_scale=1.0) -> None:
        """Bind single-step generator/prior modulation factors."""
        self._mod_g_scale = g_scale.detach() if isinstance(g_scale, torch.Tensor) else float(g_scale)
        self._mod_wakeup_boost = (
            wakeup_boost.detach() if isinstance(wakeup_boost, torch.Tensor) else float(wakeup_boost)
        )
        self._mod_prior_scale = float(prior_scale)

    bind_v4_modulation = bind_axiom_modulation

    def _collect_normal(self, overrides=None, *, clone: bool = False, only_ids: set[int] | None = None):
        out = []
        for group in self.param_groups:
            prior_sampled = group["algorithm"] == "axiom_prior" and group.get("sampled_rows_required", True)
            for param in group["params"]:
                if only_ids is not None and id(param) not in only_ids:
                    out.append(None)
                    continue
                has_ov = overrides is not None and id(param) in overrides
                src = overrides[id(param)] if has_ov else (None if param.grad is None else param.grad.detach())
                if src is not None and prior_sampled:
                    _require(param in self._sampled_rows)
                    src = _mask_prior_rows(src, self._sampled_rows[param])
                elif src is not None and clone and not has_ov:
                    src = src.clone()
                out.append(src)
        return out

    def bind_protected_from_grads(self) -> None:
        """Mark a single-objective backward pass as protected (zero clone when beta1 == 0)."""
        if not self.direction_blend:
            return
        _require(self._protected is None)
        b1_ids = {id(p) for grp in self.param_groups if grp["betas"][0] > 0.0 for p in grp["params"]}
        self._protected = [self._collect_normal(only_ids=b1_ids)] if b1_ids else [[None] * len(self._parameters())]

    def bind_protected_losses(self, losses, *, protected_evaluator=None) -> None:
        """Capture one or two protected loss gradients using standard ``.backward``."""
        del protected_evaluator
        if not self.direction_blend:
            return
        loss_tuple = tuple(losses)
        _require(self._protected is None and 1 <= len(loss_tuple) <= 2 and all(_is_scalar_loss(x) for x in loss_tuple))
        params = self._parameters()
        trainable = [p for p in params if p.requires_grad]
        saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
        normals = []
        try:
            for loss_k in loss_tuple:
                for p in params:
                    p.grad = None
                if trainable:
                    loss_k.backward(retain_graph=True, inputs=trainable)
                normals.append(self._collect_normal(clone=True))
        finally:
            for p, saved_g in zip(params, saved):
                p.grad = saved_g
        _require(_all_normals_finite(normals))
        self._protected = normals

    def backward_with_protected(self, loss_gan: torch.Tensor, aux_loss=None) -> None:
        """Backpropagate ``loss_gan`` and optional ``aux_loss``, protecting only aux-touched params."""
        _require(_is_scalar_loss(loss_gan) and not (self.direction_blend and self._protected is not None))
        has_aux = _is_scalar_loss(aux_loss)
        loss_gan.backward()
        if not self.direction_blend:
            if has_aux:
                aux_loss.backward()
            return
        params = self._parameters()
        if not has_aux:
            self._protected = [[None] * len(params)]
            return
        touched = _reachable_param_ids(aux_loss, {id(p) for p in params})
        if not touched:
            aux_loss.backward()
            self._protected = [[None] * len(params)]
            return
        saved = {id(p): (None if p.grad is None else p.grad.detach().clone()) for p in params if id(p) in touched}
        aux_loss.backward()
        self._protected = [self._collect_normal(saved, only_ids=touched)]

    def abort_pending(self) -> None:
        """Clear all single-step pending rows, modulations, and protected normals."""
        self.clear_sampled_rows()
        self._mod_g_scale = self._mod_wakeup_boost = self._mod_prior_scale = 1.0
        self._protected = None

    def _gradient_sketch(self, pairs, dev):
        rows = []
        for _, grad in pairs:
            if grad.numel() <= 0:
                continue
            flat = (
                grad.permute(0, 2, 3, 1).reshape(-1)
                if grad.ndim == 4 and grad.is_contiguous(memory_format=torch.channels_last)
                else grad.reshape(-1)
            ).detach()
            if (numel := flat.numel()) >= _SKETCH_DIM:
                stride = max(1, numel // _SKETCH_DIM)
                sample = flat[:: stride - (stride > 1 and stride % 2 == 0)][:_SKETCH_DIM].float()
            else:
                sample = torch.cat((flat.float(), flat.new_zeros(_SKETCH_DIM - numel, dtype=_F32)))
            rows.append(sample)
        if not rows:
            return torch.zeros(_SKETCH_DIM, dtype=_F32, device=dev), torch.zeros((), dtype=_BOOL, device=dev)
        accum = me_unit_rows(torch.stack(rows).to(dev))[0].sum(dim=0)
        norm = accum.norm()
        ok = torch.isfinite(norm) & (norm > 0.0)
        return torch.where(ok, accum / norm.clamp_min(1e-30), 0.0), ok

    def _critic_pathological(self, critic_mats, norm_map, dev) -> bool:
        log2_vals = [
            res[1] + res[2] - half_l2
            for idx, (p, g, _, _, half_l2, *_) in critic_mats
            if (res := _resolve_norms(p, g, norm_map.get(idx), False)) is not None
        ]
        if len(log2_vals) < 3:
            return False
        skew = max(log2_vals) - min(log2_vals)
        self.crit_skew_ema = max(skew, 0.97 * self.crit_skew_ema + 0.03 * skew)
        if self.crit_skew_ema <= 2.0:
            return False
        ranks = []
        for _, (p, g, *_, shape_info, _) in critic_mats:
            state = self.state[p]
            rank, vec = stable_rank_power(
                g.reshape(shape_info[0], shape_info[1]), init_v=state.get("v_pow"), return_v=True
            )
            state["v_pow"] = vec.detach()
            ranks.append(rank.to(dev))
        return min(torch.stack(ranks).tolist()) < 1.05

    @staticmethod
    def _critic_gammas(critic_2d):
        buckets, out = {}, [None] * len(critic_2d)
        for pos, (_, (_, g, *_, (rows, cols, min_dim), _)) in enumerate(critic_2d):
            buckets.setdefault((rows, cols, g.device), []).append((pos, g.reshape(rows, cols), min_dim))
        for items in buckets.values():
            ranks = stable_rank_power(torch.stack([item[1] for item in items]))
            gammas = torch.minimum(ranks.new_full(ranks.shape, sqrt(items[0][2])), 2.0 * ranks.sqrt()).double()
            for j, (pos, _, _) in enumerate(items):
                out[pos] = gammas[j]
        return out

    def _two_objective_pythagorean(self) -> dict[int, float]:
        if self._protected is None or len(self._protected) != 2:
            return {}
        n0, n1 = self._protected
        by_role, idx = {}, 0
        for group in self.param_groups:
            pids, g0s, g1s = by_role.setdefault(group["role"], ([], [], []))
            for param in group["params"]:
                pids.append(id(param))
                if param.grad is not None and param.numel() > 0:
                    g0s.append(n0[idx])
                    g1s.append(n1[idx])
                idx += 1
        boosts = {}
        for pids, g0s, g1s in by_role.values():
            if (boost := _role_pythagorean_boost(g0s, g1s)) is not None:
                for pid in pids:
                    boosts[pid] = boost
        return boosts

    def _prep_prior(self, param: torch.Tensor, grad: torch.Tensor, group: dict, beta1: float):
        state = self.state[param]
        row_scales = self._row_scales.get(param)
        req_rows = group.get("sampled_rows_required", True)
        if req_rows:
            row_idx = torch.sort(self._sampled_rows[param])[0] if row_scales is None else self.sampled_rows_for(param)
        else:
            row_idx = torch.arange(len(param), device=param.device, dtype=_LONG)
        if row_idx is None or row_idx.numel() == 0:
            return None
        dedup_mask = row_idx.new_ones(len(row_idx), dtype=_BOOL)
        if req_rows and row_scales is None:
            dedup_mask[1:] = row_idx[1:] != row_idx[:-1]
        dedup_mask = dedup_mask.unsqueeze(1)
        g_rows = grad[row_idx].float()
        finite = torch.isfinite(g_rows).all(dim=-1, keepdim=True)
        direction = torch.where(finite, g_rows, 0.0)
        if beta1 > 0.0:
            m_prior = state.setdefault("m_prior", param.new_zeros(param.shape, dtype=_F32))
            m_prev = m_prior[row_idx]
            direction = torch.where(finite, m_prev.mul(beta1).add(direction, alpha=1.0 - beta1), m_prev)
            m_prior.index_add_(0, row_idx, (direction - m_prev) * dedup_mask.float())
        unit_rows, log2_rows, ok_rows = me_unit_rows(direction)
        valid = dedup_mask & ok_rows
        valid_f = valid.float()
        n_valid = valid_f.sum()
        mean_log2 = ((log2_rows * valid_f).sum() / n_valid.clamp_min(1.0)).double()
        return param, row_idx, row_scales, unit_rows, log2_rows, valid, n_valid.double(), mean_log2

    def _finish_priors(self, pre_priors, synced, offset: int, s_osc, s_base, s_spike, g_eq, decay, has_hist) -> None:
        s_rel = s_osc + (0.0 if self._mod_prior_scale < 1.0 else g_eq * s_base)
        s_peak, cursor = 1.25 * g_eq * s_spike, offset
        for lr_p, item in pre_priors:
            if item is None:
                continue
            n_valid, mean_l2, cursor = synced[cursor], synced[cursor + 1], cursor + 2
            if lr_p <= 0.0:
                continue
            param, row_idx, row_scales, unit_rows, log2_rows, valid, _, _ = item
            state = self.state[param]
            ref_l2 = (
                _update_log2_ref(state, mean_l2, g_eq, decay, True) if n_valid > 0.0 else state.get("log2_ref", 0.0)
            )
            if has_hist:
                ratio = torch.exp2((ref_l2 - log2_rows).clamp(-20.0, 20.0))
                scale = torch.rsqrt(1.0 + (s_rel * ratio + s_peak * (ratio - 1.0).clamp_min(0.0)).square())
                update = torch.where(valid, unit_rows * scale, 0.0)
            else:
                update = torch.where(valid, unit_rows, 0.0)
            if row_scales is not None and len(row_scales) == len(row_idx):
                update = update * row_scales.unsqueeze(1)
            param.index_add_(0, row_idx, update.to(param.dtype), alpha=-lr_p)

    def _route_critic(self, net_items, norm_map, s_therm: float, dev):
        s6_buckets = {}
        critic_2d = [(i, item) for i, item in enumerate(net_items) if item[8]]
        if not critic_2d or self._critic_pathological([(i, it) for i, it in critic_2d if it[7][2] >= 4], norm_map, dev):
            return s6_buckets
        for (i, item), gamma_t in zip(critic_2d, self._critic_gammas(critic_2d)):
            p, g, lr_p, factor, half_l2, beta1, is_gen, (rows, cols, _), _ = item
            net_items[i] = (p, g, lr_p, factor, half_l2, beta1, is_gen, gamma_t, False)
            if s_therm > 0.0:
                trans = rows > cols
                key = (cols, rows) if trans else (rows, cols)
                s6_buckets.setdefault(key, []).append((p, factor, trans, rows, cols, lr_p * s_therm))
        return s6_buckets

    def _step_network(self, net_items, norm_map, coeffs, gen_probe: bool, dev) -> None:
        coh, s_osc, s_base, s_spike, g_eq, decay, has_hist, g_mult, d_mult = coeffs
        fast_p, fast_g, fast_a, fast_gt, upd_p, upd_d, ns_buckets = [], [], [], [], [], [], {}
        for i, (p, g, lr_p, factor, half_l2, beta1, is_gen, extra, _) in enumerate(net_items):
            can_fast = not gen_probe and beta1 == 0.0 and not isinstance(extra, tuple)
            resolved = _resolve_norms(p, g, norm_map.get(i), not can_fast)
            if resolved is None:
                continue
            g_unit, log2_g, log2_p, has_w = resolved
            state = self.state[p]
            cur_l2 = log2_g + log2_p - half_l2
            ref_l2 = _update_log2_ref(state, cur_l2, g_eq, decay, has_w)
            ratio = 2.0 ** max(-20.0, min(20.0, ref_l2 - cur_l2))
            active_smooth = has_hist and (has_w or state.get("log2_init", False))
            s_tikh = (
                (g_eq * (s_base * ratio + s_spike * max(0.0, ratio - 1.0)) + s_osc * ratio) if active_smooth else 0.0
            )
            rate = lr_p * (g_mult if is_gen else d_mult) * factor * coh
            eff = -rate / hypot(1.0, s_tikh)
            if isinstance(g_unit, float):
                fast_p.append(p)
                fast_g.append(g)
                fast_a.append(eff / g_unit)
                fast_gt.append(extra)
            elif isinstance(extra, tuple):
                rows, cols, _ = extra
                trans = rows > cols
                mat = _apply_momentum(state, p, g_unit, beta1).float().reshape(rows, cols).transpose(0, int(trans))
                ns_buckets.setdefault((mat.shape[0], mat.shape[1]), []).append((p, mat, trans, rate, s_tikh))
            else:
                delta = (_apply_momentum(state, p, g_unit, beta1) * eff).to(p.dtype)
                upd_p.append(p)
                upd_d.append(delta if extra is None else delta * extra.to(delta.dtype))
        self._apply_network_updates(fast_p, fast_g, fast_a, fast_gt, upd_p, upd_d, ns_buckets, dev)

    def _apply_network_updates(self, fast_p, fast_g, fast_a, fast_gt, upd_p, upd_d, ns_buckets, dev) -> None:
        if fast_p:
            if any(tensor is not None for tensor in fast_gt):
                scales = torch.tensor(fast_a, dtype=_F64, device=dev)
                one = scales.new_ones(())
                scales = (scales * torch.stack([one if tensor is None else tensor for tensor in fast_gt])).float()
            else:
                scales = torch.tensor(fast_a, dtype=_F32, device=dev)
            self.fast_path_steps += 1
            torch._foreach_addcmul_(fast_p, fast_g, list(scales.unbind(0)))
        for bucket in ns_buckets.values():
            self.ns_2d_activations += len(bucket)
            mats = torch.stack([item[1] for item in bucket])
            smooths = [item[4] for item in bucket]
            low, high = min(smooths), max(smooths)
            rel_smooth = (
                0.0 if high <= 0.0 else (low if low == high else torch.tensor(smooths, dtype=_F32, device=mats.device))
            )
            orth = fp32_tikhonov_ns_batched(mats, rel_smooth=rel_smooth, pre_normalized=True).float()
            for j, (param, _, trans, mult, _) in enumerate(bucket):
                upd_p.append(param)
                upd_d.append((orth[j].transpose(0, int(trans)) * (-mult)).reshape_as(param).to(param.dtype))
        _apply_param_deltas(upd_p, upd_d, dev)

    @staticmethod
    def _apply_schatten6(s6_buckets, d_mult, dev) -> None:
        if not s6_buckets:
            return
        d_mult_t = _as_scalar_tensor(d_mult, dev, _F32)
        for bucket in s6_buckets.values():
            ws = torch.stack(
                [
                    p.float().reshape(rows, cols).transpose(0, int(trans)) / fac
                    for p, fac, trans, rows, cols, _ in bucket
                ]
            )
            rates = torch.tensor([rate for *_, rate in bucket], dtype=_F32, device=ws.device) * d_mult_t.to(ws.device)
            decays = schatten6_decay_batched(ws, rates)
            for j, (param, fac, trans, _, _, _) in enumerate(bucket):
                param.add_((decays[j].transpose(0, int(trans)) * fac).reshape_as(param).to(param.dtype))

    @staticmethod
    def _flush_trust(rec, y_probe: torch.Tensor, y_pre: torch.Tensor, pre_gain):
        dev = y_probe.device
        ema_keys, ema_tensors = rec.take_pending_ema_tensors() if hasattr(rec, "take_pending_ema_tensors") else ([], [])
        packed = torch.stack(
            [
                *_probe_rms_and_finite(y_probe, y_pre),
                _as_scalar_tensor(pre_gain, dev),
                *(t.to(dev, _F64) for t in ema_tensors),
            ]
        ).tolist()
        if ema_tensors:
            rec.apply_flushed_ema_values(ema_keys, packed[3:])
        return packed[0], packed[1], packed[2]

    def _backtrack_trust(self, cache, rec, deltas, coord_scale, d_obs, d_min, d_max):
        gen_mod, probe_args, probe_kwargs, y_pre = cache

        def _eval_alpha(alpha: float) -> float:
            for param, orig, delta in deltas:
                param.copy_(orig + delta * alpha)
            y_cur = eval_probe_forward(gen_mod, probe_args, probe_kwargs, rec)
            if y_cur is None or y_cur.shape != y_pre.shape:
                return float("inf")
            fin, rms = torch.stack(_probe_rms_and_finite(y_cur, y_pre)).tolist()
            return rms / coord_scale if fin > 0.5 else float("inf")

        alpha = 1.0
        if d_min > 0.0 and isfinite(d_obs) and 1e-12 < d_obs < d_min:
            alpha = min(16.0, max(1.0, d_min / (d_obs + d_min / 16.0)))
            d_obs = _eval_alpha(alpha)
        for _ in range(4):
            if d_obs <= d_max:
                break
            self.trust_backtracks_total += 1
            next_a = min(0.5 * alpha, 0.9 * alpha * d_max / d_obs) if isfinite(d_obs) and d_obs > 1e-30 else 0.1 * alpha
            alpha = max(1e-6, next_a)
            d_obs = _eval_alpha(alpha)
        if not isfinite(d_obs) or d_obs > d_max:
            alpha = max(1e-6, 0.5 * alpha)
            for param, orig, delta in deltas:
                param.copy_(orig + delta * alpha)
        return alpha, d_obs

    def _update_func_gain(self, alpha: float, d_obs: float, d_max: float, pg_val: float, s_therm: float) -> None:
        if alpha < 1.0:
            self.ema_func_gain = 0.70 * self.ema_func_gain + 0.30 * max(0.05, min(1.0, pg_val * alpha))
        elif s_therm >= 0.5 and isfinite(d_obs) and (alpha > 1.0 or d_obs >= 0.012):
            if d_max / max(d_obs, 1e-12) * pg_val > 1.5 and self.ema_func_gain < 1.0:
                self.ema_func_gain = 0.995 * self.ema_func_gain + 0.005

    def _finish_trust(self, rec, is_critic: bool, gen_probe: bool, tracked, trust_info, stiff_gate: float):
        if not (gen_probe and tracked):
            if rec is not None and not is_critic:
                rec._g_probe_cache = None
            return 1.0, False
        cache, rec._g_probe_cache = rec._g_probe_cache, None
        gen_mod, probe_args, probe_kwargs, y_pre = cache
        deltas = [(param, orig, (param - orig).detach()) for param, orig in tracked]
        y_probe = eval_probe_forward(gen_mod, probe_args, probe_kwargs, rec)
        if y_probe is None or y_probe.shape != y_pre.shape:
            return 1.0, False
        pre_gain, step_scale, s_therm, is_simp = trust_info
        fin0, rms0, pg_val = self._flush_trust(rec, y_probe, y_pre, pre_gain)
        coord_scale = max(1e-30, float(getattr(rec, "ema_data_scale", 1.0)) / sqrt(max(1, y_pre[0].numel())))
        push_res = float(getattr(rec, "ema_push_res", 0.0))
        d_max = (0.045 + 0.16 * min(1.5, push_res)) * step_scale
        d_min = 0.012 * smoothstep01((push_res - 0.18) / 0.25) * step_scale if (not is_simp and s_therm > 0.10) else 0.0
        d_obs0 = (rms0 / coord_scale) if fin0 > 0.5 else float("inf")
        alpha, d_obs = self._backtrack_trust(cache, rec, deltas, coord_scale, d_obs0, d_min, d_max)
        self._update_func_gain(alpha, d_obs, d_max, pg_val, s_therm)
        rec.g_func_gain = float((1.0 - stiff_gate) * self.ema_func_gain + stiff_gate)
        return alpha, True

    @staticmethod
    def _apply_projected_subset(params, origs, normals, alphas) -> None:
        active_terms = [(alpha, grp) for alpha, grp in zip(alphas, normals) if alpha != 0.0]
        for idx, (param, orig) in enumerate(zip(params, origs)):
            work_dtype = _F64 if param.dtype == _F64 else _F32
            delta = (param - orig).to(work_dtype)
            for alpha, grp in active_terms:
                normal = grp[idx]
                if normal is not None:
                    delta.add_(normal if normal.dtype == work_dtype else normal.to(work_dtype), alpha=-alpha)
            param.copy_(orig + delta.to(param.dtype))

    def _solve_blend_conflict(self, num_normals: int, ad, an2, a01: float, d2: float):
        blend = self.direction_blend_stats
        blend["conflict_steps"] += 1
        antiparallel = (
            num_normals == 2 and an2[0] > 0.0 and an2[1] > 0.0 and a01 / sqrt(an2[0] * an2[1]) <= -1.0 + 2e-24
        )
        if antiparallel:
            blend["pareto_stalls"] += 1
            return (0.0, 0.0), True, True
        blend["blended_steps"] += 1
        return solve_nonascent_multipliers(ad, an2, d2, a01)

    def _apply_direction_blend(self, active_idx, active_params, active_origs) -> None:
        if self._protected is None:
            return
        stats, blend = self.constraint_geometry_stats, self.direction_blend_stats
        if not active_idx:
            stats["steps"] += 1
            return
        normals = [[grp[i] for i in active_idx] for grp in self._protected]
        dots_fn = _blend_dots_one if len(normals) == 1 else _blend_dots_two
        ad, an2, a01, d2 = dots_fn(active_params, active_origs, *normals)
        if not all(map(isfinite, (*ad, *an2, a01, d2))):
            _copy_params(active_params, active_origs)
            raise ValueError("nonfinite direction_blend inner product")
        before_max = max(ad)
        projected, after_max = False, before_max
        if before_max > 0.0:
            alphas, is_zero, projected = self._solve_blend_conflict(len(normals), ad, an2, a01, d2)
            if is_zero:
                _copy_params(active_params, active_origs)
                after_max = after_norm_sq = 0.0
            else:
                self._apply_projected_subset(active_params, active_origs, normals, alphas)
                after_max, after_norm_sq = _blend_post_metrics(ad, an2, a01, d2, alphas)
            ratio = sqrt(after_norm_sq / d2) if d2 > 0.0 else 0.0
            blend["retained_norm_ratio_sum"] += ratio
            blend["max_retained_norm_ratio"] = max(blend["max_retained_norm_ratio"], ratio)
        stats["steps"] += 1
        stats["projected_steps"] += int(projected)
        stats["max_derivative_before"] = max(stats["max_derivative_before"], before_max)
        stats["max_derivative_after"] = max(stats["max_derivative_after"], after_max)

    def _validate_grad(self, param: torch.Tensor, grad: torch.Tensor, is_prior: bool, req_rows: bool) -> None:
        _require(not grad.is_sparse and not grad.is_complex())
        if is_prior and req_rows:
            _require(param in self._sampled_rows)

    def _should_track_probe(self, gen_probe: bool, param: torch.Tensor) -> bool:
        return gen_probe and (not self._gen_probe_ids or id(param) in self._gen_probe_ids)

    def _partition_params(self, lr_scale: float, gen_probe: bool, pyth_boosts: dict[int, float]):
        pairs, net_items, prior_items, tracked, dev = [], [], [], [], None
        prior_boost = self._mod_prior_scale < 1.0
        for group in self.param_groups:
            is_prior = group["algorithm"] == "axiom_prior"
            req_rows = group.get("sampled_rows_required", True)
            is_gen, beta1 = group["role"] != "critic", group["betas"][0]
            base_lr = group["lr"] * lr_scale * (self._mod_prior_scale if is_prior else 1.0)
            for param in group["params"]:
                if (grad := param.grad) is None:
                    continue
                self._validate_grad(param, grad, is_prior, req_rows)
                dev = param.device if dev is None else dev
                self.state[param]["step"] = self.state[param].get("step", 0) + 1
                lr_p = base_lr * pyth_boosts.get(id(param), 1.0)
                if is_prior:
                    prior_items.append((param, grad, lr_p, max(beta1, 0.75) if prior_boost else beta1, group))
                    continue
                pairs.append((param, grad))
                if lr_p <= 0.0:
                    continue
                if self._should_track_probe(gen_probe, param):
                    tracked.append((param, param.detach().clone()))
                factor, half_l2, rows, cols, min_dim = self._param_metrics(param)
                shape_info = (rows, cols, min_dim) if (param.ndim in (2, 4) and min_dim >= 2) else None
                net_items.append(
                    (
                        param,
                        grad,
                        lr_p,
                        factor,
                        half_l2,
                        beta1,
                        is_gen,
                        shape_info,
                        shape_info is not None and not is_gen,
                    )
                )
        return pairs, net_items, prior_items, tracked, dev

    @staticmethod
    def _critic_coupling(is_critic: bool, rec, dev):
        if not is_critic or rec is None or (g_gain := float(getattr(rec, "g_func_gain", 1.0))) >= 0.75:
            return 1.0
        has_dev = hasattr(rec, "dev_ema")
        if not has_dev and not getattr(rec, "margin_initialized", False):
            return 1.0
        ema_margin = rec.dev_ema("ema_margin", dev) if has_dev else float(getattr(rec, "ema_margin", 0.0))
        gate = smoothstep01((ema_margin - 0.28) / 0.22) * smoothstep01((1.40 - ema_margin) / 0.30)
        if has_dev:
            gate = torch.where(rec.dev_ema("margin_initialized", dev) > 0.5, gate, 0.0)
        return 1.0 + gate * (max(0.12, g_gain) - 1.0)

    def _record_controls(self, rec, is_critic: bool):
        if rec is None:
            return 1.0, 1.0, False, False, False
        step_scale = float(getattr(rec, "step_scale", 1.0))
        s_therm = float(getattr(rec, "thermal_scale", 1.0))
        is_simp = bool(getattr(rec, "_simplicial_step", False))
        is_cond = bool(getattr(rec, "cond_initialized", False))
        gen_probe = not is_critic and not is_simp and not is_cond and len(self._protected or ()) < 2
        gen_probe = (
            gen_probe
            and getattr(rec, "_g_probe_cache", None) is not None
            and bool(getattr(rec, "_enable_secant_probe", True))
        )
        return step_scale, s_therm, is_simp, is_cond, gen_probe

    def _sync_step_pack(self, pairs, prior_items, net_items, g_mult, d_mult, dev):
        has_hist = self._prev_step_sketch is not None
        sketch, ok_sk = self._gradient_sketch(pairs, dev)
        if has_hist:
            cos_t = (sketch * self._prev_step_sketch.to(dev, _F32)).sum().double().clamp(-1.0, 1.0)
        else:
            cos_t = ok_sk.new_zeros((), dtype=_F64)
        pre_priors = [(lr_p, self._prep_prior(p, g, grp, b1_p)) for p, g, lr_p, b1_p, grp in prior_items]
        head_scalars = [ok_sk.double(), cos_t, g_mult, _as_scalar_tensor(d_mult, dev)]
        prior_offset = len(head_scalars)
        for _, item in pre_priors:
            if item is not None:
                head_scalars.extend((item[6], item[7]))
        fast_idx, fast_ps, fast_gs = [], [], []
        for i, (p, g, *_) in enumerate(net_items):
            if p.dtype == g.dtype == _F32 and p.device == dev and g.numel() > 0:
                fast_idx.append(i)
                fast_ps.append(p)
                fast_gs.append(g)
        norm_offset = len(head_scalars)
        head_t = torch.stack(head_scalars)
        if fast_gs:
            norms_t = torch.stack(torch._foreach_norm(fast_gs + fast_ps)).double()
            synced = torch.cat((head_t, norms_t)).tolist()
        else:
            synced = head_t.tolist()
        if pairs and (not has_hist or synced[0] > 0.5):
            self._prev_step_sketch = sketch
        n_fast = len(fast_idx)
        norm_map = dict(zip(fast_idx, zip(synced[norm_offset : norm_offset + n_fast], synced[norm_offset + n_fast :])))
        return has_hist, synced, prior_offset, pre_priors, norm_map

    def _step_core(self, active_idx, active_params, active_origs) -> bool:
        rec, is_critic = self._active_record(), self.critic is not None
        step_scale, s_therm, is_simp, is_cond, gen_probe = self._record_controls(rec, is_critic)
        therm_scale = 1.0 if s_therm >= 0.9999 else s_therm / hypot(s_therm, 0.0061)
        pyth_boosts = self._two_objective_pythagorean() if (is_cond and self.direction_blend) else {}
        pairs, net_items, prior_items, tracked, dev = self._partition_params(
            step_scale * therm_scale, gen_probe, pyth_boosts
        )
        if dev is None:
            self.last_step_gain, self.last_probe_consumed = 1.0, False
            if rec is not None and not is_critic:
                rec._g_probe_cache = None
            return False
        ema_fg = self.ema_func_gain
        pre_gain = (_as_scalar_tensor(self._mod_wakeup_boost, dev) * (1.0 if ema_fg >= 0.75 else ema_fg)).clamp_max(1.0)
        g_mult = _as_scalar_tensor(self._mod_g_scale, dev) * pre_gain
        d_mult = self._critic_coupling(is_critic, rec, dev)
        decay = 0.005 if s_therm >= 0.5 else 0.0
        has_hist, synced, prior_off, pre_priors, norm_map = self._sync_step_pack(
            pairs, prior_items, net_items, g_mult, d_mult, dev
        )
        ema_cos = (0.90 * self.ema_step_cos + 0.10 * synced[1]) if (has_hist and synced[0] > 0.5) else self.ema_step_cos
        g_down, g_up = smoothstep01((-0.45 - ema_cos) / 0.25), smoothstep01((ema_cos + 0.35) / 0.35)
        self.ema_step_cos = ema_cos
        self.log2_damp = max(-10.0, min(0.0, self.log2_damp - 0.12 * g_down + 0.25 * g_up))
        coh = 2.0**self.log2_damp
        s_osc_raw = self.smoothing * smoothstep01((-0.35 - ema_cos) / 0.35)
        g_eq = smoothstep01((0.55 - ema_cos) / 0.30)
        stiff_gate = max(smoothstep01((0.18 - ema_fg) / 0.07), smoothstep01((0.60 - ema_fg) / 0.30) * g_down)
        smooth_keep = 1.0 - stiff_gate
        s_base = self.smoothing * smooth_keep
        s_osc, s_spike = s_osc_raw * smooth_keep, 4.0 * s_base
        self._finish_priors(pre_priors, synced, prior_off, s_osc, s_base, s_spike, g_eq, decay, has_hist)
        s6_buckets = self._route_critic(net_items, norm_map, s_therm, dev)
        self._step_network(
            net_items,
            norm_map,
            (coh, s_osc, s_base, s_spike, g_eq, decay, has_hist, synced[2], synced[3]),
            gen_probe,
            dev,
        )
        self._apply_schatten6(s6_buckets, d_mult, dev)
        step_gain, probe_used = self._finish_trust(
            rec, is_critic, gen_probe, tracked, (pre_gain, step_scale, s_therm, is_simp), stiff_gate
        )
        if self.direction_blend:
            self._apply_direction_blend(active_idx, active_params, active_origs)
        self.last_step_gain, self.last_probe_consumed = float(step_gain), probe_used
        self.probe_consumptions += int(probe_used)
        return True

    def _snapshot_protected_subset(self):
        if not self.direction_blend or self._protected is None:
            return [], [], []
        all_params = self._parameters()
        active_idx = [i for i in range(len(all_params)) if any(grp[i] is not None for grp in self._protected)]
        active_params = [all_params[i] for i in active_idx]
        return active_idx, active_params, [param.detach().clone() for param in active_params]

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single scale-invariant Axiom optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.direction_blend and (closure is not None or self._protected is None):
            self.abort_pending()
            raise ValueError("direction_blend step requires explicit protected-loss backward")
        active_idx, active_params, active_origs = self._snapshot_protected_subset()
        try:
            ran = self._step_core(active_idx, active_params, active_origs)
        finally:
            self.abort_pending()
        if ran and self.record is not None:
            self.record.record_step(self)
        return loss

    def __getstate__(self):
        return {**self.__dict__, "_metric_cache": {}}

    def state_dict(self):
        """Return schema-2 checkpoint state dict including pending direction-blend state."""
        res = Optimizer.state_dict(self)
        srows = [
            rows.cpu()
            if (g["algorithm"] == "axiom_prior" and (rows := self.sampled_rows_for(g["params"][0])) is not None)
            else None
            for g in self.param_groups
        ]
        meta = {key: getattr(self, key) for key, *_ in _SCALAR_SPECS} | {
            "schema": SCHEMA_VERSION,
            "family": self.family,
            "direction_blend": self.direction_blend,
            "prev_step_sketch": None
            if self._prev_step_sketch is None
            else self._prev_step_sketch.detach().cpu().tolist(),
            "sampled_rows": srows,
        }
        if self.direction_blend:
            meta["constraint_geometry"] = dict(
                stats=dict(self.constraint_geometry_stats),
                blend_stats=dict(self.direction_blend_stats),
                pending=_copy_pending(self._protected),
            )
        res["axiom"] = meta
        if self.record is not None:
            res["regularizer"] = dict(
                optimizer_family=self.family, record=self.record.state_dict(), ema=None, guard=None
            )
        return res

    def _validate_constraint_geometry(self, cg) -> None:
        _require_exact_keys(cg, {"stats", "blend_stats", "pending"})
        _require_exact_keys(cg["stats"], set(self.constraint_geometry_stats))
        _require_exact_keys(cg["blend_stats"], set(self.direction_blend_stats))
        if (pending := cg["pending"]) is None:
            return
        params = self._parameters()
        _require(isinstance(pending, list) and 1 <= len(pending) <= 2)
        for group in pending:
            _require(isinstance(group, list) and len(group) == len(params))
            for tensor, param in zip(group, params):
                _require(tensor is None or (_is_finite_tensor(tensor) and tensor.shape == param.shape))

    def _validate_meta(self, meta) -> None:
        expected = set(_META_KEYS) | ({"constraint_geometry"} if self.direction_blend else set())
        _require_exact_keys(meta, expected)
        _require(
            type(meta["schema"]) is int
            and meta["schema"] == SCHEMA_VERSION
            and meta["family"] == self.family
            and type(meta["direction_blend"]) is bool
            and meta["direction_blend"] == self.direction_blend
        )
        _require(isinstance(meta["sampled_rows"], list) and len(meta["sampled_rows"]) == len(self.param_groups))
        _require(all(_is_finite_scalar(meta[k], lo, hi, slo) for k, lo, hi, slo in _SCALAR_SPECS))
        if (sketch := meta["prev_step_sketch"]) is not None:
            _require(
                isinstance(sketch, list) and len(sketch) == _SKETCH_DIM and all(_is_finite_scalar(x) for x in sketch)
            )
        if self.direction_blend:
            self._validate_constraint_geometry(meta["constraint_geometry"])

    @staticmethod
    def _validate_param_state(state_val, shape) -> None:
        _require(isinstance(state_val, dict) and "m_u" not in state_val)
        _require(type(step_val := state_val.get("step", 0)) is int and step_val >= 0)
        _require("log2_ref" not in state_val or _is_finite_scalar(state_val["log2_ref"]))
        _require("log2_init" not in state_val or type(state_val["log2_init"]) is bool)
        for key in _FP32_STATE_KEYS:
            if key in state_val:
                tensor = state_val[key]
                _require(
                    _is_finite_tensor(tensor)
                    and (key not in ("m_unit", "m_prior") or shape is None or tensor.shape == shape)
                )

    def validate_state_dict(self, saved) -> None:
        """Fail-closed validation of a schema-2 ``AxiomOptimizer`` checkpoint."""
        expected_top = {"state", "param_groups", "axiom"} | ({"regularizer"} if self.record is not None else set())
        _require_exact_keys(saved, expected_top)
        self._validate_meta(saved["axiom"])
        saved_groups, saved_state = saved["param_groups"], saved["state"]
        _require(
            isinstance(saved_groups, list)
            and len(saved_groups) == len(self.param_groups)
            and isinstance(saved_state, dict)
        )
        shapes = {}
        for cur_g, sav_g in zip(self.param_groups, saved_groups):
            _require(
                isinstance(sav_g, dict)
                and sav_g.get("role") == cur_g["role"]
                and sav_g.get("algorithm") == cur_g["algorithm"]
            )
            _require(len(sav_ids := sav_g.get("params", ())) == len(cur_g["params"]))
            for param, sid in zip(cur_g["params"], sav_ids):
                shapes[sid] = param.shape
        for sid, state_val in saved_state.items():
            self._validate_param_state(state_val, shapes.get(sid))
        if self.record is not None:
            reg = saved["regularizer"]
            _require_exact_keys(reg, {"optimizer_family", "record", "ema", "guard"})
            _require(reg["optimizer_family"] == self.family and reg["ema"] is None and reg["guard"] is None)
            copy(self.record).load_state_dict(reg["record"])

    def _restore_fp32_state(self, state_dict) -> None:
        for cur_g, sav_g in zip(self.param_groups, state_dict["param_groups"]):
            for param, sid in zip(cur_g["params"], sav_g["params"]):
                if state_val := state_dict["state"].get(sid):
                    for key in _FP32_STATE_KEYS:
                        if isinstance(tensor := state_val.get(key), torch.Tensor):
                            self.state[param][key] = tensor.detach().to(param.device, _F32).clone()

    def load_state_dict(self, state_dict) -> None:
        """Validate and load a schema-2 checkpoint, preserving FP32 optimizer state."""
        self.validate_state_dict(state_dict)
        meta = state_dict["axiom"]
        Optimizer.load_state_dict(self, {"state": state_dict["state"], "param_groups": state_dict["param_groups"]})
        self._restore_fp32_state(state_dict)
        self.clear_sampled_rows()
        self._metric_cache.clear()
        for key, *_ in _SCALAR_SPECS:
            setattr(self, key, float(meta[key]))
        params = self._parameters()
        dev = params[0].device if params else torch.device("cpu")
        sketch = meta["prev_step_sketch"]
        self._prev_step_sketch = None if sketch is None else torch.tensor(sketch, dtype=_F32, device=dev)
        self._reset_counts()
        for group, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None and group["algorithm"] == "axiom_prior":
                self.set_sampled_rows(group["params"][0], rows)
        self._protected = None
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            self.constraint_geometry_stats = dict(cg["stats"])
            self.direction_blend_stats = dict(cg["blend_stats"])
            self._protected = _copy_pending(cg["pending"], params)
        if self.record is not None:
            self.record.load_state_dict(state_dict["regularizer"]["record"])
            self.record.restore_device_state(dev)


__all__ = [
    "AxiomOptimizer",
    "collect_conv_layouts",
    "fp32_tikhonov_ns_2d",
    "fp32_tikhonov_ns_batched",
    "mantissa_exponent_normalize",
    "zero_pinv_project_nonascent",
]
