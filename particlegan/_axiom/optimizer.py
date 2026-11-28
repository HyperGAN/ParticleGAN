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
_ROLES = ("generator", "encoder", "router", "noise", "critic", "prior", "table")
_OPT_DEFAULTS = (
    ("weight_decay", 0.0),
    ("amsgrad", False),
    ("maximize", False),
    ("differentiable", False),
    ("capturable", False),
    ("foreach", None),
    ("fused", None),
)
_FP32_KEYS = ("v_pow", "m_unit", "v_unit", "m_scale", "m_prior")
_SCALARS = (
    ("last_step_gain", 0.0, None, True),
    ("ema_func_gain", 0.0, 1.0, True),
    ("ema_step_cos", -1.0, 1.0, False),
    ("log2_damp", -10.0, 0.0, False),
    ("crit_skew_ema", 0.0, None, False),
)
_META_KEYS = ("schema", "family", "direction_blend", "prev_step_sketch", "sampled_rows") + tuple(s[0] for s in _SCALARS)
_SKETCH_DIM = 67
_W_LOG2_FLOOR = log2(1e-12)
_ME_EPS_REL_SQ = 1e-12
_ME_UP_CLAMP = 0.25


def _require(ok):
    if not ok:
        raise ValueError("invalid")


def _require_keys(mapping, expected):
    _require(isinstance(mapping, dict) and set(mapping) == expected)


def _is_finite(val, low=None, high=None, strict_low=False):
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not isfinite(val):
        return False
    return (low is None or (val > low if strict_low else val >= low)) and (high is None or val <= high)


def _tensor_finite(tensor):
    return isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all().item())


def _scalar_t(val, dev, dtype=_F64):
    return (
        val.to(dev, dtype).reshape(())
        if isinstance(val, torch.Tensor)
        else torch.full((), float(val), device=dev, dtype=dtype)
    )


def _is_loss(val):
    return isinstance(val, torch.Tensor) and val.ndim == 0 and bool(val.requires_grad)


def _mask_rows(src, rows):
    hit = src.new_zeros(len(src), dtype=_F32)
    hit.index_add_(0, rows, rows.new_ones(len(rows), dtype=_F32))
    return torch.where(hit.gt(0.0).unsqueeze(1), src, 0.0)


def _update_log2_ref(state, cur_l2, g_eq, decay, has_w):
    if not state.get("log2_init", False):
        if has_w:
            state["log2_init"], state["log2_ref"] = True, cur_l2
        return cur_l2
    prev = state.get("log2_ref", 0.0)
    nxt = (
        prev + 0.25 * min(2.0, cur_l2 - prev)
        if cur_l2 > prev
        else prev - (0.25 * (1.0 - g_eq) + decay * g_eq) * min(2.0, prev - cur_l2)
    )
    state["log2_ref"] = nxt
    return nxt


def _apply_momentum(state, param, g_unit, beta1):
    if beta1 <= 0.0:
        return g_unit
    m_unit = state.setdefault("m_unit", param.new_zeros(param.shape, dtype=_F32))
    m_scale = state.setdefault("m_scale", param.new_zeros((), dtype=_F32))
    blended, _, _ = mantissa_exponent_normalize(beta1 * m_unit + (1.0 - beta1) * g_unit)
    m_unit.copy_(blended)
    m_scale.fill_(1.0)
    return blended


def _reachable_ids(loss, param_ids):
    touched, seen, stack = set(), set(), [getattr(loss, "grad_fn", None)]
    while stack:
        if (node := stack.pop()) is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(var := getattr(node, "variable", None), torch.Tensor) and id(var) in param_ids:
            touched.add(id(var))
        stack.extend(nxt for nxt, _ in getattr(node, "next_functions", ()))
    return touched


def _pack_normal_vec(normal_group, disps, dtype):
    parts = [
        (g.reshape(-1) if g.dtype == dtype else g.reshape(-1).to(dtype))
        if g is not None
        else d.new_zeros(d.numel(), dtype=dtype)
        for g, d in zip(normal_group, disps)
    ]
    return torch.cat(parts)


def _blend_dots_packed(params, origs, normals):
    disps = torch._foreach_sub(params, origs)
    dtype = _F64 if disps[0].dtype == _F64 else _F32
    d_vec = torch.cat([d.reshape(-1) if d.dtype == dtype else d.reshape(-1).to(dtype) for d in disps])
    a0_vec = _pack_normal_vec(normals[0], disps, dtype)
    d_sq, n0_sq, n0_d = tensor_dot(d_vec, d_vec), tensor_dot(a0_vec, a0_vec), tensor_dot(a0_vec, d_vec)
    if len(normals) == 1:
        v0, vd, dd = torch.stack((n0_sq, n0_d, d_sq)).tolist()
        return [vd], [v0], 0.0, dd
    a1_vec = _pack_normal_vec(normals[1], disps, dtype)
    v = torch.stack(
        (n0_sq, tensor_dot(a1_vec, a1_vec), n0_d, tensor_dot(a1_vec, d_vec), tensor_dot(a0_vec, a1_vec), d_sq)
    ).tolist()
    return [v[2], v[3]], [v[0], v[1]], v[4], v[5]


def _can_pack_blend(params):
    if len(params) <= 1:
        return False
    dev, dt = params[0].device, params[0].dtype
    total = 0
    for param in params:
        if param.device != dev or param.dtype != dt:
            return False
        total += param.numel()
    return 0 < total <= 262_144


def _blend_dots(params, origs, normals):
    if _can_pack_blend(params):
        return _blend_dots_packed(params, origs, normals)
    zero = params[0].new_zeros((), dtype=_F64)
    n0, n1 = normals[0], (normals[1] if len(normals) == 2 else None)
    n0_sq = n1_sq = n0_d = n1_d = n01 = d_sq = zero
    for idx, (param, orig) in enumerate(zip(params, origs)):
        if param.numel() == 0:
            continue
        disp = param - orig
        d_sq = d_sq + tensor_dot(disp, disp)
        a0 = n0[idx]
        if a0 is not None:
            n0_sq = n0_sq + tensor_dot(a0, a0)
            n0_d = n0_d + tensor_dot(a0, disp)
        if n1 is not None and (a1 := n1[idx]) is not None:
            n1_sq = n1_sq + tensor_dot(a1, a1)
            n1_d = n1_d + tensor_dot(a1, disp)
            if a0 is not None:
                n01 = n01 + tensor_dot(a0, a1)
    if n1 is None:
        v0, vd, dd = torch.stack((n0_sq, n0_d, d_sq)).tolist()
        return [vd], [v0], 0.0, dd
    v = torch.stack((n0_sq, n1_sq, n0_d, n1_d, n01, d_sq)).tolist()
    return [v[2], v[3]], [v[0], v[1]], v[4], v[5]


def _blend_post(ad, an2, a01, d2, alphas):
    a0, a1 = (alphas[0], alphas[1]) if len(alphas) > 1 else (alphas[0], 0.0)
    n0, n1, d0, d1 = (an2[0], an2[1], ad[0], ad[1]) if len(ad) > 1 else (an2[0], 0.0, ad[0], 0.0)
    after = max(d0 - a0 * n0 - a1 * a01, d1 - a0 * a01 - a1 * n1) if len(ad) > 1 else (d0 - a0 * n0)
    return after, max(0.0, d2 - 2.0 * (a0 * d0 + a1 * d1) + a0 * a0 * n0 + a1 * a1 * n1 + 2.0 * a0 * a1 * a01)


def _probe_rms(y_cur, y_pre):
    diff = torch.where(torch.isfinite(y_cur), y_cur.double() - y_pre.double(), 0.0)
    return torch.isfinite(y_cur).all().double(), diff.reshape(y_pre.shape[0], -1).square().mean().sqrt()


def _role_pyth(g0s, g1s):
    if (ref := next((t for t in (*g0s, *g1s) if t is not None), None)) is None:
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
            None if t is None else t.detach().to(p.device if params else "cpu", t.dtype).clone()
            for t, p in zip(g, params or g)
        ]
        for g in pending
    ]


def _resolve_norms(param, grad, pair, need_unit=True):
    if pair is not None and 1e-17 < pair[0] < 1e17 and 0.0 <= pair[1] < 1e17:
        g_norm, p_norm = pair
        has_w = p_norm > 1e-12
        return (grad / g_norm if need_unit else g_norm), log2(g_norm), (log2(p_norm) if has_w else 0.0), has_w
    g_unit, l2_g, ok_g = mantissa_exponent_normalize(grad)
    if not ok_g.item():
        return None
    _, l2_p, ok_p = mantissa_exponent_normalize(param.detach())
    lp = l2_p.item()
    has_w = bool(ok_p.item()) and lp > _W_LOG2_FLOOR
    return g_unit, l2_g.item(), (lp if has_w else 0.0), has_w


def _fast_norm_lists(net_items, dev):
    fast_idx, fast_ps, fast_gs = [], [], []
    for i, (p, g, *_) in enumerate(net_items):
        if p.dtype == g.dtype == _F32 and p.device == dev and g.numel() > 0:
            fast_idx.append(i)
            fast_ps.append(p)
            fast_gs.append(g)
    return fast_idx, fast_ps, fast_gs


def _copy_params(dsts, srcs):
    if not dsts:
        return
    if all(p.device == dsts[0].device and p.dtype == dsts[0].dtype for p in dsts):
        torch._foreach_copy_(dsts, srcs)
    else:
        for dst, src in zip(dsts, srcs):
            dst.copy_(src)


def _add_deltas(params, deltas, dev):
    if not params:
        return
    if all(p.device == dev and p.dtype == params[0].dtype for p in params):
        torch._foreach_add_(params, deltas)
    else:
        for param, delta in zip(params, deltas):
            param.add_(delta)


def _me_scale_factors(log2_s, l2_rms, beta2):
    d_eff = max(-30.0, min(_ME_UP_CLAMP, l2_rms - log2_s))
    l2_eff = log2_s + d_eff
    hi = max(log2_s, l2_eff)
    v_scale = beta2 * (2.0 ** (2.0 * (log2_s - hi))) + (1.0 - beta2) * (2.0 ** (2.0 * (l2_eff - hi)))
    new_s = hi + 0.5 * log2(max(v_scale, 1e-300))
    return new_s, 2.0 ** (log2_s - new_s), 2.0 ** (l2_eff - new_s)


def _me_adam_single(state, param, grad, rate, beta1, beta2):
    numel = param.numel()
    if numel <= 0 or rate <= 0.0:
        return
    g_u, l2_g, ok = mantissa_exponent_normalize(grad)
    if not ok.item():
        return
    l2_rms = float(l2_g.item()) - 0.5 * log2(float(numel))
    g_hat = g_u.float() * sqrt(float(numel))
    if "m_unit" not in state or "v_unit" not in state:
        state["log2_init"], state["log2_ref"] = True, l2_rms
        state["m_unit"], state["v_unit"] = g_hat.clone(), g_hat.square()
        step = g_hat / (state["v_unit"] + _ME_EPS_REL_SQ).sqrt()
        param.add_(step.to(param.dtype), alpha=-rate)
        return
    new_s, r_old, r_new = _me_scale_factors(state.get("log2_ref", l2_rms), l2_rms, beta2)
    m_u, v_u = state["m_unit"], state["v_unit"]
    m_u.mul_(beta1 * r_old).add_(g_hat, alpha=(1.0 - beta1) * r_new)
    v_u.mul_(beta2 * r_old * r_old).addcmul_(g_hat, g_hat, value=(1.0 - beta2) * r_new * r_new)
    state["log2_ref"] = new_s
    param.add_((m_u / (v_u + _ME_EPS_REL_SQ).sqrt()).to(param.dtype), alpha=-rate)


def _apply_me_adam_fast(init_items, fast_items):
    if init_items:
        ps, gs, lrs = (list(col) for col in zip(*init_items))
        denoms = torch._foreach_sqrt(torch._foreach_add(torch._foreach_mul(gs, gs), _ME_EPS_REL_SQ))
        torch._foreach_addcdiv_(ps, gs, denoms, lrs)
    if fast_items:
        ps, gs, ms, vs, m_old, g_new, v_old, v_new, lrs = (list(col) for col in zip(*fast_items))
        torch._foreach_mul_(ms, m_old)
        torch._foreach_add_(ms, torch._foreach_mul(gs, g_new))
        torch._foreach_mul_(vs, v_old)
        torch._foreach_addcmul_(vs, gs, gs, v_new)
        denoms = torch._foreach_sqrt(torch._foreach_add(vs, _ME_EPS_REL_SQ))
        torch._foreach_addcdiv_(ps, ms, denoms, lrs)


def collect_conv_layouts(modules):
    """Map Conv2d/ConvTranspose2d weight ids to ``(is_transposed, groups)``."""
    return {
        id(mod.weight): (isinstance(mod, nn.ConvTranspose2d), int(mod.groups))
        for root in modules
        if isinstance(root, nn.Module)
        for mod in root.modules()
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d))
    }


class AxiomOptimizer(Optimizer):
    """Law-4 scale-invariant, clock-free Axiom optimizer."""

    family = recipe_optimizer_family = "axiom"
    _SUPPORTS_ZEROSYNC_V4_TENSORS: bool = True
    ema_critic = anchor = guard = latent_damping = latent_history = direct_response = direct_history = None

    def __init__(
        self,
        params,
        *,
        lr=0.0002,
        betas=(0.5, 0.999),
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
        self._prev_sketch = self._protected = None
        self._conv_layouts = dict(conv_layouts) if conv_layouts else {}
        self._sampled_rows, self._row_scales, self._probe_ids, self._metrics = {}, {}, set(), {}
        self._g_scale = self._wakeup_boost = self._prior_scale = self.last_step_gain = self.ema_func_gain = 1.0
        self.ema_step_cos = self.log2_damp = self.crit_skew_ema = 0.0
        self.constraint_geometry_stats = dict(
            steps=0, projected_steps=0, max_derivative_before=0.0, max_derivative_after=0.0
        )
        self.direction_blend_stats = dict(
            conflict_steps=0, blended_steps=0, pareto_stalls=0, retained_norm_ratio_sum=0.0, max_retained_norm_ratio=0.0
        )
        self._reset_counts()
        role0 = "critic" if critic is not None else "generator"
        defaults = {
            **dict(_OPT_DEFAULTS),
            "lr": float(lr),
            "betas": tuple(float(b) for b in betas),
            "eps": float(eps),
            "role": role0,
            **options,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            group["role"] = role = group.get("role", group.get("forge_role", role0))
            _require(role in _ROLES)
            is_prior = role in ("prior", "table")
            if is_prior:
                _require(len(group["params"]) == 1 and group["params"][0].ndim == 2)
                group.setdefault("sampled_rows_required", role == "prior")
            group["algorithm"] = "axiom_prior" if is_prior else "axiom"
        if isinstance(critic, nn.Module):
            self.bind_module_layouts((critic,))

    def _reset_counts(self):
        self.ns_2d_activations = self.fast_path_steps = self.trust_backtracks_total = self.probe_consumptions = 0
        self.last_probe_consumed = False

    @staticmethod
    def _check_init(lr, betas, eps, smoothing, direction_blend, options):
        _require(
            _is_finite(lr, 0.0)
            and _is_finite(eps, 0.0, strict_low=True)
            and _is_finite(smoothing, 0.0)
            and type(direction_blend) is bool
            and len(betas) == 2
        )
        for beta in betas:
            _require(_is_finite(beta, 0.0) and beta < 1.0)
        allowed = {k for k, _ in _OPT_DEFAULTS}
        for key, val in options.items():
            if key not in allowed:
                raise TypeError(key)
            _require(val in (None, False, 0, 0.0))

    def bind_module_layouts(self, modules):
        self._conv_layouts.update(collect_conv_layouts(modules))
        self._metrics.clear()

    def _active_record(self):
        if (rec := self.record if self.record is not None else self._axiom_record) is not None:
            return rec
        return next((found for p in self._parameters() if (found := get_tensor_record(p)) is not None), None)

    def _parameters(self):
        return [param for group in self.param_groups for param in group["params"]]

    def set_generator_probe_parameters(self, parameters):
        self._probe_ids = {id(param) for param in parameters}

    def bind_origin_sample(self, sample):
        pass

    def _param_metrics(self, param):
        if (cached := self._metrics.get(id(param))) is not None:
            return cached
        if param.ndim < 2:
            return self._metrics.setdefault(id(param), (0.5 if param.numel() == 1 else 1.0, 0.0, 1, 1, 1))
        if param.ndim == 4:
            c0, c1, kh, kw = param.shape
            is_trans, groups = self._conv_layouts.get(id(param), (False, 1))
            out_c, in_c = (c1 * groups, c0) if is_trans else (c0, c1 * groups)
            if not (kh == kw == groups == 1 and not is_trans):
                area = max(1, (kh * kw) // 4 if is_trans else kh * kw)
                fan = max(1, min(out_c // max(1, groups), in_c // max(1, groups)))
                out = (
                    sqrt(out_c / max(1, in_c) * groups / area) * (fan**0.25),
                    0.5 * log2(max(1, min(out_c, in_c) * kh * kw)),
                    1,
                    1,
                    1,
                )
                return self._metrics.setdefault(id(param), out)
            rows, cols = out_c, in_c
        else:
            rows, cols = param.shape[0], param.numel() // max(1, param.shape[0])
        min_d = min(rows, cols)
        return self._metrics.setdefault(
            id(param), (sqrt(max(1.0, rows / max(1, cols))), 0.5 * log2(max(1, min_d)), rows, cols, min_d)
        )

    def set_sampled_rows(self, param, rows):
        row_t = torch.as_tensor(rows).reshape(-1).to(param.device, _LONG)
        if row_t.device.type == "cpu" and row_t.numel():
            _require(not ((row_t < 0).any().item() or (row_t >= len(param)).any().item()))
        prev = self._sampled_rows.get(param)
        self._sampled_rows[param] = row_t if prev is None else torch.cat((prev, row_t))
        set_active_prior_rows(param, self._sampled_rows[param])
        if (rec := self._active_record()) is not None:
            rec._active_prior_table, rec._active_batch_rows = param, row_t.detach()

    def sampled_rows_for(self, param):
        return (
            None
            if param not in self._sampled_rows
            else torch.unique(self._sampled_rows[param], sorted=True).detach().clone()
        )

    def clear_sampled_rows(self):
        for param in self._sampled_rows:
            clear_active_prior_rows(param)
        self._sampled_rows.clear()
        self._row_scales.clear()
        if (rec := self._active_record()) is not None:
            rec._active_prior_table = rec._active_batch_rows = None

    def set_row_scales(self, param, scales):
        if scales is None:
            self._row_scales.pop(param, None)
        else:
            self._row_scales[param] = scales.detach().to(param.device, _F32)

    def bind_axiom_modulation(self, *, g_scale=1.0, wakeup_boost=1.0, prior_scale=1.0):
        self._g_scale = g_scale.detach() if isinstance(g_scale, torch.Tensor) else float(g_scale)
        self._wakeup_boost = wakeup_boost.detach() if isinstance(wakeup_boost, torch.Tensor) else float(wakeup_boost)
        self._prior_scale = float(prior_scale)

    bind_v4_modulation = bind_axiom_modulation

    def _collect_normal(self, overrides=None, *, clone=False, only_ids=None):
        out = []
        for group in self.param_groups:
            sampled = group["algorithm"] == "axiom_prior" and group.get("sampled_rows_required", True)
            for param in group["params"]:
                if only_ids is not None and id(param) not in only_ids:
                    out.append(None)
                    continue
                has_ov = overrides is not None and id(param) in overrides
                src = overrides[id(param)] if has_ov else (None if param.grad is None else param.grad.detach())
                if src is not None and sampled:
                    _require(param in self._sampled_rows)
                    src = _mask_rows(src, self._sampled_rows[param])
                elif src is not None and clone and not has_ov:
                    src = src.clone()
                out.append(src)
        return out

    def bind_protected_from_grads(self):
        if not self.direction_blend:
            return
        _require(self._protected is None)
        b1_ids = {id(p) for g in self.param_groups if g["betas"][0] > 0.0 for p in g["params"]}
        self._protected = [self._collect_normal(only_ids=b1_ids)] if b1_ids else [[None] * len(self._parameters())]

    def bind_protected_losses(self, losses, *, protected_evaluator=None):
        """Capture one or two protected loss gradients via standard ``.backward``."""
        del protected_evaluator
        if not self.direction_blend:
            return
        loss_t = tuple(losses)
        _require(self._protected is None and 1 <= len(loss_t) <= 2 and all(_is_loss(x) for x in loss_t))
        params = self._parameters()
        trainable = [p for p in params if p.requires_grad]
        saved = [None if p.grad is None else p.grad.detach().clone() for p in params]
        normals = []
        try:
            for loss_k in loss_t:
                for param in params:
                    param.grad = None
                if trainable:
                    loss_k.backward(retain_graph=True, inputs=trainable)
                normals.append(self._collect_normal())
        finally:
            for param, grad in zip(params, saved):
                param.grad = grad
        self._protected = normals

    def backward_with_protected(self, loss_gan, aux_loss=None):
        """Backpropagate ``loss_gan`` + optional ``aux_loss``, protecting only aux-touched params."""
        _require(_is_loss(loss_gan) and not (self.direction_blend and self._protected is not None))
        has_aux = _is_loss(aux_loss)
        loss_gan.backward()
        if not self.direction_blend:
            if has_aux:
                aux_loss.backward()
            return
        params = self._parameters()
        if not has_aux:
            self._protected = [[None] * len(params)]
            return
        touched = _reachable_ids(aux_loss, {id(p) for p in params})
        if not touched:
            aux_loss.backward()
            self._protected = [[None] * len(params)]
            return
        saved = {id(p): (None if p.grad is None else p.grad.detach().clone()) for p in params if id(p) in touched}
        aux_loss.backward()
        self._protected = [self._collect_normal(saved, only_ids=touched)]

    def abort_pending(self):
        self.clear_sampled_rows()
        self._g_scale = self._wakeup_boost = self._prior_scale = 1.0
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
                sample = flat[:: stride - (stride > 1 and stride % 2 == 0)][:_SKETCH_DIM]
                rows.append(sample if sample.dtype == _F32 else sample.float())
            else:
                sample = flat if flat.dtype == _F32 else flat.float()
                rows.append(torch.cat((sample, sample.new_zeros(_SKETCH_DIM - numel, dtype=_F32))))
        if not rows:
            return torch.zeros(_SKETCH_DIM, dtype=_F32, device=dev), torch.zeros((), dtype=_BOOL, device=dev)
        stacked = torch.stack(rows) if all(r.device == dev for r in rows) else torch.stack([r.to(dev) for r in rows])
        acc = me_unit_rows(stacked, need_log2=False)[0].sum(dim=0)
        norm = acc.norm()
        ok = torch.isfinite(norm) & (norm > 0.0)
        return torch.where(ok, acc / norm.clamp_min(1e-30), 0.0), ok

    def _critic_pathological(self, critic_mats, norm_map, dev):
        l2s = [
            res[1] + res[2] - half_l2
            for idx, (p, g, _, _, half_l2, *_) in critic_mats
            if (res := _resolve_norms(p, g, norm_map.get(idx), False)) is not None
        ]
        if len(l2s) < 3:
            return False
        skew = max(l2s) - min(l2s)
        self.crit_skew_ema = max(skew, 0.97 * self.crit_skew_ema + 0.03 * skew)
        if self.crit_skew_ema <= 2.0:
            return False
        ranks = []
        for _, (p, g, *_, shape, _) in critic_mats:
            state = self.state[p]
            rank, vec = stable_rank_power(g.reshape(shape[0], shape[1]), init_v=state.get("v_pow"), return_v=True)
            state["v_pow"] = vec.detach()
            ranks.append(rank.to(dev))
        return min(torch.stack(ranks).tolist()) < 1.05

    @staticmethod
    def _critic_gammas(critic_2d, norm_map):
        buckets, out = {}, [None] * len(critic_2d)
        for pos, (net_idx, (_, g, *_, (rows, cols, min_d), _)) in enumerate(critic_2d):
            pair = norm_map.get(net_idx)
            pre = pair is not None and 1e-8 < pair[0] < 1e8
            mat = g.reshape(rows, cols) / pair[0] if pre else g.reshape(rows, cols)
            buckets.setdefault((rows, cols, g.device, pre), []).append((pos, mat, min_d))
        for (_, _, _, pre), items in buckets.items():
            ranks = stable_rank_power(torch.stack([item[1] for item in items]), pre_normalized=pre)
            gammas = (2.0 * ranks.sqrt()).clamp_max(sqrt(items[0][2])).double()
            for j, (pos, _, _) in enumerate(items):
                out[pos] = gammas[j]
        return out

    def _two_obj_pyth(self):
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
            if (boost := _role_pyth(g0s, g1s)) is not None:
                for pid in pids:
                    boosts[pid] = boost
        return boosts

    def _prep_prior(self, param, grad, group, beta1):
        state, row_sc, req = self.state[param], self._row_scales.get(param), group.get("sampled_rows_required", True)
        if req:
            row_idx = torch.sort(self._sampled_rows[param])[0] if row_sc is None else self.sampled_rows_for(param)
        else:
            row_idx = torch.arange(len(param), device=param.device, dtype=_LONG)
        if row_idx is None or row_idx.numel() == 0:
            return None
        dedup = row_idx.new_ones(len(row_idx), dtype=_BOOL)
        if req and row_sc is None:
            dedup[1:] = row_idx[1:] != row_idx[:-1]
        dedup = dedup.unsqueeze(1)
        direct = grad[row_idx].float()
        if beta1 > 0.0:
            fin = torch.isfinite(direct).all(dim=-1, keepdim=True)
            m_prior = state.setdefault("m_prior", param.new_zeros(param.shape, dtype=_F32))
            m_prev = m_prior[row_idx]
            direct = torch.where(fin, m_prev.mul(beta1).add(torch.where(fin, direct, 0.0), alpha=1.0 - beta1), m_prev)
            m_prior.index_add_(0, row_idx, (direct - m_prev) * dedup.float())
        unit, l2_rows, ok_rows = me_unit_rows(direct)
        valid = dedup & ok_rows
        valid_f = valid.float()
        n_valid = valid_f.sum()
        return (
            param,
            row_idx,
            row_sc,
            unit,
            l2_rows,
            valid,
            n_valid.double(),
            ((l2_rows * valid_f).sum() / n_valid.clamp_min(1.0)).double(),
        )

    def _finish_priors(self, pre_priors, synced, offset, s_osc, s_base, s_spike, g_eq, decay, has_hist):
        s_rel, s_peak, cursor = (
            s_osc + (0.0 if self._prior_scale < 1.0 else g_eq * s_base),
            1.25 * g_eq * s_spike,
            offset,
        )
        for lr_p, item in pre_priors:
            if item is None:
                continue
            n_valid, mean_l2, cursor = synced[cursor], synced[cursor + 1], cursor + 2
            if lr_p <= 0.0:
                continue
            param, row_idx, row_sc, unit, l2_rows, valid, _, _ = item
            state = self.state[param]
            ref_l2 = (
                _update_log2_ref(state, mean_l2, g_eq, decay, True) if n_valid > 0.0 else state.get("log2_ref", 0.0)
            )
            if has_hist:
                ratio = torch.exp2((ref_l2 - l2_rows).clamp(-20.0, 20.0))
                upd = torch.where(
                    valid,
                    unit * torch.rsqrt(1.0 + (s_rel * ratio + s_peak * (ratio - 1.0).clamp_min(0.0)).square()),
                    0.0,
                )
            else:
                upd = torch.where(valid, unit, 0.0)
            if row_sc is not None and len(row_sc) == len(row_idx):
                upd = upd * row_sc.unsqueeze(1)
            param.index_add_(0, row_idx, upd.to(param.dtype), alpha=-lr_p)

    def _route_critic(self, net_items, norm_map, s_therm, dev):
        s6_b, critic_2d = {}, [(i, item) for i, item in enumerate(net_items) if item[8]]
        if not critic_2d or self._critic_pathological([(i, it) for i, it in critic_2d if it[7][2] >= 4], norm_map, dev):
            return s6_b
        for (i, item), gamma_t in zip(critic_2d, self._critic_gammas(critic_2d, norm_map)):
            p, g, lr_p, fac, half_l2, beta1, is_gen, (rows, cols, _), _ = item
            net_items[i] = (p, g, lr_p, fac, half_l2, beta1, is_gen, gamma_t, False)
            if s_therm > 0.0:
                trans = rows > cols
                s6_b.setdefault((cols, rows) if trans else (rows, cols), []).append(
                    (p, fac, trans, rows, cols, lr_p * s_therm)
                )
        return s6_b

    def _step_network(self, net_items, norm_map, coeffs, gen_probe, dev):
        del gen_probe, dev
        coh, g_mult, d_mult = coeffs[0], coeffs[7], coeffs[8]
        init_items, fast_items = [], []
        for i, (p, g, lr_p, beta1, beta2, is_gen) in enumerate(net_items):
            rate = lr_p * (g_mult if is_gen else d_mult) * coh
            if rate <= 0.0 or (numel := p.numel()) <= 0:
                continue
            pair = norm_map.get(i)
            state = self.state[p]
            if pair is None or not (1e-17 < pair[0] < 1e17) or p.dtype != _F32 or g.dtype != _F32:
                _me_adam_single(state, p, g, rate, beta1, beta2)
                continue
            g_rms = pair[0] / sqrt(float(numel))
            l2_rms = log2(max(g_rms, 1e-300))
            if "m_unit" not in state or "v_unit" not in state:
                g_hat = g / g_rms
                state["log2_init"], state["log2_ref"] = True, l2_rms
                state["m_unit"], state["v_unit"] = g_hat.clone(), g_hat.square()
                init_items.append((p, g_hat, -rate))
                continue
            new_s, r_old, r_new = _me_scale_factors(state.get("log2_ref", l2_rms), l2_rms, beta2)
            state["log2_ref"] = new_s
            fast_items.append(
                (
                    p,
                    g,
                    state["m_unit"],
                    state["v_unit"],
                    beta1 * r_old,
                    (1.0 - beta1) * r_new / g_rms,
                    beta2 * r_old * r_old,
                    (1.0 - beta2) * r_new * r_new / (g_rms * g_rms),
                    -rate,
                )
            )
        if init_items or fast_items:
            self.fast_path_steps += 1
            _apply_me_adam_fast(init_items, fast_items)

    def _apply_updates(self, fast_p, fast_g, fast_a, fast_gt, upd_p, upd_d, ns_b, dev):
        if fast_p:
            self.fast_path_steps += 1
            for param, grad, alpha, gamma_t in zip(fast_p, fast_g, fast_a, fast_gt):
                if gamma_t is None:
                    param.add_(grad, alpha=alpha)
                else:
                    param.addcmul_(grad, (gamma_t * alpha).to(param.dtype))
        for bucket in ns_b.values():
            self.ns_2d_activations += len(bucket)
            mats = torch.stack([b[1] for b in bucket])
            smooths = [b[4] for b in bucket]
            lo, hi = min(smooths), max(smooths)
            rel_sm = 0.0 if hi <= 0.0 else (lo if lo == hi else smooths)
            orth = fp32_tikhonov_ns_batched(mats, rel_smooth=rel_sm, pre_normalized=True)
            if orth.dtype != _F32:
                orth = orth.float()
            for j, (param, _, trans, mult, _) in enumerate(bucket):
                delta = (orth[j].transpose(0, int(trans)) * (-mult)).reshape_as(param)
                upd_p.append(param)
                upd_d.append(delta if delta.dtype == param.dtype else delta.to(param.dtype))
        _add_deltas(upd_p, upd_d, dev)

    @staticmethod
    def _apply_schatten6(s6_b, d_mult_val):
        if not s6_b:
            return
        for bucket in s6_b.values():
            if len(bucket) == 1:
                param, fac, trans, rows, cols, rate = bucket[0]
                mat = (param if param.dtype == _F32 else param.float()).reshape(rows, cols).transpose(0, int(trans))
                ws, rates = (mat / fac).unsqueeze(0), rate * d_mult_val
            else:
                ws = torch.stack(
                    [
                        (p if p.dtype == _F32 else p.float()).reshape(r, c).transpose(0, int(tr)) / fac
                        for p, fac, tr, r, c, _ in bucket
                    ]
                )
                rates = [rate * d_mult_val for *_, rate in bucket]
            decays = schatten6_decay_batched(ws, rates)
            for j, (param, fac, trans, _, _, _) in enumerate(bucket):
                delta = decays[j].transpose(0, int(trans)).reshape(param.shape)
                param.add_(delta if delta.dtype == param.dtype else delta.to(param.dtype), alpha=fac)

    @staticmethod
    def _flush_trust(rec, y_probe, y_pre, pre_gain):
        dev = y_probe.device
        keys, tensors = rec.take_pending_ema_tensors() if hasattr(rec, "take_pending_ema_tensors") else ([], [])
        packed = torch.stack(
            [*_probe_rms(y_probe, y_pre), _scalar_t(pre_gain, dev), *(t.to(dev, _F64) for t in tensors)]
        ).tolist()
        if tensors:
            rec.apply_flushed_ema_values(keys, packed[3:])
        return packed[0], packed[1], packed[2]

    def _backtrack_trust(self, cache, rec, deltas, coord_sc, d_obs, d_min, d_max):
        gen_mod, args, kwargs, y_pre = cache

        def _eval(alpha):
            for param, orig, delta in deltas:
                param.copy_(orig + delta * alpha)
            y_cur = eval_probe_forward(gen_mod, args, kwargs, rec)
            if y_cur is None or y_cur.shape != y_pre.shape:
                return float("inf")
            fin, rms = torch.stack(_probe_rms(y_cur, y_pre)).tolist()
            return rms / coord_sc if fin > 0.5 else float("inf")

        alpha = 1.0
        if d_min > 0.0 and isfinite(d_obs) and 1e-12 < d_obs < d_min:
            alpha = min(16.0, max(1.0, d_min / (d_obs + d_min / 16.0)))
            d_obs = _eval(alpha)
        for _ in range(4):
            if d_obs <= d_max:
                break
            self.trust_backtracks_total += 1
            nxt = min(0.5 * alpha, 0.9 * alpha * d_max / d_obs) if isfinite(d_obs) and d_obs > 1e-30 else 0.1 * alpha
            alpha = max(1e-6, nxt)
            d_obs = _eval(alpha)
        if not isfinite(d_obs) or d_obs > d_max:
            alpha = max(1e-6, 0.5 * alpha)
            for param, orig, delta in deltas:
                param.copy_(orig + delta * alpha)
        return alpha, d_obs

    def _update_func_gain(self, alpha, d_obs, d_max, pg_val, s_therm):
        if alpha < 1.0:
            self.ema_func_gain = 0.70 * self.ema_func_gain + 0.30 * max(0.05, min(1.0, pg_val * alpha))
        elif s_therm >= 0.5 and isfinite(d_obs) and (alpha > 1.0 or d_obs >= 0.012):
            if d_max / max(d_obs, 1e-12) * pg_val > 1.5 and self.ema_func_gain < 1.0:
                self.ema_func_gain = 0.995 * self.ema_func_gain + 0.005

    def _finish_trust(self, rec, is_critic, gen_probe, tracked, info, stiff_gate):
        if not (gen_probe and tracked):
            if rec is not None and not is_critic:
                rec._g_probe_cache = None
            return 1.0, False
        cache, rec._g_probe_cache = rec._g_probe_cache, None
        gen_mod, args, kwargs, y_pre = cache
        deltas = [(p, o, (p - o).detach()) for p, o in tracked]
        y_probe = eval_probe_forward(gen_mod, args, kwargs, rec)
        if y_probe is None or y_probe.shape != y_pre.shape:
            return 1.0, False
        pre_gain, step_sc, s_therm, is_simp = info
        fin0, rms0, pg_val = self._flush_trust(rec, y_probe, y_pre, pre_gain)
        coord_sc = max(1e-30, float(getattr(rec, "ema_data_scale", 1.0)) / sqrt(max(1, y_pre[0].numel())))
        push_res = float(getattr(rec, "ema_push_res", 0.0))
        d_max = (0.045 + 0.16 * min(1.5, push_res)) * step_sc
        d_min = 0.012 * smoothstep01((push_res - 0.18) / 0.25) * step_sc if (not is_simp and s_therm > 0.10) else 0.0
        alpha, d_obs = self._backtrack_trust(
            cache, rec, deltas, coord_sc, (rms0 / coord_sc) if fin0 > 0.5 else float("inf"), d_min, d_max
        )
        self._update_func_gain(alpha, d_obs, d_max, pg_val, s_therm)
        rec.g_func_gain = float((1.0 - stiff_gate) * self.ema_func_gain + stiff_gate)
        return alpha, True

    @staticmethod
    def _apply_projected(params, origs, normals, alphas):
        active = [(alpha, grp) for alpha, grp in zip(alphas, normals) if alpha != 0.0]
        for idx, (param, orig) in enumerate(zip(params, origs)):
            dtype = _F64 if param.dtype == _F64 else _F32
            delta = (param - orig).to(dtype)
            for alpha, grp in active:
                if (normal := grp[idx]) is not None:
                    delta.add_(normal if normal.dtype == dtype else normal.to(dtype), alpha=-alpha)
            param.copy_(orig + delta.to(param.dtype))

    def _solve_conflict(self, n_cons, ad, an2, a01, d2):
        blend = self.direction_blend_stats
        blend["conflict_steps"] += 1
        if n_cons == 2 and an2[0] > 0.0 and an2[1] > 0.0 and a01 / sqrt(an2[0] * an2[1]) <= -1.0 + 2e-24:
            blend["pareto_stalls"] += 1
            return (0.0, 0.0), True, True
        blend["blended_steps"] += 1
        return solve_nonascent_multipliers(ad, an2, d2, a01)

    def _apply_dir_blend(self, active_idx, active_params, active_origs):
        if self._protected is None:
            return
        stats, blend = self.constraint_geometry_stats, self.direction_blend_stats
        if not active_idx:
            stats["steps"] += 1
            return
        normals = [[grp[i] for i in active_idx] for grp in self._protected]
        ad, an2, a01, d2 = _blend_dots(active_params, active_origs, normals)
        if not all(map(isfinite, (*ad, *an2, a01, d2))):
            _copy_params(active_params, active_origs)
            raise ValueError("invalid")
        bmax = max(ad)
        proj, after = False, bmax
        if bmax > 0.0:
            alphas, is_zero, proj = self._solve_conflict(len(normals), ad, an2, a01, d2)
            if is_zero:
                _copy_params(active_params, active_origs)
                after = after_sq = 0.0
            else:
                self._apply_projected(active_params, active_origs, normals, alphas)
                after, after_sq = _blend_post(ad, an2, a01, d2, alphas)
            ratio = sqrt(after_sq / d2) if d2 > 0.0 else 0.0
            blend["retained_norm_ratio_sum"] += ratio
            blend["max_retained_norm_ratio"] = max(blend["max_retained_norm_ratio"], ratio)
        stats["steps"] += 1
        stats["projected_steps"] += int(proj)
        stats["max_derivative_before"] = max(stats["max_derivative_before"], bmax)
        stats["max_derivative_after"] = max(stats["max_derivative_after"], after)

    def _validate_grad(self, param, grad, is_prior, req):
        _require(not grad.is_sparse and not grad.is_complex())
        if is_prior and req:
            _require(param in self._sampled_rows)

    def _track_probe(self, gen_probe, param):
        return gen_probe and (not self._probe_ids or id(param) in self._probe_ids)

    def _partition_params(self, lr_scale, gen_probe, pyth):
        pairs, net_items, prior_items, tracked, dev = [], [], [], [], None
        prior_boost = self._prior_scale < 1.0
        for group in self.param_groups:
            is_prior, req = group["algorithm"] == "axiom_prior", group.get("sampled_rows_required", True)
            is_gen, (beta1, beta2) = group["role"] != "critic", group["betas"]
            base_lr = group["lr"] * lr_scale * (self._prior_scale if is_prior else 1.0)
            for param in group["params"]:
                if (grad := param.grad) is None:
                    continue
                self._validate_grad(param, grad, is_prior, req)
                dev = param.device if dev is None else dev
                self.state[param]["step"] = self.state[param].get("step", 0) + 1
                lr_p = base_lr * pyth.get(id(param), 1.0)
                if is_prior:
                    prior_items.append((param, grad, lr_p, max(beta1, 0.75) if prior_boost else beta1, group))
                    continue
                pairs.append((param, grad))
                if lr_p <= 0.0:
                    continue
                if self._track_probe(gen_probe, param):
                    tracked.append((param, param.detach().clone()))
                net_items.append((param, grad, lr_p, beta1, beta2, is_gen))
        return pairs, net_items, prior_items, tracked, dev

    @staticmethod
    def _critic_coupling(is_critic, rec, dev):
        if not is_critic or rec is None or (g_gain := float(getattr(rec, "g_func_gain", 1.0))) >= 0.75:
            return 1.0
        has_dev = hasattr(rec, "dev_ema")
        if not has_dev and not getattr(rec, "margin_initialized", False):
            return 1.0
        ema_m = rec.dev_ema("ema_margin", dev) if has_dev else float(getattr(rec, "ema_margin", 0.0))
        gate = smoothstep01((ema_m - 0.28) / 0.22) * smoothstep01((1.40 - ema_m) / 0.30)
        if has_dev:
            gate = torch.where(rec.dev_ema("margin_initialized", dev) > 0.5, gate, 0.0)
        return 1.0 + gate * (max(0.12, g_gain) - 1.0)

    def _record_controls(self, rec, is_critic):
        if rec is None:
            return 1.0, 1.0, False, False, False
        step_sc, s_therm = float(getattr(rec, "step_scale", 1.0)), float(getattr(rec, "thermal_scale", 1.0))
        is_simp, is_cond = bool(getattr(rec, "_simplicial_step", False)), bool(getattr(rec, "cond_initialized", False))
        gen_probe = not is_critic and not is_simp and not is_cond and len(self._protected or ()) < 2
        gen_probe = (
            gen_probe
            and getattr(rec, "_g_probe_cache", None) is not None
            and bool(getattr(rec, "_enable_secant_probe", True))
        )
        return step_sc, s_therm, is_simp, is_cond, gen_probe

    def _resolve_modulations(self, is_critic, rec, dev):
        fg_val = 1.0 if self.ema_func_gain >= 0.75 else self.ema_func_gain
        wb, gs = self._wakeup_boost, self._g_scale
        pre_gain = (
            (wb.to(dev, _F64).reshape(()) * fg_val).clamp_max(1.0)
            if isinstance(wb, torch.Tensor)
            else min(1.0, wb * fg_val)
        )
        if isinstance(gs, torch.Tensor) or isinstance(pre_gain, torch.Tensor):
            g_mult = _scalar_t(gs, dev) * _scalar_t(pre_gain, dev)
        else:
            g_mult = gs * pre_gain
        return pre_gain, g_mult, self._critic_coupling(is_critic, rec, dev)

    def _sync_step_pack(self, pairs, prior_items, net_items, g_mult, d_mult, dev):
        has_hist = self._prev_sketch is not None
        sketch, ok_sk = self._gradient_sketch(pairs, dev)
        cos_t = (
            (sketch * self._prev_sketch.to(dev, _F32)).sum().double().clamp(-1.0, 1.0)
            if has_hist
            else ok_sk.new_zeros((), dtype=_F64)
        )
        pre_priors = [(lr_p, self._prep_prior(p, g, grp, b1_p)) for p, g, lr_p, b1_p, grp in prior_items]
        head = [ok_sk.double(), cos_t]
        g_idx = d_idx = -1
        if isinstance(g_mult, torch.Tensor):
            g_idx = len(head)
            head.append(g_mult)
        if isinstance(d_mult, torch.Tensor):
            d_idx = len(head)
            head.append(d_mult.to(dev, _F64).reshape(()))
        prior_off = len(head)
        for _, item in pre_priors:
            if item is not None:
                head.extend((item[6], item[7]))
        fast_idx, fast_ps, fast_gs = _fast_norm_lists(net_items, dev)
        norm_off = len(head)
        head_t = torch.stack(head)
        synced = (
            torch.cat((head_t, torch.stack(torch._foreach_norm(fast_gs + fast_ps)).double())) if fast_gs else head_t
        ).tolist()
        if pairs and (not has_hist or synced[0] > 0.5):
            self._prev_sketch = sketch
        n_fast = len(fast_idx)
        return (
            has_hist,
            synced,
            prior_off,
            pre_priors,
            dict(zip(fast_idx, zip(synced[norm_off : norm_off + n_fast], synced[norm_off + n_fast :]))),
            synced[g_idx] if g_idx >= 0 else float(g_mult),
            synced[d_idx] if d_idx >= 0 else float(d_mult),
        )

    def _step_core(self, active_idx, active_params, active_origs):
        rec, is_critic = self._active_record(), self.critic is not None
        step_sc, s_therm, is_simp, is_cond, gen_probe = self._record_controls(rec, is_critic)
        therm_sc = 1.0 if s_therm >= 0.9999 else s_therm / hypot(s_therm, 0.0061)
        pyth = self._two_obj_pyth() if (is_cond and self.direction_blend) else {}
        pairs, net_items, prior_items, tracked, dev = self._partition_params(step_sc * therm_sc, gen_probe, pyth)
        if dev is None:
            self.last_step_gain, self.last_probe_consumed = 1.0, False
            if rec is not None and not is_critic:
                rec._g_probe_cache = None
            return False
        ema_fg = self.ema_func_gain
        pre_gain, g_mult, d_mult = self._resolve_modulations(is_critic, rec, dev)
        decay = 0.005 if s_therm >= 0.5 else 0.0
        has_hist, synced, prior_off, pre_priors, norm_map, gm_val, dm_val = self._sync_step_pack(
            pairs, prior_items, net_items, g_mult, d_mult, dev
        )
        ema_cos = (0.90 * self.ema_step_cos + 0.10 * synced[1]) if (has_hist and synced[0] > 0.5) else self.ema_step_cos
        g_down, g_up = smoothstep01((-0.70 - ema_cos) / 0.20), smoothstep01((ema_cos + 0.60) / 0.30)
        self.ema_step_cos, self.log2_damp = ema_cos, max(-10.0, min(0.0, self.log2_damp - 0.12 * g_down + 0.25 * g_up))
        coh, s_osc_raw = 2.0**self.log2_damp, self.smoothing * smoothstep01((-0.35 - ema_cos) / 0.35)
        g_eq = smoothstep01((0.55 - ema_cos) / 0.30)
        stiff = max(smoothstep01((0.18 - ema_fg) / 0.07), smoothstep01((0.60 - ema_fg) / 0.30) * g_down)
        keep = 1.0 - stiff
        s_base = self.smoothing * keep
        s_osc, s_spike = s_osc_raw * keep, 4.0 * s_base
        self._finish_priors(pre_priors, synced, prior_off, s_osc, s_base, s_spike, g_eq, decay, has_hist)
        self._step_network(
            net_items,
            norm_map,
            (coh, s_osc, s_base, s_spike, g_eq, decay, has_hist, gm_val, dm_val),
            gen_probe,
            dev,
        )
        gain, used = self._finish_trust(
            rec, is_critic, gen_probe, tracked, (pre_gain, step_sc, s_therm, is_simp), stiff
        )
        if self.direction_blend:
            self._apply_dir_blend(active_idx, active_params, active_origs)
        self.last_step_gain, self.last_probe_consumed = float(gain), used
        self.probe_consumptions += int(used)
        return True

    def _snapshot_protected(self):
        if not self.direction_blend or self._protected is None:
            return [], [], []
        all_p = self._parameters()
        idx = [i for i in range(len(all_p)) if any(grp[i] is not None for grp in self._protected)]
        params = [all_p[i] for i in idx]
        return idx, params, [p.detach().clone() for p in params]

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Axiom optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.direction_blend and (closure is not None or self._protected is None):
            self.abort_pending()
            raise ValueError("invalid")
        active_idx, active_params, active_origs = self._snapshot_protected()
        try:
            ran = self._step_core(active_idx, active_params, active_origs)
        finally:
            self.abort_pending()
        if ran and self.record is not None:
            self.record.record_step(self)
        return loss

    def __getstate__(self):
        return {**self.__dict__, "_metrics": {}}

    def state_dict(self):
        """Return schema-2 checkpoint state dict."""
        res = Optimizer.state_dict(self)
        srows = [
            rows.cpu()
            if (g["algorithm"] == "axiom_prior" and (rows := self.sampled_rows_for(g["params"][0])) is not None)
            else None
            for g in self.param_groups
        ]
        meta = {k: getattr(self, k) for k, *_ in _SCALARS} | {
            "schema": SCHEMA_VERSION,
            "family": self.family,
            "direction_blend": self.direction_blend,
            "prev_step_sketch": None if self._prev_sketch is None else self._prev_sketch.detach().cpu().tolist(),
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

    def _validate_cg(self, cg):
        _require_keys(cg, {"stats", "blend_stats", "pending"})
        _require_keys(cg["stats"], set(self.constraint_geometry_stats))
        _require_keys(cg["blend_stats"], set(self.direction_blend_stats))
        if (pending := cg["pending"]) is None:
            return
        params = self._parameters()
        _require(isinstance(pending, list) and 1 <= len(pending) <= 2)
        for group in pending:
            _require(isinstance(group, list) and len(group) == len(params))
            for tensor, param in zip(group, params):
                _require(tensor is None or (_tensor_finite(tensor) and tensor.shape == param.shape))

    def _validate_meta(self, meta):
        _require_keys(meta, set(_META_KEYS) | ({"constraint_geometry"} if self.direction_blend else set()))
        _require(
            type(meta["schema"]) is int
            and meta["schema"] == SCHEMA_VERSION
            and meta["family"] == self.family
            and type(meta["direction_blend"]) is bool
            and meta["direction_blend"] == self.direction_blend
        )
        _require(isinstance(meta["sampled_rows"], list) and len(meta["sampled_rows"]) == len(self.param_groups))
        _require(all(_is_finite(meta[k], lo, hi, slo) for k, lo, hi, slo in _SCALARS))
        if (sketch := meta["prev_step_sketch"]) is not None:
            _require(isinstance(sketch, list) and len(sketch) == _SKETCH_DIM and all(_is_finite(x) for x in sketch))
        if self.direction_blend:
            self._validate_cg(meta["constraint_geometry"])

    @staticmethod
    def _validate_pstate(state_val, shape):
        _require(
            isinstance(state_val, dict)
            and "m_u" not in state_val
            and type(st := state_val.get("step", 0)) is int
            and st >= 0
        )
        _require("log2_ref" not in state_val or _is_finite(state_val["log2_ref"]))
        _require("log2_init" not in state_val or type(state_val["log2_init"]) is bool)
        for key in _FP32_KEYS:
            if key in state_val:
                _require(
                    _tensor_finite(t := state_val[key])
                    and (key not in ("m_unit", "v_unit", "m_prior") or shape is None or t.shape == shape)
                )

    def validate_state_dict(self, saved):
        """Fail-closed validation of schema-2 checkpoint state."""
        _require_keys(
            saved, {"state", "param_groups", "axiom"} | ({"regularizer"} if self.record is not None else set())
        )
        self._validate_meta(saved["axiom"])
        sgr, sst = saved["param_groups"], saved["state"]
        _require(isinstance(sgr, list) and len(sgr) == len(self.param_groups) and isinstance(sst, dict))
        shapes = {}
        for cur_g, sav_g in zip(self.param_groups, sgr):
            _require(
                isinstance(sav_g, dict)
                and sav_g.get("role") == cur_g["role"]
                and sav_g.get("algorithm") == cur_g["algorithm"]
            )
            _require(len(sids := sav_g.get("params", ())) == len(cur_g["params"]))
            for param, sid in zip(cur_g["params"], sids):
                shapes[sid] = param.shape
        for sid, state_val in sst.items():
            self._validate_pstate(state_val, shapes.get(sid))
        if self.record is not None:
            reg = saved["regularizer"]
            _require_keys(reg, {"optimizer_family", "record", "ema", "guard"})
            _require(reg["optimizer_family"] == self.family and reg["ema"] is None and reg["guard"] is None)
            copy(self.record).load_state_dict(reg["record"])

    def _restore_fp32(self, state_dict):
        for cur_g, sav_g in zip(self.param_groups, state_dict["param_groups"]):
            for param, sid in zip(cur_g["params"], sav_g["params"]):
                if state_val := state_dict["state"].get(sid):
                    for key in _FP32_KEYS:
                        if isinstance(t := state_val.get(key), torch.Tensor):
                            self.state[param][key] = t.detach().to(param.device, _F32).clone()

    def load_state_dict(self, state_dict):
        """Load validated schema-2 checkpoint preserving FP32 optimizer state."""
        self.validate_state_dict(state_dict)
        meta = state_dict["axiom"]
        Optimizer.load_state_dict(self, {"state": state_dict["state"], "param_groups": state_dict["param_groups"]})
        self._restore_fp32(state_dict)
        self.clear_sampled_rows()
        self._metrics.clear()
        for key, *_ in _SCALARS:
            setattr(self, key, float(meta[key]))
        params = self._parameters()
        dev = params[0].device if params else torch.device("cpu")
        self._prev_sketch = (
            None if (sk := meta["prev_step_sketch"]) is None else torch.tensor(sk, dtype=_F32, device=dev)
        )
        self._reset_counts()
        for group, rows in zip(self.param_groups, meta["sampled_rows"]):
            if rows is not None and group["algorithm"] == "axiom_prior":
                self.set_sampled_rows(group["params"][0], rows)
        self._protected = None
        if self.direction_blend:
            cg = meta["constraint_geometry"]
            self.constraint_geometry_stats, self.direction_blend_stats = dict(cg["stats"]), dict(cg["blend_stats"])
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
