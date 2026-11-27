"""Axiom L2 Law 1: AxiomPriorRegularizer and Whitened Burg Pullback."""

from math import log, tau

import torch
from torch import Tensor, arange, bmm, cat, eye, full, maximum, nn, no_grad, stack, where
from torch.linalg import solve_triangular
import torch.nn.functional as F

from ..optim.constraint_geometry import constraint_geometry_backward
from ..particle_prior import MoGParticlePrior
from ..priors import unwrap_prior
from .atlas import (
    MAX_FULL_COV_DIM,
    PrototypeAtlas,
    SCHEMA_VERSION,
    _ATLAS_TENSOR_KEYS,
    _EPS,
    _INF,
    _VAR_FLOOR,
    _check_scalar,
    _check_schema,
    _check_tensor,
    _copy_cpu,
    _diag,
    _flatten_batch,
    _move_tensor,
    _normalize,
    _require,
    coalesce_pythagorean_ltv,
    cov_from_scalar,
    cov_trace_mean,
    validate_atlas_state,
)
from .math_ops import (
    allocate_deficit_quotas,
    born_infeld_cost,
    log_sinkhorn,
    safe_cdist,
    smoothstep01,
    weyl_rademacher,
)
from .state import get_module_record, set_active_prior_rows

_EMA_KEYS = ("res_ema", "iso_ema", "iso_err_ema")
_ROW_SPECS = (
    ("v_real_ema", torch.float32),
    ("v_tang_ema", torch.float32),
    ("row_m", torch.float32),
    ("row_v", torch.float32),
    ("row_init", torch.bool),
)
_PRIOR_KEYS = frozenset(
    {"schema", "max_prototypes", "atlas", "rows_initialized", *_EMA_KEYS, *(k for k, _ in _ROW_SPECS)}
)
_PRIME_FRACS = tuple((n**0.5) % 1.0 for n in range(2, 320) if all(n % p for p in range(2, int(n**0.5) + 1)))


def _kernel_spec(prior):
    base = unwrap_prior(prior)
    if getattr(base, "code_law", "table") != "table":
        return base, 0.0, False
    if not getattr(prior, "transforms_codes", False):
        if isinstance(base, MoGParticlePrior):
            sigma = base._cached_sigma_float() if getattr(base, "_noise_enabled", False) else 0.0
            return base, sigma, True
        return base, 0.0, hasattr(base, "z")
    manips = getattr(prior, "manipulators", ())
    if (
        len(manips) == 1
        and getattr(manips[0], "normal_noise", False)
        and float(getattr(manips[0], "kappa_radial", 1.0)) == 1.0
    ):
        return base, float(manips[0].sigma), True
    return base, 0.0, False


def has_positive_prior_sigma(prior):
    """True when ``prior`` exposes a positive isotropic Gaussian code kernel."""
    return _kernel_spec(prior)[1] > 0.0


@no_grad()
def _cholesky_spd_safe(cov):
    sym = 0.5 * (cov.float() + cov.float().mT)
    iso_var = _diag(sym).mean(-1, keepdim=True).clamp_min(_VAR_FLOOR).unsqueeze(-1)
    eye_d = eye(sym.shape[-1], device=sym.device)
    chol, info = torch.linalg.cholesky_ex(sym + 1e-6 * iso_var * eye_d)
    return where((info != 0).view(-1, 1, 1), iso_var.sqrt() * eye_d, chol)


@no_grad()
def _box_muller_targets(rows, comp_idx, comp_c, comp_chol, spread):
    dim, num_comp = comp_c.shape[1], len(comp_c)
    num_pairs = (dim + 1) // 2
    phases = (
        (rows.double() + 0.5).unsqueeze(1) * rows.new_tensor(_PRIME_FRACS[: 2 * num_pairs], dtype=torch.double)
    ) % 1.0
    radius = (-2.0 * phases[:, 0::2].clamp(1e-4, 1.0 - 1e-4).log()).sqrt().clamp_max(2.95)
    angle = tau * phases[:, 1::2]
    gauss = stack((radius * angle.cos(), radius * angle.sin()), -1).reshape(len(rows), 2 * num_pairs)[:, :dim]
    gauss = gauss.to(comp_c.device, torch.float32)
    counts = comp_idx.bincount(minlength=num_comp).float()
    comp_mean = gauss.new_zeros((num_comp, dim)).index_add_(0, comp_idx, gauss) / counts.clamp_min(1.0).unsqueeze(1)
    centered = where((counts[comp_idx] >= 2).unsqueeze(1), gauss - comp_mean[comp_idx], gauss)
    return comp_c[comp_idx] + spread[comp_idx].unsqueeze(1) * bmm(comp_chol[comp_idx], centered.unsqueeze(-1)).squeeze(
        -1
    )


def _central_diff(gen_fn, mu, sigma_z, dirs=None):
    batch_m, dim_z = mu.shape
    if dirs is None:
        dirs = eye(dim_z, device=mu.device, dtype=mu.dtype).unsqueeze(0).expand(batch_m, dim_z, dim_z)
    num_dirs = dirs.shape[1]
    offset, mu_exp = float(sigma_z) * dirs, mu.unsqueeze(1)
    pts = cat(
        ((mu_exp + offset).reshape(batch_m * num_dirs, dim_z), (mu_exp - offset).reshape(batch_m * num_dirs, dim_z))
    )
    out = gen_fn(pts).reshape(2, batch_m, num_dirs, -1)
    out = out.float() if out.element_size() < 4 else out
    return 0.5 * (out[0] - out[1])


def _rademacher_diff(gen_fn, mu, sigma_z, phase, salts):
    batch_m, dim_z = mu.shape
    dirs = cat(
        [weyl_rademacher((batch_m, 8, dim_z), phase=phase, salt=s, device=mu.device, dtype=mu.dtype) for s in salts], 1
    )
    return _central_diff(gen_fn, mu, sigma_z, dirs * (dim_z**-0.5))


def central_pullback_cov(gen_fn, mu, sigma_z, *, phase=0.0, salt=0, return_pair=False):
    """Exact (``d_z <= 8``) or Rademacher-sketched central-difference pullback covariance."""
    dim_z = mu.shape[1]
    if dim_z <= 8:
        jac = _central_diff(gen_fn, mu, sigma_z)
        cov = bmm(jac.mT, jac)
        return (cov, cov) if return_pair else cov
    scale = dim_z / 8.0
    jac = _rademacher_diff(gen_fn, mu, sigma_z, phase, (1, 2) if return_pair else (salt,))
    if return_pair:
        return tuple(bmm(part.mT, part) * scale for part in (jac[:, :8], jac[:, 8:]))
    return bmm(jac.mT, jac) * scale


def _burg_exact(jac_det, whitened, inv_w, eps_g, dim_x, dim_z):
    rank = min(dim_x, dim_z)
    gram = bmm(whitened, whitened.mT) if dim_x <= dim_z else (dim_z / dim_x) * bmm(whitened.mT, whitened)
    eye_r, gram = eye(rank, device=whitened.device), 0.5 * (gram + gram.mT)
    logdet = 2.0 * _diag(torch.linalg.cholesky(gram + 1e-3 * eye_r)).clamp_min(_EPS).log().sum(-1)
    burg = (_diag(gram).sum(-1) - logdet - rank * (1.0 - log(1.001))) / rank
    burg_bi = 4.0 * ((1.0 + 0.5 * burg.clamp_min(0.0)).sqrt() - 1.0)
    with no_grad():
        if eps_g is not None:
            tang_var = bmm(jac_det, eps_g.float().unsqueeze(-1)).squeeze(-1).square().sum(-1) / dim_x
        else:
            tang_var = jac_det.square().sum((-2, -1)) / dim_x
        frob_err = (gram.detach() - eye_r).square().sum((-2, -1)).sqrt() / (rank**0.5)
    return 4.5 * (inv_w * burg_bi).mean(), tang_var, frob_err


def _burg_two_sample(jac_det, whitened, inv_w, eps_g, dim_x, dim_z):
    eye_d = eye(dim_x, device=whitened.device)
    gram1, gram2 = (bmm(part, part.mT) * (dim_z / 8.0) for part in (whitened[:, :, :8], whitened[:, :, 8:]))
    with no_grad():
        trace_var = jac_det.square().sum((-2, -1)) * (dim_z / 16.0) / dim_x
        tang_var = trace_var * (eps_g.float().square().sum(-1) / dim_z if eps_g is not None else 1.0)
        frob_err = (0.5 * (gram1.detach() + gram2.detach()) - eye_d).square().sum((-2, -1)).sqrt() / (dim_x**0.5)
    loss = 2.25 * (inv_w * (((gram1 - eye_d) * (gram2 - eye_d)).sum((-2, -1)) / dim_x)).mean()
    return loss, tang_var, frob_err


def whitened_burg_pullback_loss(gen_fn, mu_det, sigma_z, target_cov, inv_w, *, eps_g=None, phase=0.0):
    """Whitened Burg divergence (Born-Infeld compressed exact or 2-sample Rademacher sketch)."""
    batch_n, dim_z = mu_det.shape
    dim_x = target_cov.shape[-1]
    use_exact = dim_z <= max(dim_x, 16)
    sub_n = min(batch_n, max(32, 2048 // max(1, dim_z))) if use_exact else min(batch_n, 256)
    mu_sub = mu_det[:sub_n]
    jac = (
        (
            _central_diff(gen_fn, mu_sub, sigma_z)
            if use_exact
            else _rademacher_diff(gen_fn, mu_sub, sigma_z, phase, (1, 2))
        )
        .float()
        .mT
    )
    whitened = solve_triangular(_cholesky_spd_safe(target_cov[:sub_n]), jac, upper=False)
    eps_sub = None if eps_g is None else eps_g[:sub_n]
    fn = _burg_exact if use_exact else _burg_two_sample
    return fn(jac.detach(), whitened, inv_w[:sub_n].float(), eps_sub, dim_x, dim_z)


def validate_axiom_prior_state(state):
    """Fail-closed schema-2 validator for ``AxiomPriorRegularizer`` checkpoints."""
    _check_schema(state, _PRIOR_KEYS)
    max_p, rows_init = state["max_prototypes"], state["rows_initialized"]
    _require(type(max_p) is int and max_p >= 2 and type(rows_init) is bool)
    for key in _EMA_KEYS:
        _check_scalar(state[key], 0.0)
    atlas_state = validate_atlas_state(state["atlas"])
    _require(atlas_state["max_prototypes"] == max_p)
    num_k = len(atlas_state["proto_c"]) if atlas_state["initialized"] else 0
    for name, _ in _ROW_SPECS[:2]:
        _check_tensor(state[name], (num_k,), nonneg=True) if num_k else _require(state[name] is None)
    if not rows_init:
        _require(all(state[name] is None for name, _ in _ROW_SPECS[2:]))
    else:
        row_m = _check_tensor(state["row_m"])
        _require(row_m.ndim == 2 and min(row_m.shape) >= 1)
        _check_tensor(state["row_v"], (len(row_m), 1), nonneg=True)
        _check_tensor(state["row_init"], (len(row_m), 1), dtype=torch.bool)
    return state


def _ema_property(key):
    def _get(self):
        if key in self._ema_t:
            self._flush_emas()
        return self._ema[key]

    def _set(self, val):
        self._ema[key] = float(val)
        self._ema_t.pop(key, None)

    return property(_get, _set)


def _atlas_property(attr, as_bool=False):
    return property(
        lambda self: getattr(self.atlas, attr),
        lambda self, val: setattr(self.atlas, attr, bool(val) if as_bool else val),
    )


def _scatter_mean(idx, values, num_slots, weights=None):
    if weights is not None:
        denom = values.new_zeros(num_slots).index_add_(0, idx, weights).clamp_min(_EPS)
        numer = values.new_zeros(num_slots).index_add_(0, idx, weights * values)
    else:
        denom = idx.bincount(minlength=num_slots).float().clamp_min(1.0)
        numer = values.new_zeros(num_slots).index_add_(0, idx, values)
    return numer / denom


class AxiomPriorRegularizer(nn.Module):
    """Law 1 prior regularizer owning ``PrototypeAtlas``, curvature EMAs, and row state."""

    SCHEMA = SCHEMA_VERSION
    res_ema, iso_ema, iso_err_ema = (_ema_property(k) for k in _EMA_KEYS)
    quantizer_initialized = _atlas_property("initialized", as_bool=True)
    proto_w, w_fake_ema, proto_n, psi, v_push = (_atlas_property(k) for k in _ATLAS_TENSOR_KEYS[2:])

    def __init__(
        self,
        reference=None,
        *,
        weight=1.0,
        target_std=1.0,
        eps=1e-4,
        max_prototypes=256,
        bound_device=None,
        bound_shape=None,
    ):
        super().__init__()
        _require(type(max_prototypes) is int and max_prototypes >= 2)
        self.weight = _check_scalar(weight, 0.0)
        self.target_std, self.eps = (_check_scalar(v, 0.0, strict=True) for v in (target_std, eps))
        self.max_prototypes, self._bound_device, self._bound_shape = max_prototypes, bound_device, bound_shape
        self.record = self.axiom_record = self.thermal_source = self._var_cache = self._var_override = None
        self.atlas = PrototypeAtlas(max_prototypes=max_prototypes, device=bound_device)
        for name, _ in _ROW_SPECS:
            setattr(self, name, None)
        self.rows_initialized = False
        self._ema = {k: v for k, v in zip(_EMA_KEYS, (1.0, 1.0, 0.0))}
        self._ema_t = {}

    def _set_proto_c(self, val):
        self.atlas.proto_c = val
        if val is not None and val.ndim == 2:
            self.atlas.dim = val.shape[1]

    def _set_proto_cov(self, val):
        self._var_override, self.atlas.proto_cov = None, val

    def _set_proto_var(self, val):
        self._var_override = val
        if val is not None and self.atlas.dim >= 1:
            self.atlas.proto_cov = cov_from_scalar(val, self.atlas.dim)

    proto_c = property(lambda self: self.atlas.proto_c, _set_proto_c)
    proto_cov = property(lambda self: self.atlas.proto_cov, _set_proto_cov)
    proto_var = property(
        lambda self: self._var_override if self._var_override is not None else self.atlas.proto_var, _set_proto_var
    )

    def _flush_emas(self):
        if not self._ema_t:
            return
        tensors = tuple(self._ema_t.values())
        if len({t.device for t in tensors}) == 1:
            vals = stack([t.double() for t in tensors]).tolist()
        else:
            vals = [float(t) for t in tensors]
        self._ema.update({k: float(v) for k, v in zip(self._ema_t, vals)})
        self._ema_t = {}

    def _device_ema(self, key, dev):
        cached = self._ema_t.get(key)
        if cached is not None:
            if cached.device == dev:
                return cached
            self._flush_emas()
        return full((), self._ema[key], device=dev)

    def _resize_curvature(self, ref, num_k, dev):
        v_real, v_tang = self.v_real_ema, self.v_tang_ema
        if v_real is None or v_tang is None or len(v_real) > num_k:
            self.v_real_ema, self.v_tang_ema = ref.clone(), ref.clone()
        else:
            self.v_real_ema, self.v_tang_ema = (
                cat((t.to(dev), ref[len(t) :])) if len(t) < num_k else t.to(dev) for t in (v_real, v_tang)
            )

    def _sync_atlas(self, dev=None):
        atlas = self.atlas
        centers = atlas.proto_c
        if self._var_override is not None and centers is not None:
            atlas.dim = centers.shape[1]
            atlas.proto_cov = cov_from_scalar(self._var_override, centers.shape[1])
            self._var_override = None
        if not atlas.initialized or centers is None:
            self.v_real_ema = self.v_tang_ema = None
            return
        num_k, dst_dev = len(centers), dev if dev is not None else centers.device
        if atlas.proto_n is None or atlas.proto_n.numel() != num_k:
            atlas.proto_n = full((num_k,), 25.0, device=dst_dev)
        if atlas.v_push is None or atlas.v_push.numel() != num_k:
            atlas.v_push = cov_trace_mean(atlas.proto_cov).clone()
        self._resize_curvature(atlas.v_push.to(dst_dev), num_k, dst_dev)

    def forward(self, rows):
        """Repulsive Riesz + marginal std regularization on prior code rows."""
        _require(rows.ndim == 2 and rows.shape[1] >= 1)
        num_rows = len(rows)
        if self.weight <= 0.0 or num_rows < 2:
            return rows[:0].sum()
        sub_n = min(num_rows, 256)
        z = rows[(arange(sub_n, device=rows.device) * num_rows) // sub_n] if num_rows > 256 else rows
        with no_grad():
            dist2 = safe_cdist(z.detach().float()).square()
            tri_i, tri_j = torch.triu_indices(sub_n, sub_n, 1, device=z.device)
            med_d2 = dist2[tri_i, tri_j].median().clamp_min(_EPS).to(z.dtype)
            knn = dist2.fill_diagonal_(_INF).topk(min(8, sub_n - 1), dim=1, largest=False).indices
        repulse = (med_d2 / (med_d2 + (z.unsqueeze(1) - z[knn]).square().sum(-1))).mean()
        std_pen = ((z.var(0, unbiased=False) + self.eps).sqrt() - self.target_std).square().mean()
        loss = self.weight * (repulse + 0.15 * std_pen)
        rec = self.axiom_record or self.record or self.thermal_source or get_module_record(self)
        scale = float(getattr(rec, "thermal_scale", 1.0))
        return (loss * scale + loss.detach() * (1.0 - scale)) if (scale < 1.0 and loss.requires_grad) else loss

    @no_grad()
    def update_quantizer(self, y_real):
        dev = y_real.device
        if self._bound_device is None:
            self._bound_device = dev
        self._sync_atlas(dev)
        self.atlas.update(y_real)
        self._sync_atlas(dev)

    @no_grad()
    def update_curvature_ema(self, proto_ids, real_var_i, tang_var_i):
        self._sync_atlas(proto_ids.device)
        if self.v_real_ema is None:
            return
        num_k, atlas = len(self.v_real_ema), self.atlas
        hit = proto_ids.bincount(minlength=num_k) > 0
        real_mean = _scatter_mean(proto_ids, real_var_i.float(), num_k).clamp_min(_VAR_FLOOR)
        tang_mean = _scatter_mean(proto_ids, tang_var_i.float(), num_k).clamp_min(_VAR_FLOOR)
        self.v_real_ema, self.v_tang_ema, atlas.v_push = (
            where(hit, 0.90 * cur + 0.10 * upd, cur)
            for cur, upd in ((self.v_real_ema, real_mean), (self.v_tang_ema, tang_mean), (atlas.v_push, tang_mean))
        )

    @no_grad()
    def curvature_gamma_per_component(self, coal):
        num_comp = len(coal.comp_c)
        if self.v_real_ema is None or self.v_tang_ema is None:
            return coal.comp_c.new_ones(num_comp)
        weights = self.atlas.proto_w
        weights = coal.comp_c.new_ones(len(self.v_real_ema)) if weights is None else weights.float()
        num = _scatter_mean(coal.proto_comp_id, self.v_tang_ema, num_comp, weights)
        den = _scatter_mean(coal.proto_comp_id, self.v_real_ema, num_comp, weights).clamp_min(_VAR_FLOOR)
        return (num / den).clamp(0.80, 1.35)

    @no_grad()
    def update_laguerre_dual(self, xu_det, c_bar, w_mat, r_macro, laguerre_lr, *, row_mask=None):
        self._sync_atlas(xu_det.device)
        (num_k, dim), atlas = c_bar.shape, self.atlas
        proto_w, w_fake, psi = atlas.proto_w, atlas.w_fake_ema, atlas.psi
        nearest = (safe_cdist(xu_det, c_bar).square() - (w_mat @ psi).unsqueeze(0)).argmin(1)
        assigned = w_mat[nearest]
        if row_mask is None:
            n_eff, w_hat = xu_det.new_full((), len(xu_det)), assigned.mean(0)
        else:
            mask_f = row_mask.float()
            n_eff = mask_f.sum().clamp_min(1.0)
            w_hat = (assigned * mask_f.unsqueeze(1)).sum(0) / n_eff
        atlas.w_fake_ema = w_fake = _normalize(0.88 * w_fake + 0.12 * w_hat)
        p_comp, f_comp = w_mat @ proto_w, w_mat @ w_fake
        tv_floor = 0.35 * (num_k / (n_eff * 8.0).clamp_min(1.0)).sqrt()
        deficit_max = ((p_comp - f_comp).clamp_min(0.0) / p_comp.clamp_min(1.0 / num_k)).amax()
        gate = maximum(
            smoothstep01((0.5 * (proto_w - w_fake).abs().sum() - tv_floor) / tv_floor.clamp_min(0.08)),
            smoothstep01((deficit_max - 0.15) / 0.20),
        )
        r_sq = dim * r_macro.square()
        eta = gate * min(3.0, max(0.35, float(laguerre_lr) * num_k)) * r_sq
        psi_next = w_mat @ (
            psi * (1.0 - 0.004 * gate * (p_comp <= f_comp).float()) + eta * (proto_w - (0.35 * w_hat + 0.65 * w_fake))
        )
        psi_next -= psi_next.mean()
        atlas.psi = psi_next * (
            (2.5 * r_sq).clamp_min(_VAR_FLOOR) / psi_next.abs().amax().clamp_min(_VAR_FLOOR)
        ).clamp_max(1.0)
        return nearest, gate

    @no_grad()
    def apply_prior_row_update(self, z_param, s_idx, is_first, g_rows, part_scale, row_lr, row_settle):
        num_p, dim = z_param.shape
        if not self.rows_initialized or self.row_m is None or self.row_m.shape != (num_p, dim):
            self.row_m = z_param.new_zeros((num_p, dim), dtype=torch.float32)
            self.row_v = z_param.new_zeros((num_p, 1), dtype=torch.float32)
            self.row_init = z_param.new_zeros((num_p, 1), dtype=torch.bool)
            self.rows_initialized = True
        elif self.row_m.device != z_param.device:
            self.row_m, self.row_v, self.row_init = (getattr(self, k).to(z_param.device) for k, _ in _ROW_SPECS[2:])
        row_m, row_v, row_init = self.row_m, self.row_v, self.row_init
        grad, mask_w = g_rows.detach().float(), is_first.float().unsqueeze(1)
        was_init, cur_m, cur_v = row_init[s_idx], row_m[s_idx], row_v[s_idx]
        g_norm2 = grad.square().sum(-1, keepdim=True)
        next_m = where(was_init, 0.80 * cur_m + 0.20 * grad, grad)
        next_v = where(was_init, 0.80 * cur_v + 0.20 * g_norm2, g_norm2)
        row_m.index_add_(0, s_idx, (next_m - cur_m) * mask_w)
        row_v.index_add_(0, s_idx, (next_v - cur_v) * mask_w)
        row_init[s_idx] = was_init | is_first.unsqueeze(1)
        rms = next_v.clamp_min(0.0).sqrt()
        rms_mean = ((rms * mask_w).sum() / mask_w.sum().clamp_min(1.0)).clamp_min(1e-30)
        step = (
            (next_m / (rms + (1e-5 * rms_mean).clamp_min(1e-30)))
            * part_scale.float()
            * mask_w
            * (-float(row_lr) * row_settle.float())
        )
        z_param.index_add_(0, s_idx, step.to(z_param.dtype))

    def state_dict(self, *args, **kwargs):
        self._flush_emas()
        self._sync_atlas()
        return {
            "schema": SCHEMA_VERSION,
            "max_prototypes": self.max_prototypes,
            "atlas": self.atlas.state_dict(),
            "rows_initialized": self.rows_initialized,
            **{k: float(self._ema[k]) for k in _EMA_KEYS},
            **{k: _copy_cpu(getattr(self, k)) for k, _ in _ROW_SPECS},
        }

    def load_state_dict(self, state_dict, strict=True, assign=False, *, validate_only=False):
        val = validate_axiom_prior_state(state_dict)
        _require(
            val["max_prototypes"] == self.max_prototypes
            and (
                not val["rows_initialized"]
                or self._bound_shape is None
                or val["row_m"].shape == tuple(self._bound_shape)
            )
        )
        if validate_only:
            return
        dev = self._bound_device
        self._var_override = self._var_cache = None
        self.atlas.load_state_dict(val["atlas"], device=dev)
        self._ema = {k: float(val[k]) for k in _EMA_KEYS}
        self._ema_t = {}
        self.rows_initialized = val["rows_initialized"]
        for name, dtype in _ROW_SPECS:
            setattr(self, name, _move_tensor(val[name], dev, dtype))

    state_dict_checkpoint, load_state_dict_checkpoint = state_dict, load_state_dict


def _is_point_mass(x):
    flat = _flatten_batch(x)
    return bool(flat.var(0, unbiased=False).sum() < (1e-11 * flat.square().sum(1).mean()).clamp_min(1e-22))


def should_use_axiom_pullback(trainer, real, latent=None, indices=None):
    """Eligibility gate for Axiom Law 1 pullback generator/prior updates."""
    recipe = trainer.recipe
    if (
        recipe.prior_regularizer != "axiom"
        or recipe.t1_pullback_weight <= 0
        or recipe.encoder_mode != "none"
        or recipe.conditioning != "scalar"
        or real.ndim < 2
        or len(real) < 1
    ):
        return False
    dim = real[0].numel()
    if not (1 <= dim <= MAX_FULL_COV_DIM):
        return False
    if dim >= 2 and has_positive_prior_sigma(trainer.prior):
        return True
    is_pt = _is_point_mass(real)
    reg = getattr(trainer, "prior_regularizer", None)
    if hasattr(reg, "_var_cache"):
        reg._var_cache = (real, is_pt)
    return is_pt


@no_grad()
def _ot_route_targets(x_eval, comp_c, comp_w, dist_xc, scale_ratio, res_prev):
    dist_route = dist_xc
    if len(comp_c) >= 4:
        w_col = comp_w.unsqueeze(1)
        c_centered = comp_c - (w_col * comp_c).sum(0, keepdim=True)
        x_white = (x_eval - x_eval.mean(0, keepdim=True)) / x_eval.std(0, unbiased=False, keepdim=True).clamp_min(_EPS)
        c_white = c_centered / (w_col * c_centered.square()).sum(0, keepdim=True).sqrt().clamp_min(_EPS)
        dist_route = where(scale_ratio < 0.35, safe_cdist(x_white, c_white), dist_xc)
    temp = (0.25 * res_prev).clamp(0.010, 0.06)
    log_u = dist_route.new_full((len(dist_route),), -log(max(1, len(dist_route))))
    plan = log_sinkhorn(
        -(dist_route / dist_route.mean().clamp_min(_EPS)) / temp, log_u, comp_w.clamp_min(_EPS).log(), iters=10
    ).exp()
    return (_normalize(plan, 1) @ comp_c).float()


@no_grad()
def _empirical_burg_cov(x_ctr, comp_idx, comp_cov_eff, comp_chol, spread, all_iso):
    num_comp, dim = len(comp_cov_eff), x_ctr.shape[1]
    cnt_b = comp_idx.bincount(minlength=num_comp).float()
    mc_b = x_ctr.new_zeros((num_comp, dim)).index_add_(0, comp_idx, x_ctr) / cnt_b.clamp_min(1.0).unsqueeze(1)
    ew_c = solve_triangular(comp_chol[comp_idx], (x_ctr - mc_b[comp_idx]).unsqueeze(-1), upper=False).squeeze(-1)
    rho_c = x_ctr.new_zeros(num_comp).index_add_(0, comp_idx, ew_c.square().mean(-1)) / (cnt_b - 1.0).clamp_min(1.0)
    rho_cap = (1.0 - dim / cnt_b.clamp_min(1.0)).clamp(0.0, 0.50)
    rho_use = where(all_iso & (cnt_b >= 2.0) & (spread > 1e-3), rho_c.clamp(0.0, 0.65).minimum(rho_cap), 0.0)
    return comp_cov_eff * (1.0 - rho_use).clamp_min(0.45).view(num_comp, 1, 1)


class _PullbackStep:
    def __init__(self, trainer, latent, indices, noise):
        self.trainer, self.reg, self.recipe = trainer, trainer.prior_regularizer, trainer.recipe
        self.base_prior, self.sigma_z, self.can_condense = _kernel_spec(trainer.prior)
        self.z_param = getattr(self.base_prior, "z", None)
        self.has_z_grad = isinstance(self.z_param, Tensor) and self.z_param.requires_grad
        self.has_sigma, self.latent, self.indices, self.noise = self.sigma_z > 0.0, latent, indices, noise

    def _gen(self, z, sigma_out=0.0, rows=None):
        tr = self.trainer
        return tr._postprocess(tr.prior, tr._generate(tr.G, z, sigma_out, self.noise, rows=rows))

    def _active_z_rows(self):
        return self.z_param if self.recipe.num_particles <= 1024 else self.z_param[self.indices.unique()]

    def _code_means(self):
        use_mog = isinstance(self.base_prior, MoGParticlePrior) and not getattr(
            self.trainer.prior, "transforms_codes", False
        )
        if use_mog:
            return self.base_prior.means()[self.indices]
        if self.can_condense and self.z_param is not None:
            return self.z_param[self.indices]
        return None

    def _sample_fakes(self, batch_n, sigma_out, is_pt):
        if is_pt:
            fake_g, mu_det, eps_g = self._gen(self.latent, sigma_out, self.indices), None, None
        else:
            mu = self._code_means()
            mu = self.latent.detach() if mu is None else mu
            mu_det = mu.detach()
            eps_g = ((self.latent - mu) / self.sigma_z).detach() if self.has_sigma else None
            fake_g = self._gen(mu_det + self.sigma_z * eps_g if eps_g is not None else mu_det, sigma_out, self.indices)
        fake_flat = fake_g.reshape(batch_n, -1)
        x_center = self._gen(mu_det).reshape(batch_n, -1) if eps_g is not None else fake_flat
        self.fake_g, self.fake_flat, self.x_center, self.mu_det, self.eps_g = fake_g, fake_flat, x_center, mu_det, eps_g

    def _prepare(self, real, gen_real, sigma_out, critic, is_pt):
        tr, batch_n = self.trainer, len(real)
        self._sample_fakes(batch_n, sigma_out, is_pt)
        if hasattr(tr.opt_g, "bind_origin_sample"):
            tr.opt_g.bind_origin_sample(None if is_pt else self.fake_g)
        real_g = gen_real() if callable(gen_real) else gen_real
        real_g = real if real_g is None else tr._batch(real_g, "generator_real")
        _require(real_g.shape == real.shape)
        fake_logits, real_logits = critic(self.fake_g), critic(real_g)
        disc_rec = getattr(tr.opt_d, "record", None)
        if hasattr(disc_rec, "observe_margin_and_scale"):
            disc_rec.observe_margin_and_scale(real_logits, fake_logits, real_g, self.fake_g)
        self.real_g, self.y_real, self.fake_logits, self.disc_rec = (
            real_g,
            real_g.detach().reshape(batch_n, -1),
            fake_logits,
            disc_rec,
        )
        self.loss_gan = (
            F.softplus(-fake_logits).mean()
            if self.recipe.loss == "non_saturating"
            else tr.loss.g_loss(fake_logits, real_logits)
        )

    def _step_opt_g(self, opt_g, g_scale, wakeup, prior_scale):
        if hasattr(opt_g, "bind_axiom_modulation"):
            opt_g.bind_axiom_modulation(g_scale=g_scale, wakeup_boost=wakeup, prior_scale=prior_scale)
            opt_g.step()
            return
        scale_f, groups = float(g_scale), opt_g.param_groups
        saved_lrs = [g["lr"] for g in groups]
        for group in groups:
            group["lr"] *= scale_f
        try:
            opt_g.step()
        finally:
            for group, lr in zip(groups, saved_lrs):
                group["lr"] = lr

    def _aux_losses(self):
        tr, rec, zero = self.trainer, self.recipe, self.loss_gan.new_zeros(())
        kt = rec.kinetic_transport_loss(self.fake_g, self.real_g) if rec.kinetic_transport_weight else zero
        kl = rec.kinetic_transport_local_loss(self.fake_g, self.real_g) if rec.kinetic_transport_local_weight else zero
        prior_pen = (
            tr.prior.regularize(
                self._active_z_rows(),
                sampled_codes=self.latent,
                indices=self.indices,
                generated=self.fake_g,
                logits=self.fake_logits,
            )
            if getattr(tr, "_prior_penalty", False)
            else None
        )
        l2_pen = rec.prior_l2 * self.z_param.square().mean() if (self.has_z_grad and rec.prior_l2 > 0) else zero
        return prior_pen, kt, kl, kt + kl + (zero if prior_pen is None else prior_pen) + l2_pen

    def _finalize_g(self, loss, loss_d, penalty, g_scale, wakeup, prior_scale, res_t=None):
        tr, disc_rec, opt_g, z_param = self.trainer, self.disc_rec, self.trainer.opt_g, self.z_param
        prior_pen, kt, kl, aux = self._aux_losses()
        loss = loss + aux
        if disc_rec is not None:
            disc_rec._disc_gate = 0.0
            if res_t is not None and hasattr(disc_rec, "_ema_t"):
                disc_rec._ema_t["ema_push_res"] = res_t.detach().double()
            else:
                disc_rec.ema_push_res = float(self.reg.res_ema)
        opt_g.zero_grad()
        if getattr(opt_g, "direction_blend", False) and hasattr(opt_g, "bind_protected_from_grads"):
            loss.backward()
            opt_g.bind_protected_from_grads()
        else:
            constraint_geometry_backward(loss, opt_g, (loss,))
        if self.has_z_grad and z_param.grad is None:
            z_param.grad = z_param.new_zeros(z_param.shape)
        tr.policy.after_generator_backward(loss_gan=self.loss_gan.detach(), loss_critic=(loss_d - penalty).detach())
        self._step_opt_g(opt_g, g_scale, wakeup, prior_scale)
        tr.policy.after_generator_step()
        return loss, prior_pen, kt, kl

    def _point_mass_step(self, loss_d, penalty):
        reg, rec, fake_flat, y_real, dtype = self.reg, self.recipe, self.fake_flat, self.y_real, self.fake_flat.dtype
        batch_n, dim = y_real.shape
        with no_grad():
            err = (fake_flat.detach().float() - y_real.float()).norm(dim=-1)
            cur_err, prev_res = max(float(err.mean()), 0.35 * float(err.amax())), float(reg.res_ema)
            clamped = min(cur_err, 10.0)
            res_ema = clamped if cur_err > max(3.5 * prev_res, prev_res + 0.12) else 0.88 * prev_res + 0.12 * clamped
            reg.res_ema, g_scale = res_ema, max(0.015, min(1.0, res_ema / 0.025))
            z_det = self.latent.detach()
            step = 1e-3 / (max(1, z_det.shape[1]) ** 0.5)
            probe = self._gen(cat((z_det + step, z_det - step))).reshape(2, batch_n, dim).float()
            slope = ((probe[0] - probe[1]).abs() / 2e-3).clamp_min(1e-5)
            inv_w_rail = (slope.quantile(0.5, dim=0, keepdim=True).clamp_min(max(0.15 * res_ema, 1e-4)) / slope).clamp(
                1.0, 12.0
            )
            rail_w = min(1.0, max(0.0, (res_ema - 0.02) / 0.08))
            inv_w = ((1.0 - rail_w) + rail_w * inv_w_rail).to(dtype)
        eps_cusp = max(0.25 * res_ema, 1e-3)
        y_dt = y_real.to(dtype)
        loss_pull = ((eps_cusp * eps_cusp + inv_w * (fake_flat - y_dt).square()).sqrt() - eps_cusp).mean()
        if self.has_sigma:
            mu = self._code_means()
            if mu is not None:
                fake_anti = self._gen(2.0 * mu - self.latent, 0.0, self.indices).reshape(batch_n, dim)
                loss_anti = ((eps_cusp * eps_cusp + inv_w * (fake_anti - y_dt).square()).sqrt() - eps_cusp).mean()
                loss_pull = 0.5 * (loss_pull + loss_anti)
        if self.has_z_grad:
            loss_pull = loss_pull + 0.004 * self.z_param.float().square().mean().to(dtype)
        prior_diag = (
            reg(self._active_z_rows()).detach()
            if (self.has_z_grad and rec.prior_reg > 0)
            else self.loss_gan.new_zeros(())
        )
        loss_g, prior_pen, kt, kl = self._finalize_g(
            0.02 * min(1.0, res_ema / 0.25) * self.loss_gan + rec.t1_pullback_weight * loss_pull,
            loss_d,
            penalty,
            g_scale,
            1.0,
            g_scale,
        )
        return loss_g, self.loss_gan, prior_diag, prior_pen, kt, kl

    @no_grad()
    def _build_targets(self, coal, was_init):
        reg, y_real, comp_c, comp_w, comp_var = self.reg, self.y_real, coal.comp_c, coal.comp_w, coal.comp_var
        atlas, dim, dev, y_f, num_comp = reg.atlas, y_real.shape[1], y_real.device, y_real.float(), len(comp_c)
        sqrt_d = dim**0.5
        x_ctr, fake_det, y_var = (
            self.x_center.detach().float(),
            self.fake_flat.detach().float(),
            y_f.var(0, unbiased=False),
        )
        r_macro = y_var.clamp_min(0.0).sqrt().mean().clamp_min(_EPS)
        sigma_out = (0.5 * y_var.mean().clamp_min(_VAR_FLOOR).sqrt()).clamp_min(_EPS)
        sigma_c = comp_var.sqrt()
        sigma_med = sigma_c.quantile(0.35).clamp_min(_EPS)

        ledger = (
            atlas.commit_particles(x_ctr, self.indices, len(self.z_param), coalesced=coal)
            if (self.can_condense and self.z_param is not None)
            else None
        )
        comp_idx = ledger.batch_comp if ledger is not None else safe_cdist(x_ctr, comp_c).argmin(1)
        x_eval = where(coal.all_iso, x_ctr, fake_det)
        scale_ratio = x_eval.std(0, unbiased=False).mean().clamp_min(_EPS) / r_macro
        wakeup = where(scale_ratio < 0.50, 5.0, where(scale_ratio < 0.80, 2.0, 1.0))

        iso_prev, res_prev = reg._device_ema("iso_ema", dev), reg._device_ema("res_ema", dev)
        iso_cur = (self.sigma_z * r_macro / sigma_med).clamp(1.0, 4.5)
        iso_ema = (
            (iso_cur if not was_init else 0.95 * iso_prev + 0.05 * iso_cur) if self.has_sigma else iso_prev.new_ones(())
        )
        dist_xc = safe_cdist(x_eval, comp_c)
        res_cur = (
            (maximum((comp_w * dist_xc.amin(0)).sum(), dist_xc.amin(1).median()) - 1.45 * sqrt_d * sigma_med).clamp_min(
                0.0
            )
            / r_macro
        ).clamp_max(10.0)
        res_ema = 0.92 * res_prev + 0.08 * res_cur
        reg._ema_t["iso_ema"], reg._ema_t["res_ema"] = iso_ema, res_ema

        comp_cov_eff = coal.comp_cov_tgt * reg.curvature_gamma_per_component(coal).view(num_comp, 1, 1)
        comp_chol = _cholesky_spd_safe(comp_cov_eff)
        y_target = (
            ledger.batch_target
            if ledger is not None
            else _ot_route_targets(x_eval, comp_c, comp_w, dist_xc, scale_ratio, res_prev)
        )

        occ_est = (
            ledger.occupancy.float()
            if ledger is not None
            else ((len(self.z_param) if self.z_param is not None else len(y_f)) * comp_w)
        )
        mp_cap = (1.0 - dim / occ_est.clamp_min(1.0)).clamp(0.0, 0.50)
        vp_comp = _scatter_mean(coal.proto_comp_id, atlas.v_push, num_comp, atlas.proto_w)
        spread = (1.0 - vp_comp / comp_var.clamp_min(_VAR_FLOOR)).clamp_min(0.0).minimum(mp_cap).sqrt()
        if self.can_condense and self.has_sigma:
            halo = (
                ((x_eval - comp_c[comp_idx]).norm(dim=-1) / (3.5 * sqrt_d * sigma_c[comp_idx]).clamp_min(_EPS))
                .clamp(0.0, 1.0)
                .square()
                .unsqueeze(1)
            )
            use_bm = (coal.all_iso & (spread[comp_idx] > 1e-3) & (scale_ratio >= 0.65)).unsqueeze(1)
            y_target = where(
                use_bm,
                halo * y_target + (1.0 - halo) * _box_muller_targets(self.indices, comp_idx, comp_c, comp_chol, spread),
                y_target,
            )

        ce_burg = _empirical_burg_cov(x_ctr, comp_idx, comp_cov_eff, comp_chol, spread, coal.all_iso)
        inv_w = (1.0 / (num_comp * comp_w[comp_idx]).clamp_min(1e-4)).clamp(0.25, 6.0)
        inv_w_burg = inv_w * (sigma_c[comp_idx] / sigma_c.mean().clamp_min(_EPS)).clamp(0.5, 3.0)

        self.r_macro, self.sigma_out, self.use_ledger = r_macro, sigma_out, ledger is not None
        self.iso_ema, self.res_ema, self.comp_chol, self.spread = iso_ema, res_ema, comp_chol, spread
        y_bound = y_f.abs().amax(0, keepdim=True) + 3.5 * sigma_med
        return y_target, sigma_med, y_bound, scale_ratio, wakeup, comp_idx, ce_burg[comp_idx], inv_w, inv_w_burg

    def _burg_step(self, target_cov, inv_w_burg):
        reg = self.reg
        iso_err = reg._device_ema("iso_err_ema", self.y_real.device)
        if not self.has_sigma or self.mu_det is None or self.recipe.t1_iso_weight <= 0:
            return self.fake_flat.new_zeros(()), iso_err
        loss_iso, tang_var, frob_err = whitened_burg_pullback_loss(
            self._gen,
            self.mu_det,
            self.sigma_z,
            target_cov,
            inv_w_burg,
            eps_g=self.eps_g,
            phase=float(getattr(self.disc_rec, "chord_phase", 0.0)),
        )
        with no_grad():
            sub_n = len(tang_var)
            x_ctr = self.x_center[:sub_n].detach().float()
            real_var = (self.fake_flat[:sub_n].detach().float() - x_ctr).square().mean(-1)
            reg.update_curvature_ema(safe_cdist(x_ctr, reg.atlas.proto_c).argmin(1), real_var, tang_var)
            iso_err = 0.92 * iso_err + 0.08 * (0.60 * frob_err.mean() + 0.40 * frob_err.quantile(0.92)).clamp_max(5.0)
            reg._ema_t["iso_err_ema"] = iso_err
        return loss_iso.to(self.fake_flat.dtype), iso_err

    @no_grad()
    def _row_targets(self, coal, sorted_idx, is_first, xu_det, dtype, row_floor):
        row_comp = self.reg.atlas.ledger.row_comp
        if self.use_ledger and row_comp is not None:
            row_comp = row_comp.to(sorted_idx.device)
            comp_sel = where(
                row_comp[sorted_idx] >= 0, row_comp[sorted_idx].clamp_min(0), safe_cdist(xu_det, coal.comp_c).argmin(1)
            )
            sigma_c, lag_gate = coal.comp_var[comp_sel].sqrt().to(dtype), xu_det.new_zeros(())
            target = (
                _box_muller_targets(sorted_idx, comp_sel, coal.comp_c, self.comp_chol, self.spread).to(dtype)
                if self.has_sigma
                else coal.comp_c[comp_sel].to(dtype)
            )
        else:
            comp_sel = None
            proto_sel, lag_gate = self.reg.update_laguerre_dual(
                xu_det, coal.proto_c_bar, coal.w_mat, self.r_macro, self.recipe.t1_laguerre_lr, row_mask=is_first
            )
            sigma_c, target = coal.proto_var_bar[proto_sel].sqrt().to(dtype), coal.proto_c_bar[proto_sel].to(dtype)
        row_settle = maximum(row_floor, maximum(self.res_ema / 0.10, lag_gate).clamp_max(1.0))
        return comp_sel, target, sigma_c, row_settle

    def _row_move_objective(self, xu, xu_det, comp_sel, target, sigma_c, row_w, n_unique, row_settle):
        if comp_sel is not None:
            dim = xu.shape[1]
            sqrt_d = dim**0.5
            ew = solve_triangular(
                self.comp_chol[comp_sel], (xu.float() - target.float()).unsqueeze(-1), upper=False
            ).squeeze(-1)
            cost = (0.16 * dim + ew.square().sum(-1)).sqrt() - 0.4 * sqrt_d
            loss_move = (row_w * cost).sum() / (n_unique * sqrt_d)
            with no_grad():
                en = ew.detach().norm(dim=-1)
                far = smoothstep01((en / sqrt_d - 1.5) / 1.5)
                part_scale = (en / (1.5 * sqrt_d)).clamp(0.02, 2.5).unsqueeze(1) * (
                    row_settle + far * (1.0 - row_settle)
                ).unsqueeze(1)
            return loss_move, part_scale, row_settle.new_ones(())
        cost = born_infeld_cost(xu, target, (0.4 * sigma_c).clamp_min(_EPS), aligned=True)[0]
        loss_move = (row_w.to(cost.dtype) * cost).sum() / (n_unique * self.sigma_out).to(cost.dtype)
        with no_grad():
            part_scale = (
                ((xu_det - target.float()).norm(dim=-1) / (1.5 * sigma_c.float()).clamp_min(_EPS))
                .clamp(0.02, 1.2)
                .unsqueeze(1)
            )
        return loss_move, part_scale, row_settle

    def _update_prior_rows(self, coal, row_floor):
        base_p, z_param, sorted_idx = self.base_prior, self.z_param, self.indices.sort().values
        is_first = cat((sorted_idx.new_ones(1, dtype=torch.bool), sorted_idx[1:] != sorted_idx[:-1]))
        row_w = is_first.float()
        n_unique, gen_params = row_w.sum().clamp_min(1.0), tuple(self.trainer.G.parameters())
        saved_req = [p.requires_grad for p in gen_params]
        try:
            for param in gen_params:
                param.requires_grad_(False)
            code_table = base_p.means() if isinstance(base_p, MoGParticlePrior) else z_param
            xu = self._gen(code_table[sorted_idx]).reshape(len(sorted_idx), -1)
            xu_det = xu.detach().float()
            comp_sel, target, sigma_c, row_settle = self._row_targets(
                coal, sorted_idx, is_first, xu_det, xu.dtype, row_floor
            )
            loss_move, part_scale, eff_settle = self._row_move_objective(
                xu, xu_det, comp_sel, target, sigma_c, row_w, n_unique, row_settle
            )
            if self.has_sigma:
                z_rows = z_param[sorted_idx]
                z_dt = z_rows.dtype
                w_norm = (row_w / n_unique).unsqueeze(1).to(z_dt)
                z_std = ((w_norm * (z_rows - (w_norm * z_rows).sum(0, keepdim=True)).square()).sum(0) + 1e-6).sqrt()
                loss_move = loss_move + 0.15 * F.relu(self.iso_ema.to(z_dt) - z_std).square().mean() * (
                    n_unique >= 2
                ).to(z_dt)
            if z_param.grad is not None:
                z_param.grad.zero_()
            loss_move.backward()
            with no_grad():
                g_rows = z_param.grad[sorted_idx] * n_unique.to(z_param.dtype)
                z_param.grad.zero_()
                self.reg.apply_prior_row_update(
                    z_param, sorted_idx, is_first, g_rows, part_scale, self.recipe.t1_prior_row_lr, eff_settle
                )
            return loss_move.detach()
        finally:
            for param, req in zip(gen_params, saved_req):
                param.requires_grad_(req)
            if self.disc_rec is not None:
                self.disc_rec._g_probe_cache = None

    def run(self, real, gen_real, sigma_out, critic, loss_d, penalty):
        if isinstance(self.z_param, Tensor):
            set_active_prior_rows(self.z_param, self.indices)
        reg, rec = self.reg, self.recipe
        atlas = reg.atlas
        cached, reg._var_cache = reg._var_cache, None
        is_pt = (
            bool(cached[1])
            if (cached is not None and cached[0] is real)
            else (not (self.has_sigma and atlas.initialized and not atlas._last_point_mass) and _is_point_mass(real))
        )
        self._prepare(real, gen_real, sigma_out, critic, is_pt)
        if is_pt:
            return self._point_mass_step(loss_d, penalty)

        was_init = reg.quantizer_initialized
        reg.update_quantizer(self.y_real)
        coal = atlas.coalesce(self.y_real)
        y_target, sigma_med, y_bound, scale_ratio, wakeup, comp_idx, ce_burg, inv_w, inv_w_burg = self._build_targets(
            coal, was_init
        )

        x_src = self.x_center if self.has_sigma else self.fake_flat
        dt, so, all_iso = x_src.dtype, self.sigma_out, coal.all_iso
        dim = self.y_real.shape[1]
        sqrt_d = dim**0.5
        ew = solve_triangular(self.comp_chol[comp_idx], (x_src.float() - y_target).unsqueeze(-1), upper=False).squeeze(
            -1
        )
        bi_w = ((0.25 * dim + ew.square().sum(-1)).sqrt() - 0.5 * sqrt_d) / sqrt_d
        loss_maha = ((inv_w / inv_w.mean().clamp_min(_EPS)) * bi_w).mean().to(dt)
        loss_eucl = born_infeld_cost(x_src, y_target.to(dt), sigma_med.to(dt), aligned=True)[0].mean() / so.to(dt)
        loss_pull = where(all_iso, loss_maha, loss_eucl) + 2.0 * (
            F.relu(self.fake_flat.float().abs() - y_bound) / so
        ).square().mean().to(dt)
        loss_iso, iso_err = self._burg_step(ce_burg, inv_w_burg)

        eq_blend = where(all_iso, ((0.030 - self.res_ema) / 0.020).clamp(0.0, 1.0), 0.0)
        adv_w, g_floor, row_floor = (
            where(all_iso, a * (1.0 - eq_blend) + b * eq_blend, c)
            for a, b, c in ((0.038, 0.008, 0.15), (0.022, 0.012, 0.04), (0.15, 0.38, 0.15))
        )
        g_scale = where(
            scale_ratio < 0.60,
            1.0,
            maximum(g_floor, maximum(self.res_ema / 0.10, ((iso_err - 0.22) / 0.45).clamp(0.0, 1.0)).clamp_max(1.0)),
        )
        total_g = (
            adv_w.to(self.loss_gan.dtype) * self.loss_gan
            + rec.t1_pullback_weight * loss_pull
            + rec.t1_iso_weight * loss_iso
        )
        loss_g, prior_pen, kt, kl = self._finalize_g(total_g, loss_d, penalty, g_scale, wakeup, 1.0, res_t=self.res_ema)
        prior_diag = (
            self._update_prior_rows(coal, row_floor)
            if (self.can_condense and self.has_z_grad and rec.t1_prior_row_lr > 0)
            else self.loss_gan.new_zeros(())
        )
        return loss_g, self.loss_gan, prior_diag, prior_pen, kt, kl


def axiom_pullback_step(trainer, real, generator_real, latent, indices, *, sigma_out, noise, critic, loss_d, penalty):
    """Execute one Axiom Law 1 generator + prior-row pullback training step."""
    return _PullbackStep(trainer, latent, indices, noise).run(real, generator_real, sigma_out, critic, loss_d, penalty)


__all__ = [
    "AxiomPriorRegularizer",
    "SCHEMA_VERSION",
    "allocate_deficit_quotas",
    "axiom_pullback_step",
    "central_pullback_cov",
    "coalesce_pythagorean_ltv",
    "has_positive_prior_sigma",
    "should_use_axiom_pullback",
    "validate_axiom_prior_state",
    "whitened_burg_pullback_loss",
]
