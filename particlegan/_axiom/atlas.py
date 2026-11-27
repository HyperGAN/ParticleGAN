"""Axiom L1: PrototypeAtlas and GPU HamiltonCommitmentLedger."""

import math
from typing import NamedTuple

import torch
from torch import (
    Tensor,
    arange,
    bincount,
    cat,
    eye,
    full,
    isfinite,
    maximum,
    minimum,
    no_grad,
    ones,
    quantile,
    stack,
    where,
    zeros,
    zeros_like,
)

from .math_ops import (
    PHI1,
    hamilton_quotas,
    power_principal_axis,
    safe_cdist,
    smoothstep01,
)

SCHEMA_VERSION = 2
MAX_FULL_COV_DIM = 64
MAX_LEDGER_PARTICLES = 2048

_VAR_FLOOR = 1e-24
_EPS = 1e-12
_BETA_N = 0.04
_PROPOSAL_ROUNDS = 8
_INF = float("inf")

_LEDGER_HEADER_KEYS = ("initialized", "num_particles", "dim")
_LEDGER_TENSOR_KEYS = ("row_comp", "row_target", "row_dist")
_LEDGER_KEYS = frozenset({"schema", *_LEDGER_HEADER_KEYS, *_LEDGER_TENSOR_KEYS})

_ATLAS_TENSOR_KEYS = ("proto_c", "proto_cov", "proto_w", "w_fake_ema", "proto_n", "psi", "v_push")
_ATLAS_ALL_TENSORS = (*_ATLAS_TENSOR_KEYS, "_rho_atom_t", "_g_conn_t")
_ATLAS_KEYS = frozenset(
    {
        "schema",
        "max_prototypes",
        "beta_n",
        "initialized",
        "dim",
        "rho_atom_ema",
        "g_conn_ema",
        *_ATLAS_TENSOR_KEYS,
        "ledger",
    }
)


class CoalescedAtlas(NamedTuple):
    proto_comp_id: Tensor
    proto_c_bar: Tensor
    proto_cov_bar: Tensor
    proto_var_bar: Tensor
    w_mat: Tensor
    multicell: Tensor
    isolated: Tensor
    comp_c: Tensor
    comp_cov: Tensor
    comp_cov_tgt: Tensor
    comp_var: Tensor
    comp_w: Tensor
    comp_n_eff: Tensor
    comp_isolated: Tensor
    all_iso: Tensor


class LedgerAssignment(NamedTuple):
    batch_comp: Tensor
    batch_target: Tensor
    nn_comp_dist: Tensor
    quotas: Tensor
    occupancy: Tensor
    tail_residual: Tensor
    row_comp: Tensor


def _require(cond):
    if not cond:
        raise ValueError("invalid state")


def _check_schema(state, expected_keys):
    _require(
        isinstance(state, dict)
        and set(state) == expected_keys
        and type(state["schema"]) is int
        and state["schema"] == SCHEMA_VERSION
    )


def _check_scalar(val, lo=None, hi=None, strict=False):
    _require(type(val) in (int, float) and math.isfinite(float(val)))
    x = float(val)
    _require((lo is None or (x > lo if strict else x >= lo)) and (hi is None or (x < hi if strict else x <= hi)))
    return x


def _check_tensor(tensor, shape=None, nonneg=False, dtype=None):
    _require(
        isinstance(tensor, Tensor) and (tensor.dtype == dtype if dtype is not None else tensor.is_floating_point())
    )
    _require(shape is None or tuple(tensor.shape) == shape)
    if dtype is None:
        _require(bool(isfinite(tensor).all()) and (not nonneg or not bool((tensor < 0.0).any())))
    return tensor


def _copy_cpu(tensor):
    return None if tensor is None else tensor.detach().cpu().clone()


def _move_tensor(tensor, device, dtype=torch.float32):
    return None if tensor is None else tensor.detach().to(device, dtype).clone()


def _diag(mat):
    return mat.diagonal(dim1=-2, dim2=-1)


def _flatten_batch(y):
    return y.detach().reshape(len(y), -1).float()


def _normalize(weights, dim=None):
    if dim is None:
        return weights / weights.sum().clamp_min(_EPS)
    return weights / weights.sum(dim, keepdim=True).clamp_min(_EPS)


def _broadcast_slots(scale, is_full):
    return scale.view(-1, 1, 1) if is_full else scale.unsqueeze(1)


def _dedup_mask(points, tol):
    return (safe_cdist(points) <= tol).float().argmax(1) == arange(len(points), device=points.device)


def _slot_occupancy(row_comp, num_comp):
    return bincount(where(row_comp >= 0, row_comp, num_comp), minlength=num_comp + 1)[:num_comp]


def _sanitize_cov(cov, scale=None):
    if scale is not None:
        cov = cov * _broadcast_slots(scale, cov.ndim == 3)
    if cov.ndim == 3:
        sym = 0.5 * (cov + cov.mT)
        return sym + torch.diag_embed((_VAR_FLOOR - _diag(sym)).clamp_min(0.0))
    return cov.clamp_min(_VAR_FLOOR)


def cov_trace_mean(cov):
    """Mean diagonal variance per slot for full SPD or diagonal covariance."""
    return (_diag(cov) if cov.ndim == 3 else cov).mean(-1).clamp_min(_VAR_FLOOR)


def cov_from_scalar(var, dim):
    """Construct isotropic covariance tensor matching atlas storage for ``dim``."""
    v = var.float().clamp_min(_VAR_FLOOR)
    if dim <= MAX_FULL_COV_DIM:
        return v.view(-1, 1, 1) * eye(dim, device=v.device)
    return v.unsqueeze(-1).expand(len(v), dim).clone()


def cov_directional_var(cov, dirs):
    """Directional variance ``u^T Sigma u`` for full SPD or diagonal covariance."""
    if cov.ndim == 3:
        eq = "kmd,kde,kme->km" if dirs.ndim == 3 else "kd,kde,ke->k"
        return torch.einsum(eq, dirs, cov, dirs).clamp_min(_VAR_FLOOR)
    base = cov.unsqueeze(1) if dirs.ndim == 3 else cov
    return (dirs.square() * base).sum(-1).clamp_min(_VAR_FLOOR)


def cov_add_outer(cov, delta, scale=1.0):
    """Add ``scale * delta delta^T`` (or diagonal ``scale * delta^2``) to covariance."""
    if cov.ndim == 3:
        fac = scale.view(*scale.shape, 1, 1) if isinstance(scale, Tensor) else float(scale)
        out = cov + (delta.unsqueeze(-1) * delta.unsqueeze(-2)) * fac
        return 0.5 * (out + out.mT)
    fac = scale.unsqueeze(-1) if isinstance(scale, Tensor) else float(scale)
    return (cov + delta.square() * fac).clamp_min(_VAR_FLOOR)


def _scatter_cov(diff, idx, num_slots, is_full):
    dim = diff.shape[1]
    if is_full:
        return diff.new_zeros((num_slots, dim, dim)).index_add_(0, idx, diff.unsqueeze(2) * diff.unsqueeze(1))
    return diff.new_zeros((num_slots, dim)).index_add_(0, idx, diff.square())


def cov_principal_split(cov_k, seed=0):
    """Exact LTC principal-axis split via 6-step Weyl-seeded power iteration."""
    x, dtype = cov_k.detach().double(), cov_k.dtype
    if x.ndim == 2:
        sym = 0.5 * (x + x.T)
        lam, vec = power_principal_axis(sym, iters=6, seed=seed)
        s_vec = sym @ vec
        rayleigh = (vec * s_vec).sum()
        offset = where(rayleigh > _VAR_FLOOR, 0.40 * s_vec * rayleigh.clamp_min(_VAR_FLOOR).rsqrt(), 0.0)
        child_cov = _sanitize_cov((sym - offset.unsqueeze(1) * offset.unsqueeze(0)).unsqueeze(0))[0]
        return lam.clamp_min(0.0).to(dtype), offset.to(dtype), child_cov.to(dtype)
    diag = x.clamp_min(0.0)
    lam, max_idx = diag.max(-1)
    offset, child_cov = diag.new_zeros(diag.shape), diag.clone()
    offset[max_idx] = where(lam > _VAR_FLOOR, 0.40 * lam.clamp_min(_VAR_FLOOR).sqrt(), 0.0)
    child_cov[max_idx] = where(lam > _VAR_FLOOR, 0.84 * lam, diag[max_idx])
    return lam.to(dtype), offset.to(dtype), child_cov.clamp_min(_VAR_FLOOR).to(dtype)


def cov_shrink_mp(cov, n_eff):
    """Marchenko-Pastur + Wishart-excess SPD shrinkage toward isotropic variance."""
    dim = cov.shape[-1]
    iso_var = cov_trace_mean(cov)
    n_cov = (n_eff.float() / 3.0).clamp_min(1.0)
    gam = dim / n_cov
    mix_mp = smoothstep01((gam - 0.25) / 0.75)
    e_iso = (dim + 1.0) / n_cov
    if cov.ndim == 3:
        e_obs = ((cov / iso_var.view(-1, 1, 1)).square().sum((-2, -1)) / dim - 1.0).clamp_min(0.0)
    else:
        e_obs = ((cov / iso_var.unsqueeze(-1)).square().mean(-1) - 1.0).clamp_min(0.0)
    mix_wish = smoothstep01((2.2 * e_iso - e_obs) / (1.2 * e_iso).clamp_min(1e-6)) * smoothstep01((gam - 0.05) / 0.20)
    mix = maximum(mix_mp, mix_wish).clamp(1e-4, 1.0)
    if cov.ndim == 3:
        return (1.0 - mix.view(-1, 1, 1)) * cov + (mix * iso_var).view(-1, 1, 1) * eye(dim, device=cov.device)
    return ((1.0 - mix.unsqueeze(-1)) * cov + (mix * iso_var).unsqueeze(-1)).clamp_min(_VAR_FLOOR)


@no_grad()
def batch_annular_gap_ratio(y_real):
    """Continuous sorted-distance max log-gap ratio in ``[0, 1]``."""
    x = _flatten_batch(y_real)
    batch_n = len(x)
    if batch_n < 4:
        return x.new_ones(())
    if batch_n > 256:
        x, batch_n = x[(arange(256, device=x.device) * batch_n) // 256], 256
    tri_i, tri_j = torch.triu_indices(batch_n, batch_n, offset=1, device=x.device)
    dists = safe_cdist(x)[tri_i, tri_j].sort().values
    max_d = dists[-1]
    safe_d = dists.clamp_min(1e-7 * max_d.clamp_min(_VAR_FLOOR))
    half_n = max(2, dists.numel() // 2)
    max_gap, gap_idx = (safe_d[2 : half_n + 1].log() - safe_d[1:half_n].log()).max(0)
    blend = smoothstep01((max_gap - 0.50) / 0.70)
    ratio = (
        blend * (dists[(gap_idx + 1) // 2] / safe_d[gap_idx + 2].clamp_min(_VAR_FLOOR))
        + (1.0 - blend) * (-max_gap.clamp_min(0.0)).exp()
    )
    return where(max_d <= _VAR_FLOOR, 0.0, ratio.clamp(0.0, 1.0))


@no_grad()
def manifold_connectivity_stats(y_real):
    """Minimax ultrametric connectivity gate and scale statistics for ``y_real``."""
    x = _flatten_batch(y_real)
    batch_n = len(x)
    abs_max = x.abs().amax() if x.numel() > 0 else x.new_ones(())
    scale = where(abs_max < 1e-2, abs_max.clamp_min(_EPS), abs_max.clamp_min(1.0))
    if batch_n < 4:
        zero = x.new_zeros(())
        return zero, zero, x.new_full((), 0.1), scale
    y_sub = (x / scale)[:: max(1, batch_n // 256)] if batch_n > 512 else (x / scale)
    dist = safe_cdist(y_sub)
    off_diag = ~eye(len(y_sub), dtype=torch.bool, device=y_sub.device)
    nn_q90 = quantile(dist.masked_fill(~off_diag, _INF).amin(-1), 0.90).clamp_min(1e-8)
    diam_ratio = quantile(dist[off_diag], 0.90) / nn_q90
    ultra = dist.clone()
    for _ in range(8):
        ultra = minimum(ultra, maximum(ultra.unsqueeze(2), ultra.unsqueeze(0)).amin(1))
    g_conn = smoothstep01((10.0 - ultra.amax() / nn_q90) / 3.0) * smoothstep01((diam_ratio - 1.15) / 0.25)
    return g_conn, g_conn * smoothstep01((diam_ratio - 3.80) / 1.00), nn_q90.clamp_min(1e-4), scale


@no_grad()
def compute_rho_atom(comp_c, comp_cov, comp_n_eff, y_real=None):
    """Continuous annular-gap ratio blended with warmup batch ratio."""
    num_comp, dim = comp_c.shape
    rho_batch = batch_annular_gap_ratio(y_real) if y_real is not None else comp_c.new_ones(())
    if num_comp < 2:
        conf = smoothstep01((comp_n_eff.median() - 2.0) / 6.0) if num_comp == 1 else comp_c.new_zeros(())
        return maximum(rho_batch, conf).clamp(0.0, 1.0)
    nn_dist, nn_idx = safe_cdist(comp_c, diag_inf=True).min(1)
    unit_dir = (comp_c - comp_c[nn_idx]) / nn_dist.clamp_min(_EPS).unsqueeze(1)
    sigma_intra = (dim * cov_directional_var(comp_cov, unit_dir)).sqrt()
    rho_comp = where(nn_dist > _EPS, sigma_intra / nn_dist.clamp_min(_EPS), 1.0).median().clamp(0.0, 1.0)
    if y_real is None:
        return rho_comp
    weight = smoothstep01((comp_n_eff.median() - 2.0) / 6.0)
    return ((1.0 - weight) * rho_batch + weight * rho_comp).clamp(0.0, 1.0)


@no_grad()
def compute_connectivity_gate(y_real, *, rho_atom=None):
    """Continuous connectivity gate in ``[0, 1]``."""
    g_raw = manifold_connectivity_stats(y_real)[0]
    if rho_atom is None:
        return g_raw.clamp(0.0, 1.0)
    return (g_raw * smoothstep01((rho_atom - 0.15) / 0.20)).clamp(0.0, 1.0)


def _ltc_coalesce_pass(centers, cov, weights, adj, eye_k):
    num_proto = len(centers)
    reach = adj.clone().fill_diagonal_(1.0)
    for _ in range(min(8, max(4, (num_proto - 1).bit_length()))):
        reach = ((reach @ reach) > 0).float()
    comp_size = reach.sum(1)
    valid_comp = (comp_size >= 1.0) & (comp_size < num_proto)
    member_w = _normalize(where(valid_comp[:, None] & valid_comp[None, :], reach, eye_k) * weights.unsqueeze(0), 1)
    comp_centers = member_w @ centers
    total_cov = cov_add_outer(cov, centers - comp_centers)
    comp_cov = _sanitize_cov((member_w @ total_cov.reshape(num_proto, -1)).reshape(total_cov.shape))
    return reach, comp_size, valid_comp, member_w, comp_centers, comp_cov


@no_grad()
def coalesce_directional_ltc(proto_c, proto_cov, proto_w, proto_n=None):
    """Two-pass directional LTC coalescence with 1D Fisher inter-component isolation."""
    centers = proto_c.detach().float()
    num_proto, dim = centers.shape
    dev = centers.device
    weights = _normalize(proto_w.detach().float().clamp_min(_EPS))
    hits = ones(num_proto, device=dev) if proto_n is None else proto_n.detach().float().clamp_min(0.0)
    cov = _sanitize_cov(proto_cov.detach().float())
    eye_k = eye(num_proto, device=dev)
    dist = safe_cdist(centers)
    sqrt_d = math.sqrt(dim)
    unit_ij = (centers.unsqueeze(1) - centers.unsqueeze(0)) / dist.clamp_min(_EPS).unsqueeze(-1)

    sigma_dir = cov_directional_var(cov_shrink_mp(cov, hits), unit_ij).sqrt()
    sigma_iso = cov_trace_mean(cov).sqrt()
    scale_compat = maximum(sigma_iso[:, None], sigma_iso[None, :]) <= (
        2.2 * minimum(sigma_iso[:, None], sigma_iso[None, :]).clamp_min(_EPS)
    )
    adj_p1 = ((dist <= (2.6 * sqrt_d * maximum(sigma_dir, sigma_dir.T))) & scale_compat).float()
    reach, comp_size, valid_comp, _, _, cov_p1 = _ltc_coalesce_pass(centers, cov, weights, adj_p1, eye_k)

    halo_absorb = (
        (dist <= (2.6 * sqrt_d * cov_directional_var(cov_p1, unit_ij).sqrt().T))
        & ((comp_size >= 1.5) & valid_comp)[None, :]
        & scale_compat
    )
    adj_p2 = ((reach > 0.5) | halo_absorb | halo_absorb.T).float()
    reach, comp_size, valid_comp, w_mat, c_bar_raw, cov_p2 = _ltc_coalesce_pass(centers, cov, weights, adj_p2, eye_k)

    sigma_dir_p2 = cov_directional_var(cov_p2, unit_ij).sqrt()
    q_intra = dist / (4.2 * sqrt_d * sigma_dir_p2).clamp_min(_EPS)
    q_inter = dist / (6.0 * sigma_dir_p2).clamp_min(_EPS)
    intra_ok = (reach @ (~((q_intra * reach).amax(1) <= 1.0)).float()) < 0.5
    inter_ok = (reach @ (~(q_inter.masked_fill(reach > 0.5, _INF).amin(1) >= 1.0)).float()) < 0.5
    isolated = valid_comp & intra_ok & inter_ok
    use_coal = isolated & ((weights * isolated.float()).sum() >= 0.70)

    c_bar = where(use_coal.unsqueeze(1), c_bar_raw, centers)
    cov_bar = where(_broadcast_slots(use_coal, cov.ndim == 3), cov_p2, cov)
    eff_adj = where(use_coal[:, None] & use_coal[None, :], reach, eye_k)
    rep_id = (eff_adj > 0.5).float().argmax(1)
    is_rep = rep_id == arange(num_proto, device=dev)
    rep_idx = is_rep.nonzero().squeeze(-1)

    comp_cov = cov_bar[rep_idx]
    comp_w = _normalize(eff_adj[rep_idx] @ weights)
    comp_n_eff = eff_adj[rep_idx] @ hits
    comp_iso = isolated[rep_idx]
    comp_cov_tgt = where(
        _broadcast_slots(comp_iso, comp_cov.ndim == 3),
        cov_shrink_mp(comp_cov, comp_n_eff),
        cov_from_scalar(cov_trace_mean(comp_cov), dim),
    )
    return CoalescedAtlas(
        proto_comp_id=(is_rep.long().cumsum(0) - 1)[rep_id],
        proto_c_bar=c_bar,
        proto_cov_bar=cov_bar,
        proto_var_bar=cov_trace_mean(cov_bar),
        w_mat=where(use_coal.unsqueeze(1), w_mat, eye_k),
        multicell=(comp_size > 1.5) & use_coal,
        isolated=isolated,
        comp_c=c_bar[rep_idx],
        comp_cov=comp_cov,
        comp_cov_tgt=comp_cov_tgt,
        comp_var=cov_trace_mean(comp_cov),
        comp_w=comp_w,
        comp_n_eff=comp_n_eff,
        comp_isolated=comp_iso,
        all_iso=comp_iso.all() if rep_idx.numel() >= 2 else zeros((), device=dev, dtype=torch.bool),
    )


@no_grad()
def coalesce_pythagorean_ltv(proto_c, proto_var, proto_w):
    """Isotropic LTV wrapper delegating to ``coalesce_directional_ltc``."""
    coal = coalesce_directional_ltc(proto_c, cov_from_scalar(proto_var, proto_c.shape[1]), proto_w)
    return coal.proto_c_bar, coal.proto_var_bar, coal.w_mat, coal.multicell


def validate_ledger_state(state):
    """Fail-closed schema-2 validator for ``HamiltonCommitmentLedger`` checkpoints."""
    _check_schema(state, _LEDGER_KEYS)
    init, num_p, dim = state["initialized"], state["num_particles"], state["dim"]
    _require(type(init) is bool and type(num_p) is int and type(dim) is int)
    if not init:
        _require(num_p == 0 and dim == 0 and all(state[k] is None for k in _LEDGER_TENSOR_KEYS))
        return state
    _require(
        1 <= num_p <= MAX_LEDGER_PARTICLES
        and dim >= 1
        and not bool((_check_tensor(state["row_comp"], (num_p,), dtype=torch.int64) < -1).any())
    )
    _check_tensor(state["row_target"], (num_p, dim))
    _check_tensor(state["row_dist"], (num_p,), nonneg=True)
    return state


def _validate_atlas_tensors(state, num_proto, dim):
    is_full = dim <= MAX_FULL_COV_DIM
    cov = _check_tensor(state["proto_cov"], (num_proto, dim, dim) if is_full else (num_proto, dim), not is_full)
    if is_full:
        sym_err = float((cov - cov.mT).abs().amax())
        _require(sym_err <= 1e-4 * float(cov.abs().amax().clamp_min(1.0)) and not bool((_diag(cov) < 0.0).any()))
    for name in ("proto_w", "w_fake_ema"):
        _require(abs(float(_check_tensor(state[name], (num_proto,), nonneg=True).double().sum()) - 1.0) <= 1e-4)
    for name, nonneg in (("proto_n", True), ("v_push", True), ("psi", False)):
        _check_tensor(state[name], (num_proto,), nonneg=nonneg)


def validate_atlas_state(state):
    """Fail-closed schema-2 validator for ``PrototypeAtlas`` checkpoints."""
    _check_schema(state, _ATLAS_KEYS)
    max_p, init, dim = state["max_prototypes"], state["initialized"], state["dim"]
    _require(type(max_p) is int and max_p >= 2 and type(init) is bool and type(dim) is int)
    for name, lo, hi, strict in (
        ("beta_n", 0.0, 1.0, True),
        ("rho_atom_ema", 0.0, None, False),
        ("g_conn_ema", 0.0, 1.0, False),
    ):
        _check_scalar(state[name], lo, hi, strict)
    ledger = state["ledger"]
    if ledger is not None:
        validate_ledger_state(ledger)
    if not init:
        _require(dim == 0 and all(state[k] is None for k in _ATLAS_TENSOR_KEYS))
        return state
    centers = _check_tensor(state["proto_c"])
    _require(dim >= 1 and centers.ndim == 2 and centers.shape[1] == dim and 1 <= len(centers) <= max_p)
    _validate_atlas_tensors(state, len(centers), dim)
    _require(ledger is None or not ledger["initialized"] or ledger["dim"] == dim)
    return state


def _group_rank(group_id, dist, weyl, num_comp):
    order = (dist.clamp_max(1e6) + 1e-6 * weyl).argsort(stable=True)
    order = order[group_id[order].argsort(stable=True)]
    sorted_g = group_id[order]
    counts = bincount(sorted_g, minlength=num_comp + 1)
    rank = arange(group_id.numel(), device=group_id.device, dtype=torch.int64) - (counts.cumsum(0) - counts)[sorted_g]
    return order, sorted_g, rank


class HamiltonCommitmentLedger:
    """GPU-vectorized Hamilton commitment ledger with Schmitt-trigger hysteresis."""

    SCHEMA = SCHEMA_VERSION

    def __init__(self):
        self.reset()

    def reset(self):
        self.initialized, self.num_particles, self.dim = False, 0, 0
        self.row_comp = self.row_target = self.row_dist = None

    def state_dict(self):
        return {
            "schema": SCHEMA_VERSION,
            **{k: getattr(self, k) for k in _LEDGER_HEADER_KEYS},
            **{k: _copy_cpu(getattr(self, k)) for k in _LEDGER_TENSOR_KEYS},
        }

    def load_state_dict(self, state, *, device=None, validate_only=False):
        val = validate_ledger_state(state)
        if validate_only:
            return
        for key in _LEDGER_HEADER_KEYS:
            setattr(self, key, val[key])
        if not self.initialized:
            self.row_comp = self.row_target = self.row_dist = None
            return
        dst_dev = device if device is not None else val["row_target"].device
        for key, dtype in zip(_LEDGER_TENSOR_KEYS, (torch.int64, torch.float32, torch.float32)):
            setattr(self, key, _move_tensor(val[key], dst_dev, dtype))

    def _assign_deficit(self, row_comp, row_target, row_dist, dist_pc, comp_c, in_batch, quotas, weyl):
        num_comp, num_p, dev = len(comp_c), row_comp.numel(), row_comp.device
        deficit = (quotas - _slot_occupancy(row_comp, num_comp)).clamp_min(0)
        unassigned = in_batch & (row_comp < 0)
        for _ in range(_PROPOSAL_ROUNDS):
            prop_dist, prop_comp = dist_pc.masked_fill((deficit <= 0).unsqueeze(0), _INF).min(1)
            group_key = where(unassigned & isfinite(prop_dist), prop_comp, num_comp)
            order, sorted_g, rank = _group_rank(group_key, prop_dist, weyl, num_comp)
            keep = (sorted_g < num_comp) & (rank < deficit[sorted_g.clamp_max(num_comp - 1)])
            accept = zeros_like(unassigned).scatter_(0, order, keep)
            row_comp = where(accept, prop_comp, row_comp)
            row_target = where(accept.unsqueeze(1), comp_c[prop_comp], row_target)
            row_dist = where(accept, prop_dist, row_dist)
            unassigned = unassigned & (~accept)
            deficit = (deficit - _slot_occupancy(where(accept, prop_comp, -1), num_comp)).clamp_min(0)
        rem_min = dist_pc.masked_fill((deficit <= 0).unsqueeze(0), _INF).amin(1)
        comp_order = dist_pc.masked_fill(~unassigned.unsqueeze(1), _INF).amin(0).clamp_max(1e6).argsort(stable=True)
        slot_ends = deficit[comp_order].cumsum(0)
        ranks = arange(num_p, device=dev, dtype=torch.int64)
        slot_comp = comp_order[torch.searchsorted(slot_ends, ranks, right=True).clamp_max(num_comp - 1)]
        row_order = where(unassigned, rem_min.clamp_max(1e6) + 1e-6 * weyl, 1e12).argsort(stable=True)
        fallback_id = full((num_p,), -1, dtype=torch.int64, device=dev).scatter_(
            0, row_order, where(unassigned[row_order] & (ranks < slot_ends[-1]), slot_comp, -1)
        )
        matched = fallback_id >= 0
        safe_id = fallback_id.clamp_min(0)
        return (
            where(matched, safe_id, row_comp),
            where(matched.unsqueeze(1), comp_c[safe_id], row_target),
            where(matched, dist_pc.gather(1, safe_id.unsqueeze(1)).squeeze(1), row_dist),
        )

    @no_grad()
    def update(self, x_fake, rows, comp_c, comp_w, num_particles, *, allow_update=True):
        batch_n, num_comp, num_p, dev = len(x_fake), len(comp_c), int(num_particles), x_fake.device
        _require(rows.ndim == 1 and len(rows) == batch_n and num_comp >= 1 and 1 <= num_p <= MAX_LEDGER_PARTICLES)
        centers = comp_c.detach().to(dev, torch.float32).reshape(num_comp, -1)
        dim = centers.shape[1]
        x = x_fake.detach().to(dev, torch.float32).reshape(batch_n, dim)
        if self.initialized and self.num_particles == num_p and self.dim == dim and self.row_comp is not None:
            row_comp, row_target, row_dist = self.row_comp.to(dev), self.row_target.to(dev), self.row_dist.to(dev)
        else:
            self.initialized, self.num_particles, self.dim = True, num_p, dim
            row_comp = full((num_p,), -1, dtype=torch.int64, device=dev)
            row_target = zeros((num_p, dim), device=dev)
            row_dist = full((num_p,), 1e30, device=dev)
        row_idx = rows.detach().to(dev, torch.int64).clamp(0, num_p - 1)
        dist_xc = safe_cdist(x, centers)
        w_norm = _normalize(comp_w.detach().to(dev, torch.float32).clamp_min(0.0))
        quotas = hamilton_quotas(num_p, w_norm, active_scores=-dist_xc.amin(0))
        nn_dist = (
            safe_cdist(centers, diag_inf=True).amin(1).clamp_min(_EPS) if num_comp >= 2 else ones(num_comp, device=dev)
        )
        if allow_update and batch_n > 0:
            in_batch = zeros(num_p, dtype=torch.bool, device=dev)
            in_batch[row_idx] = True
            first_pos = full((num_p,), batch_n, dtype=torch.int64, device=dev).scatter_reduce_(
                0, row_idx, arange(batch_n, device=dev, dtype=torch.int64), reduce="amin"
            )
            dist_pc = dist_xc[first_pos.clamp_max(batch_n - 1)]
            weyl = ((arange(num_p, device=dev, dtype=torch.float32) + 0.5) * PHI1).remainder(1.0)
            active = row_comp >= 0
            nearest_c = safe_cdist(row_target, centers).argmin(1)
            matched_c = centers[nearest_c]
            row_comp = where(active, nearest_c, row_comp)
            row_dist = where(
                in_batch & (row_comp >= 0),
                dist_pc.gather(1, row_comp.clamp_min(0).unsqueeze(1)).squeeze(1),
                where(active, maximum(row_dist, (row_target - matched_c).norm(dim=1)), row_dist),
            )
            row_target = where(active.unsqueeze(1), matched_c, row_target)

            occ_now = _slot_occupancy(row_comp, num_comp)
            any_empty = ((quotas >= 1) & (occ_now == 0)).any()
            q_hyst = maximum(quotas, (num_p * w_norm + 0.80).floor().long())
            q_settled = where(any_empty, quotas, q_hyst)
            settled = (row_comp >= 0) & (row_dist <= 0.15 * nn_dist[row_comp.clamp_min(0)])
            order, sorted_g, rank = _group_rank(where(row_comp >= 0, row_comp, num_comp), row_dist, weyl, num_comp)
            safe_g = sorted_g.clamp_max(num_comp - 1)
            evict_bound = where(settled[order], q_settled[safe_g], quotas[safe_g])
            evict = zeros_like(settled).scatter_(0, order, (sorted_g < num_comp) & (rank >= evict_bound))
            row_comp, row_target, row_dist = self._assign_deficit(
                where(evict, -1, row_comp),
                row_target,
                where(evict, 1e30, row_dist),
                dist_pc,
                centers,
                in_batch,
                quotas,
                weyl,
            )
        self.row_comp, self.row_target, self.row_dist = row_comp, row_target, row_dist
        batch_comp = where(row_comp[row_idx] >= 0, row_comp[row_idx].clamp_min(0), dist_xc.argmin(1))
        occupancy = _slot_occupancy(row_comp, num_comp)
        unsettled = num_p - ((row_comp >= 0) & (row_dist <= 0.09 * nn_dist[row_comp.clamp_min(0)])).float().sum()
        commit_pen = where(
            (unsettled > 0.5) | (occupancy < quotas).any(),
            0.18 + 0.12 * (unsettled.clamp_min(1.0) / max(1, num_p)),
            0.0,
        )
        tail_res = maximum(
            1.6 * ((x - centers[batch_comp]).norm(dim=1) / nn_dist[batch_comp]).clamp_max(10.0).mean(), commit_pen
        )
        return LedgerAssignment(batch_comp, centers[batch_comp], nn_dist, quotas, occupancy, tail_res, row_comp)


class PrototypeAtlas:
    """Clock-free streaming prototype atlas with full/diagonal covariance and GPU ledger."""

    SCHEMA = SCHEMA_VERSION

    @property
    def proto_var(self):
        return None if self.proto_cov is None else cov_trace_mean(self.proto_cov)

    @property
    def n_eff(self):
        return self.proto_n

    @property
    def num_prototypes(self):
        return 0 if self.proto_c is None else len(self.proto_c)

    def __init__(self, max_prototypes=256, *, beta_n=_BETA_N, device=None):
        _require(type(max_prototypes) is int and max_prototypes >= 2)
        self.beta_n = _check_scalar(beta_n, 0.0, 1.0, strict=True)
        self.max_prototypes, self.device = max_prototypes, device
        self.ledger = HamiltonCommitmentLedger()
        self.reset()

    def _flush_scalars(self):
        if self._rho_atom_t is not None and self._g_conn_t is not None:
            vals = stack((self._rho_atom_t.double(), self._g_conn_t.double())).tolist()
            self._rho_atom_f, self._g_conn_f = float(vals[0]), float(vals[1])
            self._rho_atom_t = self._g_conn_t = None

    @property
    def rho_atom_ema(self):
        self._flush_scalars()
        return self._rho_atom_f

    @property
    def g_conn_ema(self):
        self._flush_scalars()
        return self._g_conn_f

    def reset(self):
        self.initialized, self.dim, self._last_point_mass = False, 0, False
        for attr in _ATLAS_ALL_TENSORS:
            setattr(self, attr, None)
        self._rho_atom_f, self._g_conn_f = 1.0, 0.0
        self.ledger.reset()

    def state_dict(self):
        self._flush_scalars()
        return {
            "schema": SCHEMA_VERSION,
            "max_prototypes": self.max_prototypes,
            "beta_n": self.beta_n,
            "initialized": self.initialized,
            "dim": self.dim,
            "rho_atom_ema": self._rho_atom_f,
            "g_conn_ema": self._g_conn_f,
            **{k: _copy_cpu(getattr(self, k)) for k in _ATLAS_TENSOR_KEYS},
            "ledger": self.ledger.state_dict() if self.ledger.initialized else None,
        }

    def load_state_dict(self, state, *, device=None, validate_only=False):
        val = validate_atlas_state(state)
        _require(val["max_prototypes"] == self.max_prototypes)
        if validate_only:
            return
        dst_dev = device if device is not None else self.device
        self.beta_n, self.initialized, self.dim = float(val["beta_n"]), val["initialized"], val["dim"]
        self._rho_atom_f, self._g_conn_f = float(val["rho_atom_ema"]), float(val["g_conn_ema"])
        self._last_point_mass, self._rho_atom_t, self._g_conn_t = False, None, None
        for key in _ATLAS_TENSOR_KEYS:
            setattr(self, key, _move_tensor(val[key], dst_dev))
        if val["ledger"] is None:
            self.ledger.reset()
        else:
            self.ledger.load_state_dict(val["ledger"], device=dst_dev)

    def _slots(self):
        return self.proto_c, self.proto_cov, self.proto_w, self.w_fake_ema, self.proto_n, self.psi, self.v_push

    def _append_slots(self, new_c, new_var, new_w, new_n, keep_scale=1.0):
        pc, cov, pw, wf, pn, psi, vp = self._slots()
        self.proto_c = cat((pc, new_c))
        self.proto_cov = cat((cov, cov_from_scalar(new_var, self.dim)))
        self.v_push = cat((vp, new_var))
        self.proto_n = cat((pn, new_n))
        self.proto_w = cat((pw * keep_scale, new_w))
        self.w_fake_ema = cat((wf * keep_scale, new_w))
        psi_all = cat((psi, new_c.new_zeros(len(new_c))))
        self.psi = psi_all - psi_all.mean()

    @no_grad()
    def merge_pair(self, slot_a, slot_b):
        """Exact Law-of-Total-Covariance merge of ``slot_b`` into ``slot_a``."""
        c, cov, w, wf, n, psi, v = self._slots()
        wa, wb = w[slot_a].double(), w[slot_b].double()
        w_tot = (wa + wb).clamp_min(_EPS)
        mean_c, mean_v, mean_psi = ((wa * t[slot_a].double() + wb * t[slot_b].double()) / w_tot for t in (c, v, psi))
        cov_a, cov_b = (
            cov_add_outer(cov[idx].double().unsqueeze(0), (c[idx].double() - mean_c).unsqueeze(0))[0]
            for idx in (slot_a, slot_b)
        )
        c[slot_a] = mean_c.float()
        cov[slot_a] = _sanitize_cov(((wa * cov_a + wb * cov_b) / w_tot).unsqueeze(0).float())[0]
        v[slot_a], psi[slot_a], psi[slot_b] = mean_v.clamp_min(_VAR_FLOOR).float(), mean_psi.float(), 0.0
        w[slot_a], w[slot_b] = w_tot.float(), 0.0
        wf[slot_a], wf[slot_b] = wf[slot_a] + wf[slot_b], 0.0
        n[slot_a], n[slot_b] = n[slot_a] + n[slot_b], 0.0

    @no_grad()
    def split_slot(self, parent, dst):
        """Exact LTC principal-axis split of ``parent`` into ``(parent, dst)``."""
        c, cov, w, wf, n, psi, v = self._slots()
        c_parent, cov_parent = c[parent].double(), cov[parent]
        _, delta, child_cov = cov_principal_split(cov_parent, seed=parent if isinstance(parent, int) else 0)
        c[parent], c[dst] = (c_parent - delta.double()).float(), (c_parent + delta.double()).float()
        cov[parent], cov[dst] = child_cov, child_cov.clone()
        var_ratio = cov_trace_mean(child_cov.unsqueeze(0))[0] / cov_trace_mean(cov_parent.unsqueeze(0))[0]
        v[parent] = v[dst] = (v[parent] * var_ratio).clamp_min(_VAR_FLOOR)
        n[parent] = n[dst] = 0.5 * n[parent]
        for tensor in (w, wf, psi):
            tensor[parent] = tensor[dst] = 0.5 * (tensor[parent] + tensor[dst])

    def _init_from_batch(self, x):
        batch_n, dim, dev = x.shape[0], x.shape[1], x.device
        var_x = x.var(0, unbiased=False)
        raw_sp = 2.0 * var_x.sum()
        is_pt = not bool(raw_sp > _VAR_FLOOR)
        if is_pt:
            centers = x[:1].clone()
        else:
            unique_x = x[_dedup_mask(x, 1e-5 * raw_sp.sqrt())]
            if len(unique_x) < batch_n and len(unique_x) <= self.max_prototypes:
                centers = unique_x.clone()
            else:
                pool = unique_x if len(unique_x) < batch_n else x
                pool_n = len(pool)
                num_k = min(
                    self.max_prototypes, pool_n if len(unique_x) < batch_n or pool_n < 8 else max(4, (3 * pool_n) // 4)
                )
                proj = (arange(1, dim + 1, device=dev, dtype=torch.float32) * PHI1).remainder(1.0) - 0.5
                order = (pool @ (proj / proj.norm().clamp_min(1e-8))).argsort(stable=True)
                pick = ((arange(num_k, device=dev) * pool_n + pool_n // 2) // num_k).clamp(0, pool_n - 1)
                centers = pool[order[pick]].clone()
        num_k = len(centers)
        nearest = safe_cdist(x, centers).argmin(1)
        counts = bincount(nearest, minlength=num_k).float()
        var_base = var_x.mean().clamp_min(_VAR_FLOOR) / max(1, num_k)
        if num_k >= 2:
            nn_dist = safe_cdist(centers, diag_inf=True).amin(1)
            pos = nn_dist > _EPS
            med_pos = where(pos, nn_dist, _INF).sort().values[(pos.sum() // 2).clamp_max(num_k - 1)]
            var_base = where(pos.sum() > 0, (0.04 * med_pos.square() / dim).clamp_min(_VAR_FLOOR), var_base)
        b_mean = x.new_zeros((num_k, dim)).index_add_(0, nearest, x) / counts.clamp_min(1.0).unsqueeze(1)
        var_emp = (
            x.new_zeros(num_k).index_add_(0, nearest, (x - b_mean[nearest]).square().mean(1))
            / (counts - 1.0).clamp_min(1.0)
        ).clamp_min(_VAR_FLOOR)
        v_init = (
            (quantile(var_emp[counts >= 2.0], 0.25) if bool((counts >= 2.0).any()) else var_base).expand(num_k).clone()
        )

        self.dim, self.initialized, self._last_point_mass = dim, True, is_pt
        self.proto_c, self.proto_cov = centers, cov_from_scalar(v_init, dim)
        self.proto_w = _normalize(counts.clamp_min(_EPS))
        self.w_fake_ema = self.proto_w.clone()
        self.proto_n = counts.clamp_min(1.0)
        self.psi = zeros(num_k, device=dev)
        self.v_push = v_init.clone()
        self.ledger.reset()
        rho0 = batch_annular_gap_ratio(x)
        self._rho_atom_t, self._g_conn_t = rho0, compute_connectivity_gate(x, rho_atom=rho0)

    def _stream_update(self, x):
        c, cov, pw, _, pn, _, v_push_prev = self._slots()
        c, cov = c.clone(), cov.clone()
        batch_n, dim, num_k, is_full = x.shape[0], x.shape[1], len(c), cov.ndim == 3
        var_cur = cov_trace_mean(cov)
        d_min, nearest = safe_cdist(x, c).min(1)
        d2, raw_sp = d_min.square(), 2.0 * x.var(0, unbiased=False).sum()
        has_sp, pos_sp = raw_sp > _VAR_FLOOR, raw_sp.clamp_min(_VAR_FLOOR)
        is_pt = (0.5 * raw_sp) < (1e-11 * x.square().sum(1).mean()).clamp_min(1e-22)

        shock = (
            (
                where(
                    has_sp,
                    (d2.mean() / maximum(pos_sp, 6.0 * dim * var_cur[nearest].mean().clamp_min(_VAR_FLOOR))).sqrt(),
                    0.0,
                )
                - 0.45
            )
            / 0.35
        ).clamp(0.0, 1.0)
        var_med = var_cur.median().clamp_min(_VAR_FLOOR)
        if num_k < self.max_prototypes:
            unit_dir = (x - c[nearest]) / d_min.clamp_min(_EPS).unsqueeze(1)
            uncovered = (
                has_sp
                & (shock <= 0.5)
                & (
                    d2
                    > maximum(
                        36.0 * dim * maximum(cov_directional_var(cov[nearest], unit_dir), var_med),
                        0.008 * pos_sp,
                    )
                )
            )
        else:
            uncovered = zeros(batch_n, device=x.device, dtype=torch.bool)

        valid_mask = (~uncovered).float()
        counts = x.new_zeros(num_k).index_add_(0, nearest, valid_mask)
        hit, hit2 = counts > 0.0, counts >= 2.0
        n_decayed = where(shock > 0.25, pn * (1.0 - 0.8 * shock), pn)
        self.proto_n = ((1.0 - self.beta_n) * n_decayed + counts).clamp_min(1e-6)

        alpha_w = 0.02 + 0.48 * shock * shock
        far_unhit = (shock > 0.25) & has_sp & (~hit) & ((c - x.mean(0, keepdim=True)).square().sum(1) > 100.0 * pos_sp)
        alpha_k = where(far_unhit, 0.50, alpha_w)
        self.proto_w = _normalize((1.0 - alpha_k) * pw + alpha_k * (counts / valid_mask.sum().clamp_min(1.0)))

        b_mean = x.new_zeros((num_k, dim)).index_add_(0, nearest, x * valid_mask.unsqueeze(1)) / counts.clamp_min(
            1.0
        ).unsqueeze(1)
        alpha_c = (
            (0.035 + 0.25 * self.proto_n.clamp_min(1.0).rsqrt() + 0.38 * shock * shock) * (counts / (counts + 2.0))
        ).clamp_max(0.50)
        self.proto_c = where(hit.unsqueeze(1), (1.0 - alpha_c.unsqueeze(1)) * c + alpha_c.unsqueeze(1) * b_mean, c)

        diff = (x - where(hit2.unsqueeze(1), b_mean, c)[nearest]) * valid_mask.unsqueeze(1)
        cov_emp = _sanitize_cov(
            _scatter_cov(diff, nearest, num_k, is_full), 1.0 / (counts - hit2.float()).clamp_min(1.0)
        )
        var_emp, var_tot = cov_trace_mean(cov_emp), (raw_sp / (2.0 * dim)).clamp_min(_VAR_FLOOR)
        var_collapse, log_v = has_sp & (batch_n >= 8) & (var_cur > 4.0 * var_tot), var_cur.log()
        var_geom = (
            (0.15 * log_v + 0.85 * (var_tot / max(1, num_k)).clamp_min(_VAR_FLOOR).log()).exp().clamp_min(_VAR_FLOOR)
        )
        cov_base = _sanitize_cov(cov, where(var_collapse, var_geom / var_cur, 1.0))

        var_jump = hit & (
            (shock > 0.5)
            | (hit2 & ((var_emp < 1e-8 * var_cur) | (var_emp > 25.0 * var_cur)))
            | (var_emp <= 1e-12 * var_cur)
            | ((counts >= 6.0) & (var_emp < 0.04 * var_cur))
        )
        log_ratio = var_emp.log() - log_v
        scale_fac = (
            (0.65 + 0.25 * maximum(shock * shock, smoothstep01((log_ratio.abs() - 3.2) / 3.2))).clamp_max(0.90)
            * log_ratio
        ).exp()
        alpha_cov = _broadcast_slots(0.12 * counts / (counts + 1.0), is_full)
        cov_hit = (1.0 - alpha_cov) * where(
            _broadcast_slots(var_jump, is_full), _sanitize_cov(cov, scale_fac), cov_base
        ) + alpha_cov * cov_emp
        self.proto_cov = _sanitize_cov(where(_broadcast_slots(hit, is_full), cov_hit, cov_base))
        self.v_push = where(
            var_jump, (var_cur * scale_fac).clamp_min(_VAR_FLOOR), where(var_collapse, var_geom, v_push_prev)
        )
        return nearest, d2, var_cur, has_sp, pos_sp, alpha_w, uncovered, is_pt

    def _rebalance_slots(self, dead, has_dead, parent_idx, med_var, gamma, mean_w):
        c, cov, w, _, proto_n, _, _ = self._slots()
        num_k, slot_ids = len(c), arange(len(c), device=c.device)
        if not has_dead:
            is_parent = slot_ids == parent_idx
            pair_d2, flat_idx = (
                safe_cdist(c, diag_inf=True)
                .square()
                .masked_fill(is_parent[:, None] | is_parent[None, :], _INF)
                .reshape(-1)
                .min(0)
            )
            if not bool(pair_d2 <= 8.0 * self.dim * med_var):
                return
            self.merge_pair(flat_idx // num_k, flat_idx % num_k)
            dst_slots = (flat_idx % num_k,)
        else:
            dst_slots = tuple(dead.nonzero().squeeze(-1)[:4])
        for dst in dst_slots:
            skip = (slot_ids == dst) | dead | (proto_n < 0.5)
            best_p = ((w.clamp_min(_EPS) ** gamma) * cov_trace_mean(cov)).masked_fill(skip, 0.0).argmax()
            if not bool((~skip[best_p]) & (w[best_p] > 0.20 * mean_w)):
                break
            self.split_slot(best_p, dst)

    def _spawn_slots(self, x, worst_idx, prev_med, cur_med, pos_sp, alpha_w, uncovered, mean_w, pre_flag, dead_flag):
        c, cov, w, wf, n, psi, v = self._slots()
        num_k, dim = len(c), self.dim
        if pre_flag:
            uncov_x = x[uncovered]
            tol = maximum(36.0 * dim * prev_med, 0.008 * pos_sp).sqrt()
            new_c = uncov_x[_dedup_mask(uncov_x, tol)][: min(self.max_prototypes - num_k, 64)]
            if len(new_c) > 0:
                w_slot = min(0.5 / len(new_c), 1.0 / (num_k + len(new_c)))
                new_n = bincount(safe_cdist(uncov_x, new_c).argmin(1), minlength=len(new_c)).float().clamp_min(1.0)
                self._append_slots(
                    new_c,
                    prev_med.expand(len(new_c)),
                    new_c.new_full((len(new_c),), w_slot),
                    new_n,
                    1.0 - len(new_c) * w_slot,
                )
            return
        new_w = (alpha_w / len(x)).clamp_min(0.05 * mean_w)
        if not dead_flag and num_k < self.max_prototypes:
            self._append_slots(
                x[worst_idx].unsqueeze(0), cur_med.reshape(1), new_w.reshape(1), ones(1, device=c.device)
            )
            return
        slot = w.argmin() if dead_flag else safe_cdist(c, diag_inf=True).square().argmin()
        if not dead_flag:
            self.merge_pair(slot // num_k, slot % num_k)
            slot = slot % num_k
        c[slot], cov[slot], v[slot] = x[worst_idx], cov_from_scalar(cur_med.reshape(1), dim)[0], cur_med
        w[slot], n[slot], wf[slot], psi[slot] = new_w, 1.0, new_w, 0.0

    def _maintain(self, x, nearest, d2, var_prev, has_sp, pos_sp, alpha_w, uncovered, is_pt):
        c, cov, w, _, proto_n, _, _ = self._slots()
        num_k, dim = len(c), self.dim
        mean_w, var_cur = 1.0 / num_k, cov_trace_mean(cov)
        prev_med, cur_med = var_prev.median().clamp_min(_VAR_FLOOR), var_cur.median().clamp_min(_VAR_FLOOR)
        dead = (w < (0.04 / num_k)) | (proto_n < 0.05)
        gamma = min(dim, 4) / (min(dim, 4) + 2)
        max_uncov_ratio, worst_idx = (
            d2 / maximum(36.0 * dim * maximum(var_prev[nearest], prev_med), 0.008 * pos_sp)
        ).max(0)
        energy = (w.clamp_min(_EPS) ** gamma) * var_cur
        max_energy, parent_idx = energy.max(0)

        do_uncov, any_dead = (max_uncov_ratio > 1.0) & has_sp, dead.any()
        need_geo = (
            (max_energy > 10.0 * energy.median().clamp_min(_VAR_FLOOR)) & (w[parent_idx] > 0.25 * mean_w)
            if num_k >= 16
            else zeros((), device=x.device, dtype=torch.bool)
        )
        do_split = (~do_uncov) & has_sp & (any_dead | need_geo)
        pt_flag, pre_flag, uncov_flag, split_flag, dead_flag = stack(
            (is_pt, uncovered.any(), do_uncov, do_split, any_dead)
        ).tolist()
        self._last_point_mass = bool(pt_flag)
        if pre_flag or uncov_flag:
            self._spawn_slots(
                x, worst_idx, prev_med, cur_med, pos_sp, alpha_w, uncovered, mean_w, bool(pre_flag), bool(dead_flag)
            )
        elif split_flag:
            self._rebalance_slots(dead, bool(dead_flag), parent_idx, prev_med, gamma, mean_w)
        if pre_flag or uncov_flag or split_flag:
            _, _, w2, wf2, _, psi2, _ = self._slots()
            self.proto_w, self.w_fake_ema, self.psi = _normalize(w2), _normalize(wf2), psi2 - psi2.mean()

    @no_grad()
    def update(self, y_real):
        """Streaming prototype, hit-confidence, covariance, and maintenance update."""
        _require(y_real.ndim >= 2 and len(y_real) >= 1)
        x = _flatten_batch(y_real)
        _require(x.shape[1] >= 1)
        self.device = x.device
        for attr in _ATLAS_ALL_TENSORS:
            tensor = getattr(self, attr)
            if isinstance(tensor, Tensor) and tensor.device != x.device:
                setattr(self, attr, tensor.to(x.device))
        if not self.initialized or self.proto_c is None or self.dim != x.shape[1]:
            self._init_from_batch(x)
            return
        self._maintain(x, *self._stream_update(x))

    @no_grad()
    def coalesce(self, y_real=None):
        """Coalesce prototypes via directional LTC and update ``rho_atom`` / ``g_conn`` EMAs."""
        if not self.initialized or self.proto_c is None:
            raise RuntimeError("atlas not initialized")
        coal = coalesce_directional_ltc(self.proto_c, self.proto_cov, self.proto_w, self.proto_n)
        rho_cur = compute_rho_atom(coal.comp_c, coal.comp_cov, coal.comp_n_eff, y_real=y_real)
        rho_prev = self._rho_atom_t if self._rho_atom_t is not None else rho_cur.new_full((), self._rho_atom_f)
        self._rho_atom_t = 0.85 * rho_prev + 0.15 * rho_cur
        if y_real is not None:
            g_cur = compute_connectivity_gate(y_real, rho_atom=self._rho_atom_t)
            g_prev = self._g_conn_t if self._g_conn_t is not None else g_cur.new_full((), self._g_conn_f)
            self._g_conn_t = 0.85 * g_prev + 0.15 * g_cur
        return coal

    @no_grad()
    def commit_particles(self, x_fake, rows, num_particles, *, coalesced=None, allow_update=True):
        """Commit particle rows to isolated components on GPU when ``all_iso`` and ``K_C >= 2``."""
        if not self.initialized or not (1 <= int(num_particles) <= MAX_LEDGER_PARTICLES):
            return None
        coal = self.coalesce() if coalesced is None else coalesced
        if len(coal.comp_c) < 2 or not bool(coal.all_iso):
            return None
        return self.ledger.update(x_fake, rows, coal.comp_c, coal.comp_w, int(num_particles), allow_update=allow_update)


__all__ = [
    "CoalescedAtlas",
    "HamiltonCommitmentLedger",
    "LedgerAssignment",
    "MAX_FULL_COV_DIM",
    "MAX_LEDGER_PARTICLES",
    "PrototypeAtlas",
    "SCHEMA_VERSION",
    "batch_annular_gap_ratio",
    "coalesce_directional_ltc",
    "coalesce_pythagorean_ltv",
    "compute_connectivity_gate",
    "compute_rho_atom",
    "cov_add_outer",
    "cov_directional_var",
    "cov_from_scalar",
    "cov_principal_split",
    "cov_shrink_mp",
    "cov_trace_mean",
    "manifold_connectivity_stats",
    "validate_atlas_state",
    "validate_ledger_state",
]
