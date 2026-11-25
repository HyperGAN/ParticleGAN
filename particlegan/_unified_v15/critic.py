"""Unified-v15 4-Law + CondFiber critic/generator step records, live endpoint + secant-probe hooks, and Adam."""
from __future__ import annotations

from collections import OrderedDict
import math
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from .._bispf.thermal import _LEGACY_THERMAL_CHANNEL
from .._t1_coupling.critic import (
    _T1C_PACKET_KEYS,
    _T1CouplingRecordMixin,
    _extract_t1c_inputs_parts,
    _lookup_t1c_capture_parts,
    _record_t1c_capture,
    _validate_t1c_packet,
)
from .._t2_critic.critic import (
    _clear_live_capture_entries,
    _cond_signature,
    _tensor_identity_key,
    _trim_live_captures,
)
from ..grad_regularizers import CriticStepRecord
from ..k3p import K3PCriticAdam
from .._unified_v11.bank import UnifiedV11CondFiberBank, validate_v11_bank_packet

_MAX_D_PROBES = 2
_PROBE_SLICE = 32

_UNIFIED_V15_PACKET_KEYS = frozenset({
    *_T1C_PACKET_KEYS,
    "step_scale",
    "ema_margin",
    "ema_data_scale",
    "ema_push_res",
    "margin_initialized",
    "g_func_gain",
    "cond_fiber_bank",
    "highd_basin_bank",
    "sparse_obs_bank",
})


_HIGHD_BANK_KEYS = frozenset({
    "k",
    "dim",
    "sigma_z_ema",
    "sigma_z_init",
    "w_y",
    "mu_y",
    "cov_y",
    "rad_y",
    "w_x",
    "mu_x",
    "cov_x",
})


def validate_v15_highd_bank_packet(packet: Any) -> Dict[str, Any]:
    """Validate checkpoint schema for ``UnifiedV15HighDBasinBank``."""
    if not isinstance(packet, dict):
        raise ValueError("invalid unified_v15 highd_basin_bank packet")
    k_val = packet.get("k")
    d_val = packet.get("dim")
    sig_ema = packet.get("sigma_z_ema")
    sig_init = packet.get("sigma_z_init")
    if type(k_val) is not int or k_val < 0 or k_val > 64:
        raise ValueError("invalid highd_basin_bank k")
    if type(d_val) is not int or d_val < 0 or d_val > 1024:
        raise ValueError("invalid highd_basin_bank dim")
    if isinstance(sig_ema, bool) or not isinstance(sig_ema, (int, float)) or not math.isfinite(sig_ema) or float(sig_ema) < 0.0:
        raise ValueError("invalid highd_basin_bank sigma_z_ema")
    if type(sig_init) is not bool:
        raise ValueError("invalid highd_basin_bank sigma_z_init")
    if k_val == 0:
        if set(packet) != {"k", "dim", "sigma_z_ema", "sigma_z_init"}:
            raise ValueError("invalid empty highd_basin_bank packet keys")
        return packet
    if set(packet) != _HIGHD_BANK_KEYS or d_val < 1:
        raise ValueError("invalid populated highd_basin_bank packet keys")
    expected_shapes = {
        "w_y": (k_val,),
        "mu_y": (k_val, d_val),
        "cov_y": (k_val, d_val, d_val),
        "rad_y": (k_val,),
        "w_x": (k_val,),
        "mu_x": (k_val, d_val),
        "cov_x": (k_val, d_val, d_val),
    }
    for name, shp in expected_shapes.items():
        ten = packet[name]
        if not isinstance(ten, torch.Tensor) or tuple(ten.shape) != shp or not bool(torch.isfinite(ten).all()):
            raise ValueError(f"invalid highd_basin_bank tensor {name}")
        if name in ("w_y", "w_x") and float(ten.amin().item()) < 0.0:
            raise ValueError(f"negative highd_basin_bank weight {name}")
        if name == "rad_y" and float(ten.amin().item()) <= 0.0:
            raise ValueError("nonpositive highd_basin_bank rad_y")
    return packet


class UnifiedV15HighDBasinBank:
    """Checkpointed, zero-`eigh`, Haar-O(d)-invariant streaming basin statistics for ``16 < dim <= 256``."""

    def __init__(self, *, max_basins: int = 16, gamma: float = 0.90) -> None:
        self.max_basins = int(max_basins)
        self.gamma = float(gamma)
        self.dim: int = 0
        self.k: int = 0
        self.sigma_z_ema: float = 0.0
        self.sigma_z_init: bool = False
        self.w_y: Optional[torch.Tensor] = None
        self.mu_y: Optional[torch.Tensor] = None
        self.cov_y: Optional[torch.Tensor] = None
        self.rad_y: Optional[torch.Tensor] = None
        self.w_x: Optional[torch.Tensor] = None
        self.mu_x: Optional[torch.Tensor] = None
        self.cov_x: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.dim = 0
        self.k = 0
        self.sigma_z_ema = 0.0
        self.sigma_z_init = False
        self.w_y = self.mu_y = self.cov_y = self.rad_y = None
        self.w_x = self.mu_x = self.cov_x = None

    def to_packet(self) -> Dict[str, Any]:
        if self.k == 0 or self.w_y is None:
            return {
                "k": 0,
                "dim": int(self.dim),
                "sigma_z_ema": float(self.sigma_z_ema),
                "sigma_z_init": bool(self.sigma_z_init),
            }
        return {
            "k": int(self.k),
            "dim": int(self.dim),
            "sigma_z_ema": float(self.sigma_z_ema),
            "sigma_z_init": bool(self.sigma_z_init),
            "w_y": self.w_y.detach().to(device="cpu", dtype=torch.float64).clone(),
            "mu_y": self.mu_y.detach().to(device="cpu", dtype=torch.float64).clone(),
            "cov_y": self.cov_y.detach().to(device="cpu", dtype=torch.float64).clone(),
            "rad_y": self.rad_y.detach().to(device="cpu", dtype=torch.float64).clone(),
            "w_x": self.w_x.detach().to(device="cpu", dtype=torch.float64).clone(),
            "mu_x": self.mu_x.detach().to(device="cpu", dtype=torch.float64).clone(),
            "cov_x": self.cov_x.detach().to(device="cpu", dtype=torch.float64).clone(),
        }

    def load_packet(self, packet: Any) -> None:
        validate_v15_highd_bank_packet(packet)
        self.dim = int(packet["dim"])
        self.k = int(packet["k"])
        self.sigma_z_ema = float(packet["sigma_z_ema"])
        self.sigma_z_init = bool(packet["sigma_z_init"])
        if self.k == 0:
            self.w_y = self.mu_y = self.cov_y = self.rad_y = None
            self.w_x = self.mu_x = self.cov_x = None
            return
        self.w_y = packet["w_y"].detach().to(dtype=torch.float64).clone()
        self.mu_y = packet["mu_y"].detach().to(dtype=torch.float64).clone()
        self.cov_y = packet["cov_y"].detach().to(dtype=torch.float64).clone()
        self.rad_y = packet["rad_y"].detach().to(dtype=torch.float64).clone()
        self.w_x = packet["w_x"].detach().to(dtype=torch.float64).clone()
        self.mu_x = packet["mu_x"].detach().to(dtype=torch.float64).clone()
        self.cov_x = packet["cov_x"].detach().to(dtype=torch.float64).clone()

    def restore_device(self, device: torch.device) -> None:
        if self.k == 0 or self.w_y is None:
            return
        self.w_y = self.w_y.to(device=device, dtype=torch.float64)
        self.mu_y = self.mu_y.to(device=device, dtype=torch.float64)
        self.cov_y = self.cov_y.to(device=device, dtype=torch.float64)
        self.rad_y = self.rad_y.to(device=device, dtype=torch.float64)
        self.w_x = self.w_x.to(device=device, dtype=torch.float64)
        self.mu_x = self.mu_x.to(device=device, dtype=torch.float64)
        self.cov_x = self.cov_x.to(device=device, dtype=torch.float64)

    def copy(self) -> "UnifiedV15HighDBasinBank":
        dup = UnifiedV15HighDBasinBank(max_basins=self.max_basins, gamma=self.gamma)
        dup.dim = self.dim
        dup.k = self.k
        dup.sigma_z_ema = self.sigma_z_ema
        dup.sigma_z_init = self.sigma_z_init
        if self.k > 0 and self.w_y is not None:
            dup.w_y = self.w_y.detach().clone()
            dup.mu_y = self.mu_y.detach().clone()
            dup.cov_y = self.cov_y.detach().clone()
            dup.rad_y = self.rad_y.detach().clone()
            dup.w_x = self.w_x.detach().clone()
            dup.mu_x = self.mu_x.detach().clone()
            dup.cov_x = self.cov_x.detach().clone()
        return dup


_SPARSE_OBS_SCALAR_KEYS = frozenset({
    "step",
    "dim",
    "dz",
    "buf_ptr",
    "buf_n",
    "proto_k",
    "p_y",
    "p_x",
    "sparse_push_res",
})


def validate_v15_sparse_obs_bank_packet(packet: Any) -> Dict[str, Any]:
    """Validate checkpoint schema for ``UnifiedV15SparseObsBank``."""
    if not isinstance(packet, dict):
        raise ValueError("invalid unified_v15 sparse_obs_bank packet")
    if not _SPARSE_OBS_SCALAR_KEYS.issubset(packet):
        raise ValueError("missing unified_v15 sparse_obs_bank scalar keys")
    step = packet["step"]
    dim = packet["dim"]
    dz = packet["dz"]
    buf_ptr = packet["buf_ptr"]
    buf_n = packet["buf_n"]
    proto_k = packet["proto_k"]
    for name, val, lo, hi in (
        ("step", step, 0, 10**9),
        ("dim", dim, 0, 2048),
        ("dz", dz, 0, 1024),
        ("buf_ptr", buf_ptr, 0, 4096),
        ("buf_n", buf_n, 0, 4096),
        ("proto_k", proto_k, 0, 256),
    ):
        if type(val) is not int or val < lo or val > hi:
            raise ValueError(f"invalid sparse_obs_bank {name}")
    for name in ("p_y", "p_x"):
        val = packet[name]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or float(val) < 0.0 or float(val) > 1.0:
            raise ValueError(f"invalid sparse_obs_bank {name}")
    spr = packet["sparse_push_res"]
    if spr is not None:
        if isinstance(spr, bool) or not isinstance(spr, (int, float)) or not math.isfinite(spr) or float(spr) < 0.0:
            raise ValueError("invalid sparse_obs_bank sparse_push_res")
    if step == 0:
        if set(packet) != _SPARSE_OBS_SCALAR_KEYS or buf_n != 0 or proto_k != 0:
            raise ValueError("invalid empty sparse_obs_bank packet keys")
        return packet
    if dim < 1 or buf_n < 1:
        raise ValueError("invalid active sparse_obs_bank dimensions")
    expected_keys = set(_SPARSE_OBS_SCALAR_KEYS) | {"mu_y", "mu_x", "S_y", "S_x", "buf_y", "buf_m"}
    expected_shapes: Dict[str, Tuple[int, ...]] = {
        "mu_y": (dim,),
        "mu_x": (dim,),
        "S_y": (dim, dim),
        "S_x": (dim, dim),
        "buf_y": (buf_n, dim),
        "buf_m": (buf_n, dim),
    }
    if dz > 0:
        expected_keys |= {"S_zz", "S_zx"}
        expected_shapes["S_zz"] = (dz, dz)
        expected_shapes["S_zx"] = (dz, dim)
    if proto_k > 0:
        if dz < 1:
            raise ValueError("sparse_obs_bank prototypes require dz >= 1")
        expected_keys |= {"proto_y", "proto_w", "proto_z", "proto_b"}
        expected_shapes["proto_y"] = (proto_k, dim)
        expected_shapes["proto_w"] = (proto_k,)
        expected_shapes["proto_z"] = (proto_k, dz)
        expected_shapes["proto_b"] = (proto_k,)
    if set(packet) != expected_keys:
        raise ValueError("invalid populated sparse_obs_bank packet keys")
    for name, shp in expected_shapes.items():
        ten = packet[name]
        if not isinstance(ten, torch.Tensor) or tuple(ten.shape) != shp:
            raise ValueError(f"invalid sparse_obs_bank tensor {name}")
        if name == "buf_m":
            if ten.dtype != torch.bool:
                raise ValueError("invalid sparse_obs_bank buf_m dtype")
        else:
            if not ten.is_floating_point() or not bool(torch.isfinite(ten).all()):
                raise ValueError(f"invalid sparse_obs_bank tensor {name}")
            if name == "proto_w" and float(ten.amin().item()) < 0.0:
                raise ValueError("negative sparse_obs_bank proto_w")
    return packet


class UnifiedV15SparseObsBank:
    """Checkpointed, zero-`eigh`, zero-RNG Wiener-NLM & semi-discrete Laguerre bank for sparse-masked vectors."""

    def __init__(self, *, max_buf: int = 2048, max_protos: int = 128) -> None:
        self.max_buf = int(max_buf)
        self.max_protos = int(max_protos)
        self.reset()

    def reset(self) -> None:
        self.step: int = 0
        self.dim: int = 0
        self.dz: int = 0
        self.buf_ptr: int = 0
        self.buf_n: int = 0
        self.proto_k: int = 0
        self.p_y: float = 0.0
        self.p_x: float = 0.0
        self.sparse_push_res: Optional[float] = None
        self.mu_y: Optional[torch.Tensor] = None
        self.mu_x: Optional[torch.Tensor] = None
        self.S_y: Optional[torch.Tensor] = None
        self.S_x: Optional[torch.Tensor] = None
        self.S_zz: Optional[torch.Tensor] = None
        self.S_zx: Optional[torch.Tensor] = None
        self.buf_y: Optional[torch.Tensor] = None
        self.buf_m: Optional[torch.Tensor] = None
        self.proto_y: Optional[torch.Tensor] = None
        self.proto_w: Optional[torch.Tensor] = None
        self.proto_z: Optional[torch.Tensor] = None
        self.proto_b: Optional[torch.Tensor] = None

    def to_packet(self) -> Dict[str, Any]:
        pkt: Dict[str, Any] = {
            "step": int(self.step),
            "dim": int(self.dim),
            "dz": int(self.dz),
            "buf_ptr": int(self.buf_ptr),
            "buf_n": int(self.buf_n),
            "proto_k": int(self.proto_k),
            "p_y": float(self.p_y),
            "p_x": float(self.p_x),
            "sparse_push_res": None if self.sparse_push_res is None else float(self.sparse_push_res),
        }
        if self.step == 0 or self.S_y is None or self.buf_y is None or self.buf_m is None:
            pkt["step"] = 0
            pkt["buf_n"] = 0
            pkt["proto_k"] = 0
            return pkt
        pkt["mu_y"] = self.mu_y.detach().to(device="cpu", dtype=torch.float32).clone()
        pkt["mu_x"] = self.mu_x.detach().to(device="cpu", dtype=torch.float32).clone()
        pkt["S_y"] = self.S_y.detach().to(device="cpu", dtype=torch.float32).clone()
        pkt["S_x"] = self.S_x.detach().to(device="cpu", dtype=torch.float32).clone()
        pkt["buf_y"] = self.buf_y[: self.buf_n].detach().to(device="cpu", dtype=torch.float32).clone()
        pkt["buf_m"] = self.buf_m[: self.buf_n].detach().to(device="cpu", dtype=torch.bool).clone()
        if self.dz > 0 and self.S_zz is not None and self.S_zx is not None:
            pkt["S_zz"] = self.S_zz.detach().to(device="cpu", dtype=torch.float32).clone()
            pkt["S_zx"] = self.S_zx.detach().to(device="cpu", dtype=torch.float32).clone()
        if self.proto_k > 0 and self.proto_y is not None:
            pk = int(self.proto_k)
            pkt["proto_y"] = self.proto_y[:pk].detach().to(device="cpu", dtype=torch.float32).clone()
            pkt["proto_w"] = self.proto_w[:pk].detach().to(device="cpu", dtype=torch.float32).clone()
            pkt["proto_z"] = self.proto_z[:pk].detach().to(device="cpu", dtype=torch.float32).clone()
            pkt["proto_b"] = self.proto_b[:pk].detach().to(device="cpu", dtype=torch.float32).clone()
        return pkt

    def load_packet(self, packet: Any) -> None:
        validate_v15_sparse_obs_bank_packet(packet)
        self.reset()
        self.step = int(packet["step"])
        self.dim = int(packet["dim"])
        self.dz = int(packet["dz"])
        self.buf_ptr = int(packet["buf_ptr"])
        self.buf_n = int(packet["buf_n"])
        self.proto_k = int(packet["proto_k"])
        self.p_y = float(packet["p_y"])
        self.p_x = float(packet["p_x"])
        self.sparse_push_res = None if packet["sparse_push_res"] is None else float(packet["sparse_push_res"])
        if self.step == 0:
            return
        self.mu_y = packet["mu_y"].detach().to(dtype=torch.float32).clone()
        self.mu_x = packet["mu_x"].detach().to(dtype=torch.float32).clone()
        self.S_y = packet["S_y"].detach().to(dtype=torch.float32).clone()
        self.S_x = packet["S_x"].detach().to(dtype=torch.float32).clone()
        cap_b = max(self.max_buf, self.buf_n)
        self.buf_y = torch.zeros((cap_b, self.dim), dtype=torch.float32)
        self.buf_m = torch.zeros((cap_b, self.dim), dtype=torch.bool)
        self.buf_y[: self.buf_n] = packet["buf_y"].detach().to(dtype=torch.float32)
        self.buf_m[: self.buf_n] = packet["buf_m"].detach().to(dtype=torch.bool)
        if self.dz > 0:
            self.S_zz = packet["S_zz"].detach().to(dtype=torch.float32).clone()
            self.S_zx = packet["S_zx"].detach().to(dtype=torch.float32).clone()
        if self.proto_k > 0:
            cap_p = max(self.max_protos, self.proto_k)
            self.proto_y = torch.zeros((cap_p, self.dim), dtype=torch.float32)
            self.proto_w = torch.zeros((cap_p,), dtype=torch.float32)
            self.proto_z = torch.zeros((cap_p, self.dz), dtype=torch.float32)
            self.proto_b = torch.zeros((cap_p,), dtype=torch.float32)
            pk = self.proto_k
            self.proto_y[:pk] = packet["proto_y"].detach().to(dtype=torch.float32)
            self.proto_w[:pk] = packet["proto_w"].detach().to(dtype=torch.float32)
            self.proto_z[:pk] = packet["proto_z"].detach().to(dtype=torch.float32)
            self.proto_b[:pk] = packet["proto_b"].detach().to(dtype=torch.float32)

    def restore_device(self, device: torch.device) -> None:
        if self.step == 0 or self.S_y is None:
            return
        for attr in ("mu_y", "mu_x", "S_y", "S_x", "S_zz", "S_zx", "buf_y", "proto_y", "proto_w", "proto_z", "proto_b"):
            t = getattr(self, attr, None)
            if t is not None:
                setattr(self, attr, t.to(device=device, dtype=torch.float32))
        if self.buf_m is not None:
            self.buf_m = self.buf_m.to(device=device, dtype=torch.bool)

    def copy(self) -> "UnifiedV15SparseObsBank":
        dup = UnifiedV15SparseObsBank(max_buf=self.max_buf, max_protos=self.max_protos)
        dup.step = self.step
        dup.dim = self.dim
        dup.dz = self.dz
        dup.buf_ptr = self.buf_ptr
        dup.buf_n = self.buf_n
        dup.proto_k = self.proto_k
        dup.p_y = self.p_y
        dup.p_x = self.p_x
        dup.sparse_push_res = self.sparse_push_res
        for attr in ("mu_y", "mu_x", "S_y", "S_x", "S_zz", "S_zx", "buf_y", "buf_m", "proto_y", "proto_w", "proto_z", "proto_b"):
            t = getattr(self, attr, None)
            if t is not None:
                setattr(dup, attr, t.detach().clone())
        return dup


def _has_spatial_dims(x: torch.Tensor) -> bool:
    """Return True iff ``x`` has non-singleton spatial dimensions beyond ``(B, C)``."""
    return x.ndim >= 3 and any(int(s) > 1 for s in x.shape[2:])


def _probe_batch_count(b_full: int, elem_dim: int) -> int:
    """Element-budgeted probe count eliminating b_p=2 rare-mode blindness (D-04)."""
    if elem_dim <= 16:
        return min(b_full, 16)
    if elem_dim <= 16384:
        return min(b_full, max(8, min(_PROBE_SLICE, 65536 // max(1, elem_dim))))
    return min(b_full, max(4, min(16, 65536 // max(1, elem_dim))))


@torch.no_grad()
def _select_probe_indices(ref: torch.Tensor, b_full: int, b_probe: int) -> Optional[torch.Tensor]:
    """Zero-sync deterministic stratified / rare-tail probe indices (0 RNG, 0% rare-mode blindness)."""
    if b_probe >= b_full:
        return None
    elem_dim = max(1, int(ref[0].numel()))
    flat = ref.detach().reshape(b_full, -1).float()
    dev_sq = (flat - flat.mean(dim=0, keepdim=True)).square().sum(dim=1)
    order = torch.argsort(dev_sq, stable=True)
    if elem_dim > 16:
        if b_probe <= 1:
            return order[-1:]
        pos = torch.linspace(0.0, float(b_full - 1), steps=b_probe, device=ref.device).round().to(dtype=torch.long)
        pos = pos.clamp(0, b_full - 1)
        return order.index_select(0, pos)
    idx = torch.arange(b_probe, device=ref.device, dtype=torch.long)
    i95 = min(b_full - 1, int(round(0.95 * float(b_full - 1))))
    i99 = max(0, b_full - max(1, b_full // 100))
    v95 = dev_sq[order[i95]]
    v99 = dev_sq[order[i99]]
    vmax = dev_sq[order[-1]]
    is_extreme = (v95 > 0.0) & (vmax > (4.0 * v95))
    is_rare_cluster = (v95 > 0.0) & (v99 > (1.60 * v95))
    tail_idx = torch.where(is_extreme, order[-1], order[i99])
    use_tail = (is_extreme | is_rare_cluster) & (tail_idx >= b_probe)
    idx[-1] = torch.where(use_tail, tail_idx, idx[-1])
    return idx


def _index_probe_arg(val: Any, b_full: int, idx: Optional[torch.Tensor]) -> Any:
    if isinstance(val, torch.Tensor):
        if idx is not None and val.ndim >= 1 and val.shape[0] == b_full:
            return val.detach().index_select(0, idx.to(device=val.device))
        return val.detach()
    if isinstance(val, dict):
        return {k: _index_probe_arg(v, b_full, idx) for k, v in val.items()}
    if isinstance(val, tuple):
        return tuple(_index_probe_arg(v, b_full, idx) for v in val)
    if isinstance(val, list):
        return [_index_probe_arg(v, b_full, idx) for v in val]
    return val


def _slice_probe_arg(val: Any, b_full: int, b_probe: int) -> Any:
    if isinstance(val, torch.Tensor):
        if val.ndim >= 1 and val.shape[0] == b_full:
            return val.detach()[:b_probe]
        return val.detach()
    if isinstance(val, dict):
        return {k: _slice_probe_arg(v, b_full, b_probe) for k, v in val.items()}
    if isinstance(val, tuple):
        return tuple(_slice_probe_arg(v, b_full, b_probe) for v in val)
    if isinstance(val, list):
        return [_slice_probe_arg(v, b_full, b_probe) for v in val]
    return val


@torch.no_grad()
def _eval_v15_probe_forward(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    record: Any,
) -> Optional[torch.Tensor]:
    """Evaluate ``module(*args, **kwargs)`` in ``eval()`` mode without mutating RNGs, hooks, or BN buffers."""
    modes = [(m, m.training) for m in module.modules()]
    prev_probe = bool(getattr(record, "_in_probe", False)) if record is not None else False
    cpu_rng = torch.get_rng_state()
    py_rng = random.getstate()
    np_mod = sys.modules.get("numpy")
    np_rng = np_mod.random.get_state() if np_mod is not None and hasattr(np_mod, "random") else None
    cuda_rng = None
    cuda_dev = None
    if torch.cuda.is_initialized():
        cuda_dev = torch.cuda.current_device()
        cuda_rng = torch.cuda.get_rng_state(cuda_dev)
    if record is not None:
        record._in_probe = True
    try:
        module.eval()
        first_p = next(module.parameters(), None)
        use_bf16 = (
            first_p is not None
            and first_p.device.type == "cuda"
            and bool(getattr(module, "_particlegan_bf16_autocast", getattr(module, "_unified_v4_bf16_autocast", False)))
        )
        if use_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = module(*args, **kwargs)
        else:
            out = module(*args, **kwargs)
        target = out[0] if isinstance(out, (tuple, list)) and out else out
        if not isinstance(target, torch.Tensor):
            return None
        return target.detach().to(dtype=torch.float32)
    finally:
        if record is not None:
            record._in_probe = prev_probe
        for m, flag in modes:
            m.training = flag
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None and cuda_dev is not None:
            torch.cuda.set_rng_state(cuda_rng, cuda_dev)
        random.setstate(py_rng)
        if np_rng is not None and np_mod is not None:
            np_mod.random.set_state(np_rng)


def _validate_unified_v15_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        raise ValueError("invalid unified_v15 record packet fields")
    pkeys = set(packet)
    if pkeys != _UNIFIED_V15_PACKET_KEYS and pkeys != (_UNIFIED_V15_PACKET_KEYS | {"atomic_ledger"}):
        raise ValueError("invalid unified_v15 record packet fields")
    step_scale = packet["step_scale"]
    if (
        isinstance(step_scale, bool)
        or not isinstance(step_scale, (int, float))
        or not math.isfinite(step_scale)
        or float(step_scale) <= 0.0
        or float(step_scale) > 1.0
    ):
        raise ValueError("invalid unified_v15 step_scale outside (0, 1]")
    ema_margin = packet["ema_margin"]
    if isinstance(ema_margin, bool) or not isinstance(ema_margin, (int, float)) or not math.isfinite(ema_margin):
        raise ValueError("invalid unified_v15 ema_margin")
    ema_data_scale = packet["ema_data_scale"]
    if (
        isinstance(ema_data_scale, bool)
        or not isinstance(ema_data_scale, (int, float))
        or not math.isfinite(ema_data_scale)
        or float(ema_data_scale) <= 0.0
    ):
        raise ValueError("invalid unified_v15 ema_data_scale")
    ema_push_res = packet["ema_push_res"]
    if (
        isinstance(ema_push_res, bool)
        or not isinstance(ema_push_res, (int, float))
        or not math.isfinite(ema_push_res)
        or float(ema_push_res) < 0.0
    ):
        raise ValueError("invalid unified_v15 ema_push_res")
    if type(packet["margin_initialized"]) is not bool:
        raise ValueError("invalid unified_v15 margin_initialized")
    g_func_gain = packet["g_func_gain"]
    if (
        isinstance(g_func_gain, bool)
        or not isinstance(g_func_gain, (int, float))
        or not math.isfinite(g_func_gain)
        or float(g_func_gain) <= 0.0
        or float(g_func_gain) > 1.0
    ):
        raise ValueError("invalid unified_v15 g_func_gain outside (0, 1]")
    validate_v11_bank_packet(packet["cond_fiber_bank"])
    validate_v15_highd_bank_packet(packet["highd_basin_bank"])
    validate_v15_sparse_obs_bank_packet(packet["sparse_obs_bank"])
    t1c_sub = {k: packet[k] for k in _T1C_PACKET_KEYS}
    if "atomic_ledger" in packet:
        t1c_sub["atomic_ledger"] = packet["atomic_ledger"]
    _validate_t1c_packet(t1c_sub)
    return packet


class _UnifiedV11RecordMixin(_T1CouplingRecordMixin):
    """Unified-v15 critic record mixin combining 4-Law persistent state and zero-`eigh` CondFiberBank."""

    def _init_unified_v15_fields(self) -> None:
        self._init_t1c_fields()
        self._step_scale: float = 1.0
        self._t1c_disc_gate: float = 0.0
        self._simplicial_step: bool = False
        self._ema_margin_val: float = 0.0
        self._ema_data_scale_val: float = 1.0
        self._ema_push_res_val: float = 1.0
        self._margin_initialized_val: bool = False
        self._ema_margin_t: Optional[torch.Tensor] = None
        self._ema_data_scale_t: Optional[torch.Tensor] = None
        self._ema_push_res_t: Optional[torch.Tensor] = None
        self._margin_init_t: Optional[torch.Tensor] = None
        self._g_func_gain: float = 1.0
        self.cond_fiber_bank = UnifiedV11CondFiberBank()
        self.highd_basin_bank = UnifiedV15HighDBasinBank()
        self.sparse_obs_bank = UnifiedV15SparseObsBank()
        self._live_captures: List[List[Any]] = []
        self._last_probe_actual_input: Optional[torch.Tensor] = None
        self._d_probe_list: List[Tuple[Tuple[Any, ...], Dict[str, Any], torch.Tensor]] = []
        self._g_probe_cache: Optional[Tuple[nn.Module, Tuple[Any, ...], Dict[str, Any], torch.Tensor]] = None
        self._g_last_latent: Optional[Tuple[nn.Module, torch.Tensor]] = None
        self._in_probe: bool = False
        self._enable_secant_probe: bool = True

    def _take_pending_ema_tensors(self) -> Tuple[List[str], List[torch.Tensor]]:
        keys: List[str] = []
        tens: List[torch.Tensor] = []
        for k, attr in (
            ("ema_margin", "_ema_margin_t"),
            ("ema_data_scale", "_ema_data_scale_t"),
            ("ema_push_res", "_ema_push_res_t"),
            ("margin_initialized", "_margin_init_t"),
        ):
            t = getattr(self, attr, None)
            if t is not None:
                keys.append(k)
                tens.append(t)
        return keys, tens

    def _apply_flushed_ema_values(self, keys: List[str], vals: List[float]) -> None:
        for k, v in zip(keys, vals):
            if k == "ema_margin":
                self._ema_margin_val = float(v)
                self._ema_margin_t = None
            elif k == "ema_data_scale":
                self._ema_data_scale_val = float(v)
                self._ema_data_scale_t = None
            elif k == "ema_push_res":
                self._ema_push_res_val = float(v)
                self._ema_push_res_t = None
            elif k == "margin_initialized":
                self._margin_initialized_val = bool(v > 0.5)
                self._margin_init_t = None

    def _flush_ema_scalars(self) -> None:
        keys, tens = self._take_pending_ema_tensors()
        if not tens:
            return
        vals = torch.stack(tens).tolist()
        self._apply_flushed_ema_values(keys, vals)

    @property
    def ema_margin(self) -> float:
        if self._ema_margin_t is not None:
            self._flush_ema_scalars()
        return self._ema_margin_val

    @ema_margin.setter
    def ema_margin(self, val: float) -> None:
        self._ema_margin_val = float(val)
        self._ema_margin_t = None

    @property
    def ema_data_scale(self) -> float:
        if self._ema_data_scale_t is not None:
            self._flush_ema_scalars()
        return self._ema_data_scale_val

    @ema_data_scale.setter
    def ema_data_scale(self, val: float) -> None:
        self._ema_data_scale_val = float(val)
        self._ema_data_scale_t = None

    @property
    def ema_push_res(self) -> float:
        if self._ema_push_res_t is not None:
            self._flush_ema_scalars()
        return self._ema_push_res_val

    @ema_push_res.setter
    def ema_push_res(self, val: float) -> None:
        self._ema_push_res_val = float(val)
        self._ema_push_res_t = None

    @property
    def margin_initialized(self) -> bool:
        if self._margin_init_t is not None:
            self._flush_ema_scalars()
        return self._margin_initialized_val

    @margin_initialized.setter
    def margin_initialized(self, val: bool) -> None:
        self._margin_initialized_val = bool(val)
        self._margin_init_t = None

    @property
    def ema_residual(self) -> float:
        base = super().ema_residual
        if self.margin_initialized:
            margin_res = max(0.0, abs(float(self.ema_margin)) - 0.80) * 0.25
            return max(base, margin_res)
        return base

    @property
    def rgdt_cond_initialized(self) -> bool:
        return self.t1c_cond_initialized

    @rgdt_cond_initialized.setter
    def rgdt_cond_initialized(self, val: bool) -> None:
        self.t1c_cond_initialized = bool(val)

    @property
    def rgdt_cond_ema(self) -> float:
        return self.t1c_cond_ema

    @rgdt_cond_ema.setter
    def rgdt_cond_ema(self, val: float) -> None:
        self.t1c_cond_ema = float(val)

    @torch.no_grad()
    def observe_margin_and_scale(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        x_real: Optional[torch.Tensor] = None,
        x_fake: Optional[torch.Tensor] = None,
    ) -> None:
        """Update checkpointed two-player critic margin, data scale, and pushforward residual EMAs (0 sync)."""
        if getattr(self, "_in_penalty", False) or getattr(self, "_in_probe", False) or getattr(self, "_in_protected_probe", False):
            return
        dev = real_logits.device
        r_m = real_logits.detach().float().mean().double()
        f_m = fake_logits.detach().float().mean().double()
        margin = (r_m - f_m).clamp(-50.0, 50.0)
        fin_m = torch.isfinite(margin)

        if self._ema_margin_t is not None and self._ema_margin_t.device == dev:
            prev_m = self._ema_margin_t
        else:
            if self._ema_margin_t is not None:
                self._flush_ema_scalars()
            prev_m = real_logits.new_full((), self._ema_margin_val, dtype=torch.float64)

        if self._margin_init_t is not None and self._margin_init_t.device == dev:
            prev_init = self._margin_init_t
        else:
            if self._margin_init_t is not None:
                self._flush_ema_scalars()
            prev_init = real_logits.new_full((), 1.0 if self._margin_initialized_val else 0.0, dtype=torch.float64)

        was_init = prev_init > 0.5
        updated_m = torch.where(was_init, 0.90 * prev_m + 0.10 * margin, margin)
        self._ema_margin_t = torch.where(fin_m, updated_m, prev_m)
        self._margin_init_t = torch.where(fin_m, prev_init.new_full((), 1.0), prev_init)

        if x_real is not None and x_real.numel() > 0:
            self.observe_scale(x_real, x_fake)

    @torch.no_grad()
    def observe_sparse_push_res(self, val: float) -> None:
        """Record checkpointed sparse-observation transport residual into pushforward EMA."""
        v = max(0.0, min(10.0, float(val)))
        if not math.isfinite(v):
            return
        self.sparse_obs_bank.sparse_push_res = v
        self._ema_push_res_val = v
        self._ema_push_res_t = None

    @torch.no_grad()
    def observe_scale(
        self,
        x_real: torch.Tensor,
        x_fake: Optional[torch.Tensor] = None,
    ) -> None:
        """Update checkpointed data scale and pushforward residual EMAs (0 sync)."""
        if getattr(self, "_in_penalty", False) or getattr(self, "_in_probe", False) or getattr(self, "_in_protected_probe", False):
            return
        if x_real is None or x_real.numel() == 0:
            return
        dev = x_real.device
        step_tag = int(getattr(self, "observed_steps", 0))
        xr = x_real.detach().reshape(x_real.shape[0], -1).float()
        dim_s = int(xr.shape[1])
        is_sparse: Optional[torch.Tensor] = None
        if dim_s >= 128 and not _has_spatial_dims(x_real):
            obs_rate = (xr != 0).float().mean().double()
            is_sparse = (obs_rate < 0.50) & (obs_rate > 1e-4)
            p_eff = torch.where(is_sparse, obs_rate.clamp_min(1e-4), obs_rate.new_full((), 1.0))
            sc_sq = xr.square().sum(dim=-1).mean().double() / p_eff
            sc = sc_sq.clamp_min(1e-24).sqrt().clamp_min(1e-12)
        else:
            sc = xr.square().sum(dim=-1).mean().sqrt().clamp_min(1e-12).double()
        fin_sc = torch.isfinite(sc)

        if self._ema_data_scale_t is not None and self._ema_data_scale_t.device == dev:
            prev_sc = self._ema_data_scale_t
        else:
            if self._ema_data_scale_t is not None:
                self._flush_ema_scalars()
            prev_sc = x_real.new_full((), self._ema_data_scale_val, dtype=torch.float64)

        sc_cap = sc if step_tag == 0 else sc.clamp(min=prev_sc * 0.25, max=prev_sc * 4.0)
        cand_sc = sc if step_tag == 0 else (0.95 * prev_sc + 0.05 * sc_cap)
        new_sc = torch.where(fin_sc, cand_sc, prev_sc)
        self._ema_data_scale_t = new_sc

        if (
            x_fake is not None
            and x_fake.ndim >= 2
            and int(x_fake.shape[0]) == int(x_real.shape[0])
            and int(x_fake.numel()) == int(x_real.numel())
        ):
            xf = x_fake.detach().reshape(x_fake.shape[0], -1).float()
            mean_gap = (xf.mean(dim=0) - xr.mean(dim=0)).norm().double()
            std_gap = (xf.std().double() - xr.std().double()).abs()
            rel_gap = ((mean_gap + 0.5 * std_gap) / new_sc.clamp_min(1e-12)).clamp_max(10.0)
            fin_rg = torch.isfinite(rel_gap)

            if self._ema_push_res_t is not None and self._ema_push_res_t.device == dev:
                prev_pr = self._ema_push_res_t
            else:
                if self._ema_push_res_t is not None:
                    self._flush_ema_scalars()
                prev_pr = x_real.new_full((), self._ema_push_res_val, dtype=torch.float64)

            cand_pr = rel_gap if step_tag == 0 else (0.92 * prev_pr + 0.08 * rel_gap)
            sp_res = self.sparse_obs_bank.sparse_push_res
            if is_sparse is not None and sp_res is not None:
                sp_t = x_real.new_full((), float(sp_res), dtype=torch.float64)
                cand_pr = torch.where(is_sparse, sp_t, cand_pr)
            self._ema_push_res_t = torch.where(fin_rg, cand_pr, prev_pr)

    def clear_transient_captures(self) -> None:
        if hasattr(self, "_live_captures") and self._live_captures:
            _clear_live_capture_entries(self._live_captures)
        self._last_probe_actual_input = None

    def clear_step_probes(self) -> None:
        if hasattr(self, "_d_probe_list") and self._d_probe_list:
            self._d_probe_list.clear()
        self._g_last_latent = None

    def _unified_v15_packet(self) -> Dict[str, Any]:
        return {
            **self._t1c_packet(),
            "step_scale": float(self._step_scale),
            "ema_margin": float(self.ema_margin),
            "ema_data_scale": float(self.ema_data_scale),
            "ema_push_res": float(self.ema_push_res),
            "margin_initialized": bool(self.margin_initialized),
            "g_func_gain": float(self._g_func_gain),
            "cond_fiber_bank": self.cond_fiber_bank.to_packet(),
            "highd_basin_bank": self.highd_basin_bank.to_packet(),
            "sparse_obs_bank": self.sparse_obs_bank.to_packet(),
        }

    def restore_device_state(self, device: torch.device) -> None:
        super().restore_device_state(device)
        self.highd_basin_bank.restore_device(device)
        self.sparse_obs_bank.restore_device(device)

    def _restore_unified_v15_packet(self, packet: Dict[str, Any]) -> None:
        _validate_unified_v15_packet(packet)
        t1c_sub = {k: packet[k] for k in _T1C_PACKET_KEYS}
        if "atomic_ledger" in packet:
            t1c_sub["atomic_ledger"] = packet["atomic_ledger"]
        self._restore_t1c_packet(t1c_sub)
        self._step_scale = float(packet["step_scale"])
        self.ema_margin = float(packet["ema_margin"])
        self.ema_data_scale = float(packet["ema_data_scale"])
        self.ema_push_res = float(packet["ema_push_res"])
        self.margin_initialized = bool(packet["margin_initialized"])
        self._g_func_gain = float(packet["g_func_gain"])
        self.cond_fiber_bank.load_packet(packet["cond_fiber_bank"])
        self.highd_basin_bank.load_packet(packet["highd_basin_bank"])
        self.sparse_obs_bank.load_packet(packet["sparse_obs_bank"])
        self._t1c_disc_gate = 0.0
        self._simplicial_step = False
        self.clear_transient_captures()
        self._live_captures = []
        self._d_probe_list = []
        self._g_probe_cache = None
        self._g_last_latent = None
        self._last_probe_actual_input = None
        self._in_probe = False

    def _copy_unified_v15_to(self, dup: Any) -> None:
        self._copy_t1c_to(dup)
        dup._step_scale = self._step_scale
        dup.ema_margin = self.ema_margin
        dup.ema_data_scale = self.ema_data_scale
        dup.ema_push_res = self.ema_push_res
        dup.margin_initialized = self.margin_initialized
        dup._g_func_gain = self._g_func_gain
        dup.cond_fiber_bank = self.cond_fiber_bank.copy()
        dup.highd_basin_bank = self.highd_basin_bank.copy()
        dup.sparse_obs_bank = self.sparse_obs_bank.copy()
        dup._t1c_disc_gate = 0.0
        dup._simplicial_step = False
        dup._live_captures = []
        dup._d_probe_list = []
        dup._g_probe_cache = None
        dup._g_last_latent = None
        dup._last_probe_actual_input = None
        dup._in_probe = False
        dup._enable_secant_probe = self._enable_secant_probe


class _UnifiedV15ObserverStepRecord(_UnifiedV11RecordMixin, CriticStepRecord):
    """Observer record for UnifiedV11EquilibratedOptimizer, DualNorm, and plain Adam critics."""

    def __init__(self) -> None:
        super().__init__()
        self._init_unified_v15_fields()

    def record_step(self, optimizer_or_lr: Any) -> None:
        super().record_step(optimizer_or_lr)
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()
        self.clear_step_probes()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "unified_v15": self._unified_v15_packet()}

    @staticmethod
    def base_state(state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict) or "unified_v15" not in state:
            raise ValueError("unified_v15 observer checkpoint is missing 'unified_v15' state")
        if set(state) != {*CriticStepRecord.KEYS, "unified_v15"}:
            raise ValueError("invalid unified_v15 observer record fields")
        _validate_unified_v15_packet(state["unified_v15"])
        return {key: state[key] for key in CriticStepRecord.KEYS}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(self.base_state(state))
        self._restore_unified_v15_packet(state["unified_v15"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_UnifiedV15ObserverStepRecord":
        dup = _UnifiedV15ObserverStepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_v15_to(dup)
        return dup


class UnifiedV15StepRecord(_UnifiedV11RecordMixin, CriticStepRecord):
    """Formulation critic step record traveling with ``UnifiedV15CriticAdam.state_dict()``."""

    RECORD_KEYS = (*CriticStepRecord.KEYS, "formulation", "unified_v15")

    def __init__(self) -> None:
        super().__init__(anchor=None)
        self._init_unified_v15_fields()

    def record_step(self, optimizer: torch.optim.Optimizer) -> None:
        lr = max(float(group["lr"]) for group in optimizer.param_groups)
        self.lr_last = lr
        self.lr_max = max(self.lr_max, lr)
        self.observed_steps += 1
        self._capture_ring = OrderedDict()
        self.clear_transient_captures()
        self.clear_step_probes()

    def state_dict(self) -> Dict[str, Any]:
        return {**super().state_dict(), "formulation": "unified_v15", "unified_v15": self._unified_v15_packet()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or set(state) != set(self.RECORD_KEYS):
            raise ValueError("invalid UnifiedV15StepRecord checkpoint keys")
        if state.get("formulation") != "unified_v15":
            raise ValueError("checkpoint formulation is not 'unified_v15'")
        for key in ("calls", "observed_steps"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError(f"invalid UnifiedV15StepRecord {key}")
        if state["anchor_started"] is not False:
            raise ValueError("UnifiedV15StepRecord cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            val = state[key]
            if key == "lr_last" and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"invalid UnifiedV15StepRecord {key}")
        _validate_unified_v15_packet(state["unified_v15"])
        super().load_state_dict({key: state[key] for key in CriticStepRecord.KEYS})
        self._restore_unified_v15_packet(state["unified_v15"])

    def __deepcopy__(self, memo: Dict[int, Any]) -> "UnifiedV15StepRecord":
        dup = UnifiedV15StepRecord()
        memo[id(self)] = dup
        dup.lr_max = self.lr_max
        dup.lr_last = self.lr_last
        dup.anchor_started = self.anchor_started
        dup.calls = self.calls
        dup.observed_steps = self.observed_steps
        self._copy_unified_v15_to(dup)
        return dup


def _unified_v15_critic_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> Any:
    """Instance-bound critic forward hook capturing live endpoints, secant probes, chart inputs, and thermal cooling."""
    rec = getattr(module, "_unified_v15_record", getattr(module, "_t1c_record", None))
    if rec is not None and getattr(rec, "_in_probe", False):
        return None
    x_first = args[0] if (args and isinstance(args[0], torch.Tensor)) else None

    if rec is not None and getattr(rec, "_in_penalty", False):
        if x_first is not None:
            rec._last_probe_actual_input = x_first.detach()
        return None

    raw_target = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(raw_target, torch.Tensor) or raw_target.ndim < 1:
        return None

    if (
        rec is not None
        and x_first is not None
        and module.training
        and torch.is_grad_enabled()
        and raw_target.requires_grad
        and not x_first.requires_grad
        and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in args[1:])
        and not any(isinstance(t, torch.Tensor) and t.requires_grad for t in kwargs.values())
    ):
        src_key = getattr(module, "_t2_pending_source_key", None) or _tensor_identity_key(x_first)
        if src_key is not None:
            unscaled = raw_target.float() if raw_target.dtype == torch.bfloat16 else raw_target
            x_act = x_first.detach()
            captures = getattr(rec, "_live_captures", None)
            if captures is None:
                captures = []
                rec._live_captures = captures
            c_sig = _cond_signature(args, kwargs)
            step_tag = int(getattr(rec, "observed_steps", 0))
            alive = [True]
            entry: List[Any] = [step_tag, src_key, c_sig, x_act, unscaled, alive]
            raw_target.register_hook(lambda _g, _a=alive: _a.clear())
            captures.append(entry)
            _trim_live_captures(captures)

        if (
            getattr(rec, "_enable_secant_probe", True)
            and x_first.ndim >= 2
            and not _has_spatial_dims(x_first)
            and x_first.shape[0] >= 1
            and len(args) == 1
            and not kwargs
        ):
            b_full = int(x_first.shape[0])
            in_dim = max(1, int(x_first[0].numel()))
            b_p = _probe_batch_count(b_full, in_dim)
            p_idx = _select_probe_indices(x_first, b_full, b_p)
            p_args = tuple(_index_probe_arg(a, b_full, p_idx) for a in args)
            raw_slice = (
                raw_target.detach().index_select(0, p_idx.to(device=raw_target.device)).to(dtype=torch.float32)
                if p_idx is not None
                else raw_target.detach().to(dtype=torch.float32)
            )
            if _module_needs_eval_probe_baseline(module, p_args, {}, raw_slice, rec):
                d_pre = _eval_v15_probe_forward(module, p_args, {}, rec)
            else:
                d_pre = raw_slice
            if d_pre is not None:
                probes = getattr(rec, "_d_probe_list", None)
                if probes is None:
                    probes = []
                    rec._d_probe_list = probes
                probes.append((p_args, {}, d_pre))
                if len(probes) > _MAX_D_PROBES:
                    del probes[: len(probes) - _MAX_D_PROBES]

    modified = False
    target = raw_target
    if rec is not None and torch.is_grad_enabled():
        s = float(getattr(rec, "_thermal_scale", 1.0))
        if s < 1.0 and target.requires_grad:
            scaled = target * s + target.detach() * (1.0 - s)
            scaled._t1c_thermally_scaled = True
            if isinstance(output, torch.Tensor):
                output = scaled
                modified = True
            elif isinstance(output, tuple):
                output = (scaled, *output[1:])
                modified = True
            elif isinstance(output, list):
                output = [scaled, *output[1:]]
                modified = True
            target = scaled

    max_t1c_dim = 4096 if (x_first is not None and _has_spatial_dims(x_first)) else 1024
    if (
        x_first is not None
        and x_first.ndim >= 2
        and x_first.shape[0] > 0
        and x_first[0].numel() > max_t1c_dim
        and len(args) == 1
        and not kwargs
    ):
        target._t1c_bypassed = True
        if rec is not None:
            target._t1c_record = rec
        return output if modified else None

    x_in, cond_disc, has_cont_cond, float_parts = _extract_t1c_inputs_parts(
        args, kwargs, int(target.shape[0])
    )
    if x_in is not None:
        _record_t1c_capture(
            output,
            x_in,
            cond_disc,
            rec,
            has_cont_cond=has_cont_cond,
            float_parts=float_parts,
            overwrite=True,
        )
    return output if modified else None


def _module_needs_eval_probe_baseline(
    module: nn.Module,
    p_args: Optional[Tuple[Any, ...]] = None,
    p_kwargs: Optional[Dict[str, Any]] = None,
    raw_slice: Optional[torch.Tensor] = None,
    record: Any = None,
) -> bool:
    cached = getattr(module, "_unified_v15_has_bn", getattr(module, "_unified_v15_gen_has_bn", None))
    if cached is True:
        return True
    if any(
        isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
        for m in module.modules()
    ):
        return True
    verified = getattr(module, "_unified_v15_eval_verified", None)
    if isinstance(verified, bool):
        return verified
    if p_args is not None and p_kwargs is not None and raw_slice is not None and module.training:
        y_eval = _eval_v15_probe_forward(module, p_args, p_kwargs, record)
        if y_eval is not None and y_eval.shape == raw_slice.shape:
            differs = not bool(torch.allclose(y_eval, raw_slice, atol=1e-4, rtol=1e-3))
            module._unified_v15_eval_verified = differs
            return differs
    return False


def record_unified_v15_generator_probe(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
    record: Any,
    *,
    require_module_training: bool = True,
) -> bool:
    """Capture a stratified element-budgeted generator probe for Unified-v15 function-space secant regulation.

    Supports single-tensor and multi-input/composite generator modules whose ``forward(*args, **kwargs)``
    replays the full generator computation from sampled latent (and conditioning) to generated output.
    """
    if (
        record is None
        or getattr(record, "_in_probe", False)
        or getattr(record, "_in_penalty", False)
        or getattr(record, "_in_protected_probe", False)
        or not getattr(record, "_enable_secant_probe", True)
        or (require_module_training and not module.training)
        or not torch.is_grad_enabled()
    ):
        return False
    raw_out = output[0] if isinstance(output, (tuple, list)) and output else output
    if (
        not isinstance(raw_out, torch.Tensor)
        or not raw_out.is_floating_point()
        or not raw_out.requires_grad
        or raw_out.ndim < 2
        or raw_out.shape[0] < 1
    ):
        return False
    import torch.autograd.forward_ad as fwAD
    if fwAD.unpack_dual(raw_out).tangent is not None:
        return False
    b_full = int(raw_out.shape[0])
    out_dim = max(1, int(raw_out[0].numel()))
    if (
        not _has_spatial_dims(raw_out)
        and 16 < out_dim <= 1024
        and len(args) == 1
        and not kwargs
        and isinstance(args[0], torch.Tensor)
        and args[0].ndim >= 2
        and not _has_spatial_dims(args[0])
        and args[0].shape[0] == b_full
    ):
        record._g_last_latent = (module, args[0].detach())
    b_p = _probe_batch_count(b_full, out_dim)
    p_idx = _select_probe_indices(raw_out, b_full, b_p)
    p_args = tuple(_index_probe_arg(a, b_full, p_idx) for a in args)
    p_kwargs = {k: _index_probe_arg(v, b_full, p_idx) for k, v in kwargs.items()}
    raw_slice = (
        raw_out.detach().index_select(0, p_idx.to(device=raw_out.device)).to(dtype=torch.float32)
        if p_idx is not None
        else raw_out.detach().to(dtype=torch.float32)
    )
    if _module_needs_eval_probe_baseline(module, p_args, p_kwargs, raw_slice, record):
        y_pre = _eval_v15_probe_forward(module, p_args, p_kwargs, record)
        if y_pre is None:
            return False
    else:
        y_pre = raw_slice
    record._g_probe_cache = (module, p_args, p_kwargs, y_pre)
    return True


def _unified_v15_generator_forward_hook(
    module: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
) -> None:
    """Capture a dimension-agnostic element-budgeted generator slice for function-space secant trust-band regulation."""
    rec = getattr(module, "_unified_v15_record", None)
    if not getattr(module, "_unified_v15_allow_multi_input", False) and (len(args) != 1 or bool(kwargs)):
        return
    record_unified_v15_generator_probe(module, args, kwargs, output, rec, require_module_training=True)


def _attach_unified_v15_critic_hook(critic: nn.Module | None, record: CriticStepRecord) -> None:
    """Attach instance-local unified_v15 forward hook to ``critic`` (zero global state)."""
    if not hasattr(record, "_in_penalty"):
        record._in_penalty = False
    if not hasattr(record, "_in_protected_probe"):
        record._in_protected_probe = False
    if not hasattr(record, "_in_probe"):
        record._in_probe = False
    if not hasattr(record, "_capture_ring"):
        record._capture_ring = OrderedDict()
    if not hasattr(record, "_live_captures"):
        record._live_captures = []
    if not hasattr(record, "_d_probe_list"):
        record._d_probe_list = []
    if getattr(record, "legacy_thermal_channel", True):
        _LEGACY_THERMAL_CHANNEL.clear()
    if not isinstance(critic, nn.Module):
        return
    critic._unified_v15_record = record
    critic._unified_record = record
    critic._t1c_record = record
    critic._t2_record = record
    critic._unified_v15_has_bn = any(
        isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
        for m in critic.modules()
    )
    if getattr(critic, "_unified_v15_hook_registered", False):
        return
    critic.register_forward_hook(_unified_v15_critic_forward_hook, with_kwargs=True)
    critic._unified_v15_hook_registered = True


def _attach_unified_v15_generator_hook(
    generator: nn.Module | None,
    record: CriticStepRecord,
    *,
    allow_multi_input: bool = False,
) -> None:
    """Attach instance-local unified_v15 secant probe hook to ``generator``."""
    if not isinstance(generator, nn.Module):
        return
    generator._unified_v15_record = record
    generator._unified_v15_allow_multi_input = bool(allow_multi_input)
    generator._unified_v15_gen_has_bn = any(
        isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd))
        for m in generator.modules()
    )
    if getattr(generator, "_unified_v15_gen_hook_registered", False):
        return
    generator.register_forward_hook(_unified_v15_generator_forward_hook, with_kwargs=True)
    generator._unified_v15_gen_hook_registered = True


def bind_unified_v15_record(record: Any, *consumers: Any) -> None:
    """Bind instance-local ``unified_v11`` critic record to losses, optimizers, and priors."""
    if record is not None:
        record.legacy_thermal_channel = False
    for consumer in consumers:
        if consumer is None:
            continue
        if hasattr(consumer, "record"):
            consumer.record = record
        if hasattr(consumer, "thermal_source"):
            consumer.thermal_source = record
        if hasattr(consumer, "empf_record"):
            consumer.empf_record = record
        if hasattr(consumer, "unified_record"):
            consumer.unified_record = record
        if hasattr(consumer, "unified_v15_record"):
            consumer.unified_v15_record = record


class UnifiedV15CriticAdam(K3PCriticAdam):
    """Formulation critic Adam for ``critic_formulation='unified_v15'`` (zero EMA critic copy)."""

    def __init__(
        self,
        params: Any,
        *,
        critic: nn.Module | None = None,
        ema_critic: nn.Module | None = None,
        guard_ratio: float = 0.0,
        guard_min_steps: int = 200,
        **adam_kwargs: Any,
    ) -> None:
        del ema_critic
        super().__init__(
            params,
            critic=critic,
            ema_critic=None,
            anchor_decay=0.0,
            guard_ratio=guard_ratio,
            guard_min_steps=guard_min_steps,
            **adam_kwargs,
        )
        self.record = UnifiedV15StepRecord()
        self._t1c_record = self.record
        self._t2_record = self.record
        self._unified_record = self.record
        self._unified_v15_record = self.record
        _attach_unified_v15_critic_hook(critic, self.record)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        if self.param_groups and self.param_groups[0]["params"]:
            self.record.restore_device_state(self.param_groups[0]["params"][0].device)


__all__ = [
    "UnifiedV15CriticAdam",
    "UnifiedV15HighDBasinBank",
    "UnifiedV15SparseObsBank",
    "UnifiedV15StepRecord",
    "_UnifiedV15ObserverStepRecord",
    "_attach_unified_v15_critic_hook",
    "_attach_unified_v15_generator_hook",
    "_cond_signature",
    "_eval_v15_probe_forward",
    "_extract_t1c_inputs_parts",
    "_lookup_t1c_capture_parts",
    "_record_t1c_capture",
    "_tensor_identity_key",
    "_validate_unified_v15_packet",
    "bind_unified_v15_record",
    "record_unified_v15_generator_probe",
    "validate_v15_highd_bank_packet",
    "validate_v15_sparse_obs_bank_packet",
]
