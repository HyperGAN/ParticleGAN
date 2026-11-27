"""Deterministic network initialization, in the style of ``torch.nn.init``.

``deterministic_orthogonal_(module, seed=0)`` rewrites a module's trainable
parameters in place. Each supported layer declares the distribution PyTorch's
own constructor draws from (``Uniform``/``Normal``); a matrix becomes an
orthogonal matrix at that distribution's RMS, a vector becomes a fixed pattern
with its mean and std, and a particle table becomes R2 low-discrepancy points
(``R2Normal``). Values come from hashing ``(seed, parameter index, shape)`` in
float64 on the CPU, so no RNG state is read or consumed and results do not
depend on device.

Layers declare their parameters with :func:`register`. Parameters no
declaration covers raise by default (``strict=True``), so a custom layer is
never silently left on a different init.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping, Union

import torch
from torch import nn

from . import _qr

__all__ = ["Uniform", "Normal", "R2Normal", "KEEP", "register", "declarations",
           "deterministic_orthogonal_", "initialize_"]


@dataclass(frozen=True)
class Uniform:
    """The constructor's U(low, high): matrices orthogonal at its RMS, vectors a pattern."""
    low: float
    high: float


@dataclass(frozen=True)
class Normal:
    """The constructor's N(mean, std**2): matrices orthogonal at its RMS, vectors a pattern."""
    mean: float = 0.0
    std: float = 1.0


@dataclass(frozen=True)
class R2Normal:
    """Particle table: each row is one R2 point mapped through the N(mean, std**2) quantile."""
    mean: float = 0.0
    std: float = 1.0


class _Keep:
    def __repr__(self):
        return "KEEP"


KEEP = _Keep()
"""Declare a parameter whose constructor value is deliberate (norm scales, temperatures)."""

Spec = Union[Uniform, Normal, R2Normal, _Keep]
Declarations = Union[Mapping[str, Spec], Callable[[nn.Module], Mapping[str, Spec]]]
_REGISTRY: dict[type, tuple[Declarations, Callable[[nn.Module], None] | None]] = {}


def register(cls: type, declarations: Declarations = None, *,
             finalize: Callable[[nn.Module], None] | None = None) -> None:
    """Declare how ``deterministic_orthogonal_`` treats ``cls``'s own parameters.

    ``declarations`` maps each direct parameter name (not a submodule's) to a
    spec: ``Uniform``, ``Normal``, ``R2Normal`` or ``KEEP``. Pass a callable
    ``module -> mapping`` when the spec depends on the instance, e.g. its fan-in.
    Declarations merge along the class hierarchy, so a subclass declares only
    the parameters it adds (or overrides). ``finalize(module)`` runs without
    grad after the call writes any parameter inside the module, for layout
    fix-ups such as zeroing an embedding padding row. Re-registering a class
    replaces its entry.
    """
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError("register expects an nn.Module subclass")
    if declarations is None:
        declarations = {}
    if not callable(declarations) and not isinstance(declarations, Mapping):
        raise TypeError("declarations must be a mapping or a callable returning one")
    _REGISTRY[cls] = (declarations, finalize)


def _resolve(module):
    merged, finalizers = {}, []
    for cls in reversed(type(module).__mro__):
        if cls not in _REGISTRY:
            continue
        entry, finalize = _REGISTRY[cls]
        found = entry(module) if callable(entry) else entry
        for name, spec in found.items():
            if not isinstance(spec, (Uniform, Normal, R2Normal, _Keep)):
                raise TypeError(f"{cls.__name__} declares {name!r} as {spec!r}; "
                                "use Uniform, Normal, R2Normal or KEEP")
            if not hasattr(module, name):
                raise ValueError(f"{cls.__name__} declares {name!r}, which "
                                 f"{type(module).__name__} does not have")
        merged.update(found)
        if finalize is not None:
            finalizers.append(finalize)
    return merged, finalizers


def _plan(module):
    if not isinstance(module, nn.Module):
        raise TypeError("expected an nn.Module")
    owners, finalizers, holders = {}, [], {}  # holders: parameter id -> owning class name
    for child in module.modules():
        if any(isinstance(p, nn.parameter.UninitializedParameter)
               for p in child.parameters(recurse=False)):
            raise ValueError("materialize lazy layers before initializing them")
        declared, finalize = _resolve(child)
        if finalize:
            finalizers.append((child, finalize))
        for name, parameter in child.named_parameters(recurse=False):
            holders.setdefault(id(parameter), type(child).__name__)
            if name in declared:
                owners.setdefault(id(parameter), declared[name])
    return owners, finalizers, holders


def declarations(module: nn.Module) -> dict[str, Spec | None]:
    """Resolved spec for every trainable parameter; ``None`` marks an undeclared one.

    Changes nothing. Use it to check a custom network before initializing it.
    """
    owners, _, _ = _plan(module)
    return {path: owners.get(id(parameter))
            for path, parameter in module.named_parameters() if parameter.requires_grad}


def _keep_constant(parameter):
    # Deliberate constructor values: zero vectors, constant or identity matrices.
    if parameter.ndim < 2:
        return bool(torch.all(parameter == 0))
    if (parameter.numel() > 1 or bool(torch.all(parameter == 0))) and bool(
            torch.all(parameter == parameter.flatten()[0])):
        return True
    return (parameter.ndim == 2 and parameter.shape[0] == parameter.shape[1]
            and torch.equal(parameter, torch.eye(parameter.shape[0],
                                                device=parameter.device, dtype=parameter.dtype)))


def _moments(spec):
    if isinstance(spec, Uniform):
        a, b = spec.low, spec.high
        return math.sqrt((a * a + a * b + b * b) / 3), (a + b) / 2, (b - a) / math.sqrt(12)
    return math.sqrt(spec.mean ** 2 + spec.std ** 2), spec.mean, spec.std


def _draw(spec, key, shape):
    if isinstance(spec, R2Normal):
        if len(shape) != 2:
            raise ValueError("R2Normal declares a 2-D particle table")
        return spec.mean + spec.std * torch.special.ndtri(_qr.r2_points(*shape))
    rms, mean, std = _moments(spec)
    if len(shape) < 2:
        return _qr.pattern(key, shape, mean, std)
    rows, cols = shape[0], math.prod(shape[1:])
    return (_qr.semi_orthogonal(key, rows, cols) * rms * math.sqrt(max(rows, cols))).reshape(shape)


@torch.no_grad()
def deterministic_orthogonal_(module: nn.Module, *, seed: int = 0, strict: bool = True,
                              parameter_seeds: Mapping[str, int] | None = None) -> nn.Module:
    """Initialize ``module``'s trainable parameters in place and return it.

    Matrices become orthogonal at the RMS of the distribution their layer
    declares (PyTorch's default scale); vectors become a deterministic pattern
    with that distribution's mean and std; particle tables become R2 points.
    ``seed`` selects the values: the same seed and architecture give the same
    weights, so give networks with matching shapes different seeds (the
    examples use G=0, D=1, E=2). R2 tables do not depend on the seed.

    Values also depend on each parameter's position in
    ``module.named_parameters()`` of the module passed: a submodule initialized
    on its own gets different values than inside its whole network, and
    adding or reordering parameters shifts later values. Initialize each whole
    network in one call. For component-isolated comparisons, ``parameter_seeds``
    supplies an explicit seed for every trainable parameter name. In that mode
    its positional index is zero: inserting unrelated parameters cannot shift
    shared parameter draws. Shape and declared distribution still affect values.

    Kept as-is: frozen parameters, buffers, ``KEEP`` declarations, zero
    vectors and constant or identity matrices the constructor set on purpose.
    With ``strict=True`` any other trainable parameter without a declaration
    raises before anything is written; declare it with :func:`register`, or
    pass ``strict=False`` to leave it on its constructor's init.

    Call it on a fresh network, before loading weights or building optimizers.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    owners, finalizers, holders = _plan(module)
    trainable = [(i, path, p) for i, (path, p) in enumerate(module.named_parameters())
                 if p.requires_grad and p.numel()]
    if parameter_seeds is not None:
        if (not isinstance(parameter_seeds, Mapping)
                or set(parameter_seeds) != {path for _, path, _ in trainable}
                or any(type(value) is not int or value < 0 for value in parameter_seeds.values())):
            raise ValueError("parameter_seeds must provide one nonnegative integer per trainable parameter")
    missing = [f"{path!r} ({holders[id(p)]})" for _, path, p in trainable if id(p) not in owners]
    if strict and missing:
        shown = ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else "")
        raise ValueError(
            f"deterministic_orthogonal_ has no declaration for {len(missing)} trainable "
            f"parameter(s): {shown}. Declare them with particlegan.init.register(<layer class>, "
            f"{{name: Uniform/Normal/R2Normal/KEEP}}), or pass strict=False to leave them as-is.")
    changed = set()
    for index, path, parameter in trainable:
        spec = owners.get(id(parameter))
        if spec is None or spec is KEEP or (not isinstance(spec, R2Normal) and _keep_constant(parameter)):
            continue
        key = _qr.key(seed, index, tuple(parameter.shape)) if parameter_seeds is None else \
              _qr.key(parameter_seeds[path], 0, tuple(parameter.shape))
        value = _draw(spec, key, tuple(parameter.shape))
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        changed.add(id(parameter))
    for child, finalize in finalizers:
        if any(id(p) in changed for p in child.parameters()):
            for step in finalize:
                step(child)
    return module


@torch.no_grad()
def initialize_(module: nn.Module, *, method: str,
                parameter_generators: Mapping[str, torch.Generator] | None = None,
                distributions: Mapping[str, Spec] | None = None,
                gain: float = 1.0, strict: bool = True) -> nn.Module:
    """Apply an explicit initialization method through public PyTorch operations.

    ``identity_linear_v1`` writes identity/zero to a square Linear's trainable
    weight/bias. ``xavier_uniform_zero_bias_v1`` initializes Linear weights with
    Xavier uniform and zeros their biases. ``sample_distributions_v1`` literally
    samples the registry's Uniform/Normal distributions, with call-local full-name
    overrides; KEEP remains untouched and R2Normal needs an explicit override.

    Each randomly drawn parameter requires a distinct CPU Generator under its
    full name. The mapping must contain exactly those names, excluding constants,
    frozen and empty parameters. Draws use CPU scratch in the parameter's dtype,
    independent of its destination device; no global RNG is used. Distribution
    widths/std and Xavier gain must be finite and positive.

    Trainable parameters must be contiguous: this conservatively excludes
    internally overlapping views whose in-place commit could fail. The global
    default generator is not an owned parameter stream and is rejected.

    Complete input validation and staged finalization precede caller mutation.
    Finalizers may only alter selected trainable parameters, never frozen values,
    unselected parameters or buffers. A validation/finalizer failure leaves the
    caller's tensors and supplied RNG states unchanged. Calibrated MoG width
    updates require a separate explicit operation; explicit-sigma MoG is supported.
    ``strict=False`` only permits undeclared parameters in the sampling method.
    Returns the same module. Existing deterministic_orthogonal_ is unchanged.
    """
    methods = {"identity_linear_v1", "xavier_uniform_zero_bias_v1", "sample_distributions_v1"}
    if not isinstance(method, str) or method not in methods:
        raise ValueError("unknown initialization method")
    if type(strict) is not bool:
        raise TypeError("strict must be a bool")
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    if method != "xavier_uniform_zero_bias_v1" and gain != 1.0:
        raise ValueError("gain only applies to Xavier initialization")
    owners, _, _ = _plan(module)
    parameters = dict(module.named_parameters())
    trainable = {name: p for name, p in parameters.items() if p.requires_grad and p.numel()}
    # _plan/named_parameters normally deduplicate aliases; this API refuses them
    # so different names cannot give one tensor conflicting policies or streams.
    seen, storage = set(), set()
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            raise ValueError(f"aliased parameter is unsupported: {name}")
        seen.add(id(parameter))
        if parameter.layout != torch.strided or parameter.device.type == "meta":
            raise ValueError(f"unsupported parameter storage: {name}")
        if parameter.numel():
            key = (str(parameter.device), parameter.untyped_storage().data_ptr())
            if key in storage:
                raise ValueError(f"shared parameter storage is unsupported: {name}")
            storage.add(key)
    for name, buffer in module.named_buffers():
        if buffer.layout != torch.strided or buffer.device.type == "meta":
            raise ValueError(f"unsupported buffer storage: {name}")
        if buffer.numel() and (str(buffer.device), buffer.untyped_storage().data_ptr()) in storage:
            raise ValueError(f"parameter/buffer shared storage is unsupported: {name}")
    if distributions is not None and not isinstance(distributions, Mapping):
        raise TypeError("distributions must be a full-parameter-name mapping")
    overrides = dict(distributions or {})
    if overrides and method != "sample_distributions_v1":
        raise ValueError("distribution overrides only apply to sampled distributions")
    if any(not isinstance(name, str) or name not in trainable for name in overrides):
        raise ValueError("distribution overrides must name nonempty trainable parameters")
    operations = {}
    if method == "identity_linear_v1":
        if not isinstance(module, nn.Linear) or module.in_features != module.out_features:
            raise ValueError("identity initialization requires one square Linear")
        if set(trainable) - {"weight", "bias"}:
            raise ValueError("identity Linear has unsupported trainable parameters")
    holders = {id(p): (child, name) for child in module.modules()
               for name, p in child.named_parameters(recurse=False)}
    for name, parameter in trainable.items():
        if not parameter.is_contiguous():
            raise ValueError(f"initialization requires contiguous trainable parameters: {name}")
        if not parameter.is_floating_point() or parameter.is_complex():
            raise ValueError(f"initialization requires real floating parameters: {name}")
        if method in {"identity_linear_v1", "xavier_uniform_zero_bias_v1"}:
            child, local = holders[id(parameter)]
            if not isinstance(child, nn.Linear) or local not in {"weight", "bias"}:
                raise ValueError(f"Linear initialization has unsupported trainable parameter: {name}")
            shape = (child.out_features, child.in_features) if local == "weight" else (child.out_features,)
            if tuple(parameter.shape) != shape:
                raise ValueError(f"Linear parameter shape differs from its declaration: {name}")
            operations[name] = ("zero" if local == "bias" else
                                "identity" if method == "identity_linear_v1" else "xavier", None)
            continue
        spec = overrides.get(name, owners.get(id(parameter)))
        if spec is None and not strict:
            continue
        if spec is KEEP:
            continue
        if not isinstance(spec, (Uniform, Normal)):
            raise ValueError(f"sampled parameter {name!r} requires Uniform, Normal or KEEP; override R2Normal/undeclared parameters")
        values = (spec.low, spec.high) if isinstance(spec, Uniform) else (spec.mean, spec.std)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"distribution bounds must be finite numbers: {name}")
        if (isinstance(spec, Uniform) and spec.low >= spec.high) or (isinstance(spec, Normal) and spec.std <= 0):
            raise ValueError(f"sampled distribution width/std must be positive: {name}")
        if any(abs(v) > torch.finfo(parameter.dtype).max for v in values):
            raise ValueError(f"distribution is outside parameter dtype range: {name}")
        operations[name] = ("uniform" if isinstance(spec, Uniform) else "normal", spec)
    if parameter_generators is not None and not isinstance(parameter_generators, Mapping):
        raise TypeError("parameter_generators must be a full-parameter-name mapping")
    generators = dict(parameter_generators or {})
    random_names = {name for name, (operation, _) in operations.items()
                    if operation in {"xavier", "uniform", "normal"}}
    if set(generators) != random_names:
        raise ValueError("parameter_generators must match exactly the randomly drawn parameters")
    if any(not isinstance(g, torch.Generator) or g.device.type != "cpu" for g in generators.values()):
        raise ValueError("every parameter generator must be a CPU torch.Generator")
    if any(g is torch.default_generator for g in generators.values()):
        raise ValueError("the global default generator is unsupported; supply owned parameter streams")
    if len({id(g) for g in generators.values()}) != len(generators):
        raise ValueError("each random parameter requires a distinct generator")
    from .particle_prior import MoGParticlePrior
    for child in module.modules():
        if isinstance(child, MoGParticlePrior) and any(id(p) == id(child.z) for n, p in trainable.items() if n in operations):
            if child.sigma_rel > 0 or child.d0.item() > 0:
                raise ValueError("calibrated MoG initialization would change buffers; use an explicit-sigma prior")
    if not operations:
        return module

    staged = deepcopy(module).cpu()
    staged_parameters = dict(staged.named_parameters())
    clones = {name: torch.Generator(device="cpu").set_state(g.get_state()) for name, g in generators.items()}
    for name, (operation, spec) in operations.items():
        value = staged_parameters[name]
        if operation == "identity":
            nn.init.eye_(value)
        elif operation == "zero":
            nn.init.zeros_(value)
        elif operation == "xavier":
            nn.init.xavier_uniform_(value, gain=gain, generator=clones[name])
        elif operation == "uniform":
            nn.init.uniform_(value, spec.low, spec.high, generator=clones[name])
        else:
            nn.init.normal_(value, spec.mean, spec.std, generator=clones[name])
    _, finalizers, _ = _plan(staged)
    changed = {id(staged_parameters[name]) for name in operations}
    with torch.random.fork_rng(devices=[]):
        global_state = torch.get_rng_state()
        for child, callbacks in finalizers:
            if any(id(p) in changed for p in child.parameters()):
                for callback in callbacks:
                    callback(child)
        if not torch.equal(global_state, torch.get_rng_state()):
            raise ValueError("initialization finalizer consumed the global RNG")

    def equal_bytes(left, right):
        return (left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))

    final_parameters, buffers = dict(staged.named_parameters()), dict(module.named_buffers())
    final_buffers = dict(staged.named_buffers())
    if set(final_parameters) != set(parameters) or set(final_buffers) != set(buffers):
        raise ValueError("initialization finalizer changed the parameter/buffer structure")
    for name, original in parameters.items():
        value = final_parameters[name]
        if value.shape != original.shape or value.dtype != original.dtype or value.requires_grad != original.requires_grad:
            raise ValueError(f"initialization finalizer changed parameter metadata: {name}")
        if name not in operations and not equal_bytes(original, value):
            raise ValueError(f"initialization finalizer changed an unselected/frozen parameter: {name}")
        if name in operations and not torch.isfinite(value).all():
            raise ValueError(f"initialization produced nonfinite values: {name}")
    if any(not equal_bytes(value, final_buffers[name]) for name, value in buffers.items()):
        raise ValueError("initialization finalizer changed a buffer")
    for name in operations:
        parameters[name].copy_(final_parameters[name])
    for name, generator in generators.items():
        generator.set_state(clones[name].get_state())
    return module


# Built-in declarations: PyTorch's own constructor distributions.
def _fan_in_uniform(module):
    fan_in = math.prod(module.weight.shape[1:])
    bound = 1 / math.sqrt(fan_in) if fan_in else 0.
    return {"weight": Uniform(-bound, bound), "bias": Uniform(-bound, bound)}


for _cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
             nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d):
    register(_cls, _fan_in_uniform)


def _zero_padding_row(module):
    if module.padding_idx is not None:
        module.weight[module.padding_idx].zero_()


register(nn.Embedding, {"weight": Normal(0., 1.)}, finalize=_zero_padding_row)


def _attention(module):
    result = {"in_proj_bias": KEEP}
    for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
        weight = getattr(module, name)
        if weight is not None:
            bound = math.sqrt(6 / sum(weight.shape))
            result[name] = Uniform(-bound, bound)
    for name in ("bias_k", "bias_v"):
        weight = getattr(module, name)
        if weight is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
            result[name] = Normal(0., math.sqrt(2 / (fan_in + fan_out)))
    return result


register(nn.MultiheadAttention, _attention)
def _norm_declarations(module):
    return {name: KEEP for name in ("weight", "bias") if getattr(module, name, None) is not None}


for _cls in (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm,
             nn.modules.instancenorm._InstanceNorm, nn.RMSNorm):
    register(_cls, _norm_declarations)
register(nn.PReLU, {"weight": KEEP})


def _register_particlegan():
    from .bispf import StratifiedBlueNoisePrior
    from .diffusion import DrawSource
    from .discriminators import BatchDistanceDiscriminator
    from .particle_prior import MoGParticlePrior, ParticlePrior, calibrate_mog_sigma

    def zero_batch_readout(module):
        count = module.scales.numel()
        if module.head.in_features != module.layers[-1].out_features + count:
            raise ValueError("unsupported batch-distance feature layout")
        module.head.weight[:, -count:].zero_()

    def recalibrate(prior):
        # Calibrated spacing (d0 > 0, even at sigma_rel=0) follows the new means;
        # an explicit sigma is kept.
        if prior.sigma_rel > 0 or prior.d0 > 0:
            sigma, d0 = calibrate_mog_sigma(prior.means(), prior.sigma_rel)
            prior.set_sigma(sigma)
            prior.d0.copy_(d0)

    register(ParticlePrior, lambda prior: {"z": R2Normal(0., prior.init_std)})
    register(StratifiedBlueNoisePrior, lambda prior: {"z": R2Normal(0., prior.init_std)})
    register(MoGParticlePrior, finalize=recalibrate)
    register(DrawSource, {"table": R2Normal(0., 1.)})
    register(BatchDistanceDiscriminator, finalize=zero_batch_readout)


_register_particlegan()
