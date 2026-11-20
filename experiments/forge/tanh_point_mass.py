"""Origin point-mass tanh host for the zero_tanh_smoke card.

Construction, target draws and the four smoke gates live here so the frozen
vector law in ``benchmarks/transfer_suite/vector_tasks.py`` stays unchanged.
The card is an origin Dirac in R^2 with a tanh generator. It has no ``kind``
and no shifting target. Rail escape is a different law and is not served here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vectorprofiles import is_tanh_point_mass_host


class TanhPointMassGenerator(nn.Module):
    """MLP whose range is tanh, for the declared origin point-mass host.

    ``layers`` counts Linear maps. The zero_tanh_smoke card's ``layers: 2`` is
    Linear(z, hidden), LeakyReLU(0.2), Linear(hidden, 2), Tanh. That is the
    2-layer tanh head, not an extra hidden block on top of SimpleMLPGenerator.
    """

    def __init__(self, z_dim: int, hidden: int, layers: int, out_dim: int = 2) -> None:
        super().__init__()
        if type(layers) is not int or layers < 1:
            raise ValueError("tanh host layers must be a positive integer")
        blocks = []
        in_dim = z_dim
        for _ in range(layers - 1):
            blocks.append(nn.Linear(in_dim, hidden))
            blocks.append(nn.LeakyReLU(0.2, inplace=False))
            in_dim = hidden
        blocks.append(nn.Linear(in_dim, out_dim))
        blocks.append(nn.Tanh())
        self.net = nn.Sequential(*blocks)

    def forward(self, z):
        return self.net(z)


def accepts_origin_smoke(task: dict, spec: dict) -> bool:
    """True only for the published zero_tanh_smoke card and its host shape."""
    evaluation = task.get("evaluation") if isinstance(task, dict) else None
    return (isinstance(task, dict)
            and task.get("adapter") == "transfer_vector"
            and task.get("id") == "zero_tanh_smoke"
            and isinstance(evaluation, dict)
            and evaluation.get("kind") == "zero_tanh_smoke"
            and is_tanh_point_mass_host(spec))


def _valid_target_pair(value) -> bool:
    import math
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(x) in (int, float) and math.isfinite(x) for x in value))


def accepts_rail_escape(task: dict, spec: dict) -> bool:
    """True only for the published zero_tanh_rail_escape card and its shift law."""
    if not isinstance(task, dict) or task.get("adapter") != "transfer_vector":
        return False
    execution = task.get("execution")
    evaluation = task.get("evaluation")
    if not isinstance(execution, dict) or not isinstance(evaluation, dict):
        return False
    shift_step = execution.get("shift_step")
    return (task.get("id") == "zero_tanh_rail_escape"
            and evaluation.get("kind") == "zero_tanh_rail_escape"
            and type(shift_step) is int and 1 <= shift_step <= int(execution.get("steps", 0))
            and _valid_target_pair(execution.get("initial_target"))
            and _valid_target_pair(execution.get("shifted_target"))
            and is_tanh_point_mass_host(spec))


def refuse_unsupported_tanh(task: dict, spec: dict) -> None:
    """Refuse every tanh host except published zero_tanh_smoke and zero_tanh_rail_escape."""
    if not is_tanh_point_mass_host(spec) or accepts_origin_smoke(task, spec) or accepts_rail_escape(task, spec):
        return
    from .api import CapabilityError
    evaluation = task.get("evaluation") if isinstance(task, dict) else None
    kind = evaluation.get("kind") if isinstance(evaluation, dict) else None
    name = task.get("id") if isinstance(task, dict) and isinstance(task.get("id"), str) else "tanh host"
    detail = f"{name}: unsupported tanh host"
    if isinstance(kind, str):
        detail += f" {kind}"
    raise CapabilityError([
        detail + "; only zero_tanh_smoke and zero_tanh_rail_escape are routed"])


def sample_target(spec, count, rng, completed_steps):
    """Origin Dirac. The draw does not consume ``rng`` or any training stream."""
    del rng, completed_steps
    if not is_tanh_point_mass_host(spec):
        raise ValueError("origin tanh sampling applies only to the declared point-mass host")
    return torch.zeros(count, 2)


def sample_rail_escape_target(task: dict, spec: dict, count: int, rng, completed_steps: int) -> torch.Tensor:
    """Shifted point-mass target for zero_tanh_rail_escape.

    ``adapters._vector`` calls ``sample_target(..., step - 1)`` before update ``step``,
    so update ``shift_step`` (1,001) has ``completed_steps == shift_step - 1`` (1,000).
    """
    del rng
    if not accepts_rail_escape(task, spec):
        raise ValueError("rail-escape sampling applies only to zero_tanh_rail_escape")
    execution = task["execution"]
    next_step = int(completed_steps) + 1
    pair = execution["shifted_target"] if next_step >= int(execution["shift_step"]) else execution["initial_target"]
    return torch.tensor(pair, dtype=torch.float32).unsqueeze(0).expand(count, 2).clone()


def _score_point_mass(fake, center_pair):
    names = ("mean_abs_error", "rms_error", "p99_abs_error", "saturated_fraction")
    if (not torch.is_tensor(fake) or fake.ndim != 2 or fake.shape[1] != 2 or len(fake) == 0
            or not torch.isfinite(fake).all()):
        return {name: None for name in names}
    center = torch.tensor(center_pair, dtype=fake.dtype, device=fake.device).unsqueeze(0)
    error = fake - center
    absolute = error.abs()
    return {"mean_abs_error": float(absolute.mean()),
            "rms_error": float(error.square().mean().sqrt()),
            "p99_abs_error": float(torch.quantile(absolute.reshape(-1), 0.99)),
            "saturated_fraction": float((fake.abs() >= 0.99).float().mean()),
            "sample_count": int(len(fake))}


def _score_origin(fake, spec, completed_steps):
    """Four zero_tanh_smoke gates: error of the draw against the origin Dirac."""
    names = ("mean_abs_error", "rms_error", "p99_abs_error", "saturated_fraction")
    if (not torch.is_tensor(fake) or fake.ndim != 2 or fake.shape[1] != 2 or len(fake) == 0
            or not torch.isfinite(fake).all()):
        return {name: None for name in names}
    center = sample_target(spec, len(fake), torch.Generator().manual_seed(991), completed_steps).to(
        dtype=fake.dtype, device=fake.device
    )
    error = fake - center
    absolute = error.abs()
    return {"mean_abs_error": float(absolute.mean()),
            "rms_error": float(error.square().mean().sqrt()),
            "p99_abs_error": float(torch.quantile(absolute.reshape(-1), 0.99)),
            "saturated_fraction": float((fake.abs() >= 0.99).float().mean()),
            "sample_count": int(len(fake))}


@torch.no_grad()
def score_samples(fake, spec, completed_steps):
    if not is_tanh_point_mass_host(spec):
        raise ValueError("origin tanh scoring applies only to the declared point-mass host")
    return _score_origin(fake, spec, completed_steps)


@torch.no_grad()
def score_rail_escape_samples(fake, task: dict, spec: dict, completed_steps: int):
    """Score post-update draw at ``step == completed_steps`` against active rail-escape target."""
    if not accepts_rail_escape(task, spec):
        raise ValueError("rail-escape scoring applies only to zero_tanh_rail_escape")
    execution = task["execution"]
    pair = execution["shifted_target"] if int(completed_steps) >= int(execution["shift_step"]) else execution["initial_target"]
    return _score_point_mass(fake, pair)
