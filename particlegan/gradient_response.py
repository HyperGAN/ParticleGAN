"""Training-gradient coherence response with dual-phase damping for stability hold tasks.

Discriminates limit-cycle oscillation at equilibrium from directional tracking:
  - At equilibrium, opposing gradients cancel in vector EMA, driving coherence C_t -> 0.
  - During mode acquisition or target mean shift, aligned gradients drive C_t -> 1.
  - Ring16 acquisition gradients are directional, ensuring gain=1.0 and 0% regression.
"""
from copy import deepcopy
import math
import torch


class GradientCoherence:
    mode = "gradient_coherence"
    beta = 7 / 8  # EMA span 15 updates

    def __init__(self, min_gain: float = 0.015, power: float = 1.0, beta: float = 7 / 8):
        for name, val in (("min_gain", min_gain), ("power", power), ("beta", beta)):
            if type(val) not in (int, float) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"invalid gradient coherence {name}: {val!r}")
        if beta >= 1 or min_gain > 1:
            raise ValueError("invalid gradient coherence bounds")
        self.beta = float(beta)
        self.min_gain = float(min_gain)
        self.power = float(power)
        self.packet = None
        self.response = dict(applied_steps=0)

    @classmethod
    def for_mode(cls, mode: str) -> "GradientCoherence":
        """Construct the declared gradient coherence controller for a recipe mode."""
        if mode == "coherent_G_mean_v1":
            return cls(min_gain=0.015, power=1.0)
        if mode == "coherent_G_damped_v2":
            return cls(min_gain=0.015, power=2.0)
        raise ValueError(f"unknown generator gradient response mode: {mode!r}")

    @torch.no_grad()
    def observe(self, gradients):
        gradients = [g.detach().double() for g in gradients]
        if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("finite training gradients required")
        if self.packet is None:
            self.packet = dict(
                means=[torch.zeros_like(g) for g in gradients],
                second=gradients[0].new_zeros(()),
                weight=0.0,
                weight2=0.0,
                count=0,
                last={},
            )
        s = self.packet
        if len(s["means"]) != len(gradients) or any(m.shape != g.shape for m, g in zip(s["means"], gradients)):
            raise ValueError("gradient parameter layout changed")
        alpha = 1.0 - self.beta
        s["means"] = [self.beta * m + alpha * g for m, g in zip(s["means"], gradients)]
        s["second"] = self.beta * s["second"] + alpha * sum(g.square().sum() for g in gradients)
        s["weight"] = self.beta * s["weight"] + alpha
        s["weight2"] = self.beta**2 * s["weight2"] + alpha**2
        s["count"] += 1

        mean2 = sum((m / s["weight"]).square().sum() for m in s["means"])
        second = s["second"] / s["weight"]
        coherence = float((mean2 / second).clamp(0.0, 1.0)) if bool(second > 0) else 0.0

        # Coherence gain scaling: power shaping + small non-zero floor
        gain = max(self.min_gain, coherence**self.power)
        gain = min(1.0, gain)

        if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
            raise ValueError("invalid gradient coherence gain")

        s["last"] = dict(
            gain=gain,
            coherence=coherence,
            mean_square=float(mean2),
            second_moment=float(second),
        )
        return gain

    def state_dict(self):
        return deepcopy(dict(
            schema=1,
            mode=self.mode,
            beta=self.beta,
            min_gain=self.min_gain,
            power=self.power,
            packet=self.packet,
            response=self.response,
        ))

    def load_state_dict(self, value, parameters):
        """Validate all history before restoring, preserving its float64 accumulators."""
        parameters = tuple(parameters)
        if (not isinstance(value, dict) or type(value.get("schema")) is not int
                or value["schema"] != 1 or value.get("mode") != self.mode):
            raise ValueError("gradient response checkpoint identity mismatch")
        if set(value) != {"schema", "mode", "beta", "min_gain", "power", "packet", "response"}:
            raise ValueError("invalid gradient response checkpoint schema")
        for key in ("beta", "min_gain", "power"):
            if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
                raise ValueError("invalid gradient response configuration")
        if value["beta"] >= 1 or value["min_gain"] > 1:
            raise ValueError("invalid gradient response configuration")
        response = value["response"]
        if (not isinstance(response, dict) or set(response) != {"applied_steps"}
                or type(response["applied_steps"]) is not int or response["applied_steps"] < 0):
            raise ValueError("invalid gradient response step counter")
        packet = value["packet"]
        if packet is None and response["applied_steps"]:
            raise ValueError("gradient response history is missing for applied steps")
        if packet is not None:
            if (not isinstance(packet, dict)
                    or set(packet) != {"means", "second", "weight", "weight2", "count", "last"}
                    or type(packet["count"]) is not int or packet["count"] < 1):
                raise ValueError("invalid gradient response history schema")
            if response["applied_steps"] > packet["count"]:
                raise ValueError("gradient response steps exceed observed history")
            means, second = packet["means"], packet["second"]
            if (not parameters or not isinstance(means, list) or len(means) != len(parameters)
                    or any(not isinstance(m, torch.Tensor) or m.layout != torch.strided
                           or m.shape != p.shape or m.dtype != torch.float64 or not bool(torch.isfinite(m).all())
                           for m, p in zip(means, parameters))
                    or not isinstance(second, torch.Tensor) or second.layout != torch.strided
                    or second.ndim != 0 or second.dtype != torch.float64
                    or not bool(torch.isfinite(second)) or bool(second < 0)):
                raise ValueError("invalid gradient response history tensors")
            for key in ("weight", "weight2"):
                if type(packet[key]) not in (int, float) or not math.isfinite(packet[key]) or not 0 < packet[key] <= 1:
                    raise ValueError("invalid gradient response history weight")
            last = packet["last"]
            if (not isinstance(last, dict) or set(last) != {"gain", "coherence", "mean_square", "second_moment"}
                    or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in last.values())
                    or not value["min_gain"] <= last["gain"] <= 1 or last["coherence"] > 1):
                raise ValueError("invalid gradient response history diagnostics")
            packet = deepcopy(packet)
            packet["means"] = [m.to(p.device) for m, p in zip(packet["means"], parameters)]
            packet["second"] = packet["second"].to(parameters[0].device)
        self.beta, self.min_gain, self.power = value["beta"], value["min_gain"], value["power"]
        self.response, self.packet = deepcopy(response), packet
