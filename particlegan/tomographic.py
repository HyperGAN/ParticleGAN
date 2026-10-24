"""Law 3 Tomographic Multi-Query Log-Odds and Anti-Collapse Regularizers.

This module implements the third fundamental mathematical law for under-determined,
non-injective critics:
1. symmetric_log_odds: Vectorized, numerically stable one-vs-rest log-odds
   satisfying constant shift invariance, binary anti-symmetry, and soft-cap boundedness.
2. TomographicConstraintTarget: Stateless, deterministic Cartesian-product multi-query
   target distribution sampler over R^(B x (Q * K_max)).
3. MarginalEntropyRegularizer: Batch-marginal Shannon entropy regularizer enforcing
   H(p_bar_q) >= min_entropy_ratio * ln(M_q) with squared hinge and capacity-balanced
   mode attraction.
4. MarginalEntropyManipulator: Composable prior manipulator wrapping MarginalEntropyRegularizer.
5. NormalizedFeatureCosineRepulsion: Pairwise cosine repulsion on unit-normalized features
   enforcing feature diversity across generated samples.
6. CosineRepulsionManipulator: Composable prior manipulator wrapping NormalizedFeatureCosineRepulsion.
7. evaluate_tomographic_constraints: Quantitative evaluation harness returning the 6 acceptance
   metrics: mean_target_prob, primary_accuracy, entropy_ratio, feature_diversity,
   patch_seam_ratio, and shift_sensitivity.
"""
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import PriorManipulator
from .wavefront import WavefrontManifoldPrior


def symmetric_log_odds(
    logits: Union[torch.Tensor, Sequence[torch.Tensor]],
    *,
    tau: Optional[float] = None,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]:
    r"""Computes one-vs-rest symmetric log-odds with constant shift invariance.

    For logits :math:`y \in \mathbb{R}^K` (:math:`K \ge 2`), option :math:`k` has log-odds:
        .. math::
            u_k = y_k - \ln \sum_{j \neq k} e^{y_j} = y_k - \operatorname{logsumexp}_{j \neq k}(y_j)

    When :math:`\tau > 0` is provided, the outputs are soft-capped into :math:`(-\tau, \tau)` via:
        .. math::
            \tilde{u}_k = \tau \tanh\left(\frac{u_k}{\tau}\right)

    Invariants satisfied:
    1. Constant shift invariance: :math:`u(y + c \mathbf{1}) = u(y)` for all :math:`c \in \mathbb{R}`.
    2. Binary anti-symmetry (:math:`K = 2`): :math:`u_0 = -u_1` and :math:`u_0 + u_1 = 0`.
    3. Soft-cap bounds: :math:`|u_k| \le \tau`.

    Args:
        logits: Floating-point tensor with shape `(..., K)` or sequence of tensors.
        tau: Optional positive float soft-capping threshold.

    Returns:
        Tensor of identical shape and dtype containing computed log-odds.
    """
    if isinstance(logits, (list, tuple)):
        mapped = [symmetric_log_odds(item, tau=tau) for item in logits]
        return tuple(mapped) if isinstance(logits, tuple) else mapped

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)}")
    if not logits.is_floating_point():
        raise TypeError(f"logits must be a floating-point tensor, got {logits.dtype}")

    K = logits.shape[-1]
    if K < 2:
        raise ValueError(f"symmetric_log_odds requires at least 2 classes in last dimension, got {K}")

    eye = torch.eye(K, device=logits.device, dtype=torch.bool)
    leading_shape = logits.shape[:-1]
    expanded = logits.unsqueeze(-2).expand(*leading_shape, K, K)
    masked = expanded.masked_fill(eye, float("-inf"))
    lse_other = torch.logsumexp(masked, dim=-1)
    u = logits - lse_other

    if tau is not None:
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or tau <= 0.0 or not math.isfinite(tau):
            raise ValueError(f"tau must be a positive finite number, got {tau}")
        tau_val = float(tau)
        u = tau_val * torch.tanh(u / tau_val)

    return u


class TomographicConstraintTarget:
    r"""Stateless deterministic target distribution sampler over multi-query log-odds.

    Given Q queries with mode counts :math:`M_0, \dots, M_{Q-1}`, mode combinations are
    drawn from the full Cartesian product :math:`\prod_{q=0}^{Q-1} \{0, \dots, M_q - 1\}`.
    For each mode, target probabilities :math:`p \in \Delta^{K_q - 1}` are converted to
    centered scaled logits :math:`y = \ln(p) / T` and soft-capped one-vs-rest log-odds
    :math:`u = \tau \tanh(u / \tau)`.

    Args:
        queries: Sequence of query mappings configuring questions, options, or target_modes.
        num_classes: Optional sequence or int specifying class count per query.
        num_modes: Optional sequence or int specifying mode count per query.
        tau: Soft-capping threshold (default 3.0).
        temperature: Calibrated logit temperature (default 1.0).
        jitter_std: Optional Gaussian target perturbation standard deviation (default 0.0).
        target_prob: Target mode probability (default 0.95).
        pad_to_k_max: Whether to pad queries with fewer than K_max classes with -tau (default False).
    """

    resume_stateless = True

    def __init__(
        self,
        queries: Optional[Sequence[Mapping[str, Any]]] = None,
        *,
        num_classes: Optional[Union[int, Sequence[int]]] = None,
        num_modes: Optional[Union[int, Sequence[int]]] = None,
        tau: float = 3.0,
        temperature: float = 1.0,
        jitter_std: float = 0.0,
        target_prob: float = 0.95,
        pad_to_k_max: bool = False,
    ) -> None:
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0.0:
            raise ValueError(f"tau must be a positive finite number, got {tau}")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperature must be a positive finite number, got {temperature}")
        if isinstance(jitter_std, bool) or not isinstance(jitter_std, (int, float)) or not math.isfinite(jitter_std) or jitter_std < 0.0:
            raise ValueError(f"jitter_std must be a finite nonnegative number, got {jitter_std}")
        if isinstance(target_prob, bool) or not isinstance(target_prob, (int, float)) or not math.isfinite(target_prob) or not (0.0 < target_prob < 1.0):
            raise ValueError(f"target_prob must be a float in (0.0, 1.0), got {target_prob}")

        self.tau = float(tau)
        self.temperature = float(temperature)
        self.jitter_std = float(jitter_std)
        self.target_prob = float(target_prob)
        self.pad_to_k_max = bool(pad_to_k_max)

        if queries is not None:
            if not isinstance(queries, (list, tuple)) or len(queries) == 0:
                raise ValueError("queries must be a non-empty sequence of query mappings")
            normalized_queries: List[Dict[str, Any]] = []
            num_classes_list: List[int] = []
            num_modes_list: List[int] = []
            mode_probs_list: List[torch.Tensor] = []

            for idx, q in enumerate(queries):
                if not isinstance(q, Mapping):
                    raise ValueError(f"queries[{idx}] must be a mapping")

                # Parse target modes / options
                if "target_probs" in q:
                    raw_probs = q["target_probs"]
                    probs_tensor = torch.as_tensor(raw_probs, dtype=torch.float32)
                    if probs_tensor.ndim == 1:
                        probs_tensor = probs_tensor.unsqueeze(0)
                    K_q = probs_tensor.shape[-1]
                    target_modes = list(q.get("target_modes", probs_tensor.argmax(dim=-1).tolist()))
                elif "target_modes" in q or "target_options" in q or "target_index" in q:
                    raw_modes = q.get("target_modes", q.get("target_options", q.get("target_index")))
                    if isinstance(raw_modes, int) and not isinstance(raw_modes, bool):
                        target_modes = [raw_modes]
                    elif isinstance(raw_modes, (list, tuple)) and len(raw_modes) > 0 and all(isinstance(m, int) and not isinstance(m, bool) for m in raw_modes):
                        target_modes = [int(m) for m in raw_modes]
                    else:
                        raise ValueError(f"queries[{idx}] target_modes must be an int or sequence of ints")

                    options = q.get("options")
                    if options is not None:
                        K_q = len(options)
                    elif "num_classes" in q:
                        K_q = int(q["num_classes"])
                    else:
                        K_q = max(target_modes) + 1

                    if any(m < 0 or m >= K_q for m in target_modes):
                        raise ValueError(f"queries[{idx}] target_modes entries must be in [0, {K_q})")

                    rows = []
                    other_prob = (1.0 - self.target_prob) / float(K_q - 1) if K_q > 1 else 0.0
                    for m in target_modes:
                        row = [other_prob] * K_q
                        row[m] = self.target_prob
                        rows.append(row)
                    probs_tensor = torch.tensor(rows, dtype=torch.float32)
                else:
                    raise ValueError(f"queries[{idx}] requires 'target_modes', 'target_options', or 'target_probs'")

                num_modes = len(probs_tensor)
                num_classes_list.append(K_q)
                num_modes_list.append(num_modes)
                mode_probs_list.append(probs_tensor)

                entry: Dict[str, Any] = {
                    "target_modes": target_modes,
                    "num_classes": K_q,
                    "num_modes": num_modes,
                }
                if "question" in q:
                    entry["question"] = q["question"]
                if "options" in q:
                    entry["options"] = list(q["options"])
                normalized_queries.append(entry)

            self.queries = normalized_queries
            self._num_classes = num_classes_list
            self._num_modes = num_modes_list
            self.num_queries = len(normalized_queries)
        else:
            if num_classes is None or num_modes is None:
                raise ValueError("Either queries or num_classes and num_modes must be provided")

            if isinstance(num_classes, int):
                num_classes_seq = [num_classes]
            else:
                num_classes_seq = [int(c) for c in num_classes]

            if isinstance(num_modes, int):
                num_modes_seq = [num_modes] * len(num_classes_seq)
            else:
                num_modes_seq = [int(m) for m in num_modes]

            if len(num_classes_seq) != len(num_modes_seq):
                raise ValueError("num_classes and num_modes sequence lengths must match")

            self.num_queries = len(num_classes_seq)
            self._num_classes = num_classes_seq
            self._num_modes = num_modes_seq
            self.queries = []
            mode_probs_list = []

            for q_idx in range(self.num_queries):
                K_q = self._num_classes[q_idx]
                M_q = self._num_modes[q_idx]
                target_modes = list(range(M_q))
                rows = []
                other_prob = (1.0 - self.target_prob) / float(K_q - 1) if K_q > 1 else 0.0
                for m in target_modes:
                    row = [other_prob] * K_q
                    row[m] = self.target_prob
                    rows.append(row)
                probs_tensor = torch.tensor(rows, dtype=torch.float32)
                mode_probs_list.append(probs_tensor)
                self.queries.append({
                    "question": f"Q{q_idx}",
                    "target_modes": target_modes,
                    "num_classes": K_q,
                    "num_modes": M_q,
                })

        self.k_max = max(self._num_classes)

        # Precompute target soft-capped log-odds for each query mode
        self._centered_logits: List[torch.Tensor] = []
        self._mode_targets: List[torch.Tensor] = []

        for q_idx in range(self.num_queries):
            probs = mode_probs_list[q_idx]
            log_p = probs.clamp_min(1e-12).log()
            scaled_logits = log_p / self.temperature
            centered = scaled_logits - scaled_logits.mean(dim=-1, keepdim=True)
            self._centered_logits.append(centered)

            u = symmetric_log_odds(centered, tau=self.tau)
            K_q = self._num_classes[q_idx]
            if self.pad_to_k_max and K_q < self.k_max:
                pad = torch.full((len(probs), self.k_max - K_q), -self.tau, dtype=torch.float32)
                u = torch.cat([u, pad], dim=-1)
            self._mode_targets.append(u)

    def __call__(
        self,
        batch_size: int,
        *,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Draws batch_size independent target vectors across the Cartesian product.

        Returns:
            Dict containing:
                "real": FloatTensor of shape (batch_size, total_classes) on target device.
                "modes": LongTensor of shape (batch_size, num_queries) containing sampled mode indices.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError(f"generator must be a torch.Generator or None, got {type(generator)}")

        dev = device if device is not None else (generator.device if generator is not None else torch.device("cpu"))

        sampled_modes: List[torch.Tensor] = []
        sampled_u: List[torch.Tensor] = []

        for q_idx in range(self.num_queries):
            M_q = self._num_modes[q_idx]
            if M_q > 1:
                m_indices = torch.randint(0, M_q, (batch_size,), generator=generator, device=dev)
            else:
                m_indices = torch.zeros((batch_size,), dtype=torch.long, device=dev)
            sampled_modes.append(m_indices)

            if self.jitter_std > 0.0:
                base = self._centered_logits[q_idx].to(device=dev)[m_indices]
                K_q = self._num_classes[q_idx]
                noise = torch.randn((batch_size, K_q), generator=generator, device=dev, dtype=torch.float32)
                y = base + self.jitter_std * noise
                y = y - y.mean(dim=-1, keepdim=True)
                u = symmetric_log_odds(y, tau=self.tau)
                if self.pad_to_k_max and K_q < self.k_max:
                    pad = torch.full((batch_size, self.k_max - K_q), -self.tau, device=dev, dtype=torch.float32)
                    u = torch.cat([u, pad], dim=-1)
                sampled_u.append(u)
            else:
                targets = self._mode_targets[q_idx].to(device=dev)
                sampled_u.append(targets[m_indices])

        U_target = torch.cat(sampled_u, dim=-1)
        mode_indices = torch.stack(sampled_modes, dim=-1)

        return {"real": U_target, "modes": mode_indices}

    def sample_cartesian_product(
        self,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Enumerates all prod(M_q) combinations in the Cartesian product mode space."""
        dev = device if device is not None else torch.device("cpu")
        grids = [torch.arange(M_q, device=dev) for M_q in self._num_modes]
        mesh = torch.meshgrid(*grids, indexing="ij")
        stacked_modes = torch.stack([m.reshape(-1) for m in mesh], dim=-1)  # (N_prod, Q)
        N_prod = stacked_modes.shape[0]

        sampled_u: List[torch.Tensor] = []
        for q_idx in range(self.num_queries):
            m_indices = stacked_modes[:, q_idx]
            targets = self._mode_targets[q_idx].to(device=dev)
            sampled_u.append(targets[m_indices])

        U_target = torch.cat(sampled_u, dim=-1)
        return {"real": U_target, "modes": stacked_modes}

    def resume_identity(self) -> Dict[str, Any]:
        """Returns serializable configuration for exact run resumption."""
        return {
            "queries": copy.deepcopy(self.queries),
            "tau": self.tau,
            "temperature": self.temperature,
            "jitter_std": self.jitter_std,
            "target_prob": self.target_prob,
            "pad_to_k_max": self.pad_to_k_max,
        }


class MarginalEntropyRegularizer(nn.Module):
    """Batch-marginal Shannon entropy regularizer on option probabilities or logits.

    Enforces H(p_bar_q) >= min_entropy_ratio * ln(M_q) across queries by minimizing
    the squared hinge deficit:
        L_entropy = (1 / |Q|) sum_{q in Q} max(0, min_entropy_ratio * ln(M_q) - H(p_bar_q))^2

    Optionally includes capacity-balanced mode attraction with ceil(B / M_q) target capacity
    to actively pull batch samples toward distinct modes and resolve mode collapse.
    """

    def __init__(
        self,
        query_indices: Optional[Sequence[int]] = None,
        query_index: Optional[int] = None,
        *,
        num_classes: int = 2,
        num_modes: Union[int, Sequence[int], Dict[int, int]] = 2,
        min_entropy_ratio: float = 0.85,
        target_entropy_ratio: Optional[float] = None,
        hinge: bool = True,
        input_mode: str = "logits",
        eps: float = 1e-8,
        target_attraction_weight: float = 0.0,
        target_log_odds: float = 2.85,
        query_slices: Optional[Sequence[Tuple[int, int]]] = None,
        single_mode_queries: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if query_indices is not None:
            indices = [int(q) for q in query_indices]
        elif query_index is not None:
            indices = [int(query_index)]
        else:
            indices = [0]

        if len(indices) == 0:
            raise ValueError("query_indices must not be empty")

        if isinstance(num_modes, int):
            modes_dict = {q: int(num_modes) for q in indices}
        elif isinstance(num_modes, dict):
            modes_dict = {q: int(num_modes[q]) for q in indices}
        elif isinstance(num_modes, (list, tuple)):
            if len(num_modes) == 1 and len(indices) > 1:
                modes_dict = {q: int(num_modes[0]) for q in indices}
            elif len(num_modes) == len(indices):
                modes_dict = {q: int(m) for q, m in zip(indices, num_modes)}
            else:
                raise ValueError("num_modes sequence length must match query_indices length")
        else:
            raise ValueError(f"num_modes must be int, sequence, or dict, got {type(num_modes)}")

        effective_entropy_ratio = min_entropy_ratio if target_entropy_ratio is None else target_entropy_ratio
        if isinstance(effective_entropy_ratio, bool) or not isinstance(effective_entropy_ratio, (int, float)) or not (0.0 < effective_entropy_ratio <= 1.0):
            raise ValueError(f"min_entropy_ratio must be a float in (0.0, 1.0], got {effective_entropy_ratio}")
        if input_mode not in ("logits", "joint_log_odds", "probs"):
            raise ValueError(f"input_mode must be 'logits', 'joint_log_odds', or 'probs', got {input_mode!r}")
        if eps <= 0.0 or not math.isfinite(eps):
            raise ValueError(f"eps must be a positive finite float, got {eps}")

        self.query_indices = indices
        self.num_classes = int(num_classes)
        self.num_modes = modes_dict
        self.min_entropy_ratio = float(effective_entropy_ratio)
        self.hinge = bool(hinge)
        self.input_mode = input_mode
        self.eps = float(eps)
        self.target_attraction_weight = float(target_attraction_weight)
        self.target_log_odds = float(target_log_odds)
        self.query_slices = list(query_slices) if query_slices is not None else None
        self.single_mode_queries = list(single_mode_queries) if single_mode_queries is not None else []

    def forward(
        self,
        logits: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluates batch-marginal entropy deficit and optional mode attraction."""
        if logits is None:
            if "u" in kwargs:
                logits = kwargs["u"]
            elif "probs" in kwargs:
                logits = kwargs["probs"]
            elif len(kwargs) == 1:
                logits = next(iter(kwargs.values()))
            else:
                raise ValueError("MarginalEntropyRegularizer requires logits input")

        if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
            raise TypeError("MarginalEntropyRegularizer requires a floating-point tensor")

        # Normalize 1D tensor to 2D
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        B = logits.shape[0]
        losses: List[torch.Tensor] = []
        attr_losses: List[torch.Tensor] = []

        for q_idx, q in enumerate(self.query_indices):
            M_q = self.num_modes[q]

            # Extract query slice
            if logits.ndim == 3:
                # Shape (B, Q, K)
                idx_dim1 = q if q < logits.shape[1] else (q_idx if q_idx < logits.shape[1] else 0)
                u_q = logits[:, idx_dim1]
            elif self.query_slices is not None:
                s, e = self.query_slices[q_idx]
                u_q = logits[:, s:e]
            elif logits.shape[1] == self.num_classes and len(self.query_indices) == 1:
                u_q = logits
            elif logits.shape[1] >= (max(self.query_indices) + 1) * self.num_classes:
                q_start = q * self.num_classes
                q_end = q_start + self.num_classes
                u_q = logits[:, q_start:q_end]
            elif logits.shape[1] == sum(self.num_modes.values()):
                # Sequentially packed modes
                prev_sum = sum(self.num_modes[self.query_indices[i]] for i in range(q_idx))
                u_q = logits[:, prev_sum : prev_sum + M_q]
            else:
                u_q = logits

            # Option probabilities for active modes
            u_modes = u_q[:, :M_q] if u_q.shape[-1] >= M_q else u_q
            if self.input_mode == "joint_log_odds":
                p = torch.sigmoid(u_modes)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            elif self.input_mode == "logits":
                p = F.softmax(u_modes, dim=-1)
            else:  # "probs"
                p = u_modes / u_modes.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            # Batch-marginal Shannon entropy
            p_bar = p.mean(dim=0)
            H = - torch.sum(p_bar * torch.log(p_bar + self.eps))
            target_H = self.min_entropy_ratio * math.log(max(M_q, 1))

            if self.hinge:
                loss_q = F.relu(target_H - H).pow(2)
            else:
                loss_q = -H
            losses.append(loss_q)

            # Capacity-balanced mode attraction if configured
            if self.target_attraction_weight > 0.0 and B > 0 and M_q > 1:
                scores = u_modes.detach()
                if B > 1:
                    std = scores.std(dim=0, keepdim=True)
                    scores_norm = (scores - scores.mean(dim=0, keepdim=True)) / (std + 1e-5)
                else:
                    scores_norm = scores

                assigned = torch.full((B,), -1, dtype=torch.long, device=logits.device)
                cap = (B + M_q - 1) // M_q
                counts = [0] * M_q
                flat_idx = torch.argsort(scores_norm.reshape(-1), descending=True)
                for idx_val in flat_idx.tolist():
                    b_i, m_i = idx_val // M_q, idx_val % M_q
                    if assigned[b_i] == -1 and counts[m_i] < cap:
                        assigned[b_i] = m_i
                        counts[m_i] += 1
                assigned = assigned.clamp_min(0)

                if self.input_mode == "joint_log_odds":
                    pos_val = u_modes.gather(1, assigned.unsqueeze(-1)).squeeze(-1)
                    l_pos = F.relu(self.target_log_odds - pos_val).mean()
                    if u_q.shape[-1] > M_q:
                        l_inv = F.relu(u_q[:, M_q:] + (self.target_log_odds + 0.05)).mean()
                    else:
                        l_inv = torch.zeros((), device=logits.device, dtype=logits.dtype)
                    attr_losses.append(l_pos + 1.5 * l_inv)
                else:
                    p_assigned = p.gather(1, assigned.unsqueeze(-1)).squeeze(-1)
                    l_pos = - torch.log(p_assigned + self.eps).mean()
                    attr_losses.append(l_pos)

        entropy_loss = torch.stack(losses).mean()
        if len(attr_losses) > 0:
            return entropy_loss + self.target_attraction_weight * torch.stack(attr_losses).mean()
        return entropy_loss

    def resume_identity(self) -> Dict[str, Any]:
        """Returns serializable configuration for exact run resumption."""
        return {
            "query_indices": list(self.query_indices),
            "num_classes": self.num_classes,
            "num_modes": dict(self.num_modes),
            "min_entropy_ratio": self.min_entropy_ratio,
            "hinge": self.hinge,
            "input_mode": self.input_mode,
            "eps": self.eps,
            "target_attraction_weight": self.target_attraction_weight,
            "target_log_odds": self.target_log_odds,
            "query_slices": self.query_slices,
            "single_mode_queries": list(self.single_mode_queries),
        }


class MarginalEntropyManipulator(PriorManipulator):
    """PriorManipulator enforcing batch-marginal Shannon entropy and mode diversity.

    Evaluates MarginalEntropyRegularizer on critic logits, generator outputs, or
    sampled simplex codes during GANTrainer._step, adding the penalty to loss_g.
    """

    row_local_gradients: bool = True

    def __init__(
        self,
        query_indices: Optional[Sequence[int]] = None,
        query_index: Optional[int] = None,
        *,
        num_classes: int = 2,
        num_modes: Union[int, Sequence[int], Dict[int, int]] = 2,
        min_entropy_ratio: float = 0.85,
        target_entropy_ratio: Optional[float] = None,
        weight: float = 1.0,
        hinge: bool = True,
        input_mode: str = "logits",
        source: str = "auto",
        eps: float = 1e-8,
        target_attraction_weight: float = 0.0,
        target_log_odds: float = 2.85,
        query_slices: Optional[Sequence[Tuple[int, int]]] = None,
        single_mode_queries: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.source = source
        self.regularizer = MarginalEntropyRegularizer(
            query_indices=query_indices,
            query_index=query_index,
            num_classes=num_classes,
            num_modes=num_modes,
            min_entropy_ratio=min_entropy_ratio if target_entropy_ratio is None else target_entropy_ratio,
            hinge=hinge,
            input_mode=input_mode,
            eps=eps,
            target_attraction_weight=target_attraction_weight,
            target_log_odds=target_log_odds,
            query_slices=query_slices,
            single_mode_queries=single_mode_queries,
        )

    def regularize(
        self,
        raw_rows: torch.Tensor,
        *,
        sampled_codes: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight <= 0.0:
            return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)

        target = None
        if self.source == "logits":
            target = logits
        elif self.source == "codes":
            target = sampled_codes
        elif self.source == "auto":
            if logits is not None and logits.shape[-1] >= self.regularizer.num_classes:
                target = logits
            elif sampled_codes is not None:
                target = sampled_codes
            else:
                target = logits

        if target is None:
            return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)

        loss = self.regularizer(target)
        return (self.weight * loss).to(device=raw_rows.device, dtype=raw_rows.dtype)


class NormalizedFeatureCosineRepulsion(nn.Module):
    r"""Normalized feature pairwise cosine repulsion regularizer.

    Penalizes pairwise cosine similarity between normalized feature representations
    that exceed margin:
        .. math::
            S_{ij} = \frac{f_i^\top f_j}{\|f_i\|_2 \|f_j\|_2}
            L_{\text{repulse}} = \frac{1}{B(B-1)} \sum_{i \neq j} \max(0, S_{ij} - \text{margin})

    Setting :math:`\text{margin} \le 0.88` directly enforces mean pairwise cosine distance
    :math:`1 - S_{ij} \ge 0.12` (the AC3 feature diversity gate).
    """

    def __init__(
        self,
        margin: float = 0.88,
        *,
        weight: float = 1.0,
        squared: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not math.isfinite(margin) or not (0.0 <= margin < 1.0):
            raise ValueError(f"margin must be finite in [0.0, 1.0), got {margin}")
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"weight must be finite and nonnegative, got {weight}")
        if eps <= 0.0 or not math.isfinite(eps):
            raise ValueError(f"eps must be positive finite float, got {eps}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

        self.margin = float(margin)
        self.weight = float(weight)
        self.squared = bool(squared)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(
        self,
        features: Optional[torch.Tensor] = None,
        margin: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if features is None:
            if "v" in kwargs:
                features = kwargs["v"]
            elif "pooled_vision" in kwargs:
                features = kwargs["pooled_vision"]
            elif len(kwargs) == 1:
                features = next(iter(kwargs.values()))
            else:
                raise ValueError("NormalizedFeatureCosineRepulsion requires feature tensor input")

        if not isinstance(features, torch.Tensor) or not features.is_floating_point():
            raise TypeError("NormalizedFeatureCosineRepulsion requires a floating-point tensor")

        # Pool higher-dimensional tensors to 2D representations
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        elif features.ndim != 2:
            raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {tuple(features.shape)}")

        B = features.shape[0]
        if B <= 1:
            return torch.zeros((), device=features.device, dtype=features.dtype)

        eff_margin = self.margin if margin is None else float(margin)
        v_norm = F.normalize(features.float(), p=2, dim=-1, eps=self.eps)
        sim = torch.mm(v_norm, v_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
        off_diag = sim[mask]

        penalty = F.relu(off_diag - eff_margin)
        if self.squared:
            loss = penalty.pow(2)
        else:
            loss = penalty

        scalar_loss = loss.mean() if self.reduction == "mean" else loss.sum()
        return (self.weight * scalar_loss).to(dtype=features.dtype)

    def compute_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """Computes mean pairwise cosine distance across off-diagonal pairs."""
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(1.0, device=features.device, dtype=features.dtype)
        v_norm = F.normalize(features.float(), p=2, dim=-1, eps=self.eps)
        sim = torch.mm(v_norm, v_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
        return (1.0 - sim[mask].mean()).to(dtype=features.dtype)

    def resume_identity(self) -> Dict[str, Any]:
        """Returns serializable configuration for exact run resumption."""
        return {
            "margin": self.margin,
            "weight": self.weight,
            "squared": self.squared,
            "eps": self.eps,
            "reduction": self.reduction,
        }


class CosineRepulsionManipulator(PriorManipulator):
    """PriorManipulator penalizing pairwise cosine similarity between sample representations.

    Evaluates NormalizedFeatureCosineRepulsion on generator outputs (fake_g),
    generator latent features, or feature extractor projections during GANTrainer._step.
    """

    row_local_gradients: bool = True

    def __init__(
        self,
        *,
        margin: float = 0.88,
        weight: float = 1.0,
        source: str = "auto",
        feature_extractor: Optional[nn.Module] = None,
        squared: bool = False,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.source = source
        self.feature_extractor = feature_extractor
        self.repulsion = NormalizedFeatureCosineRepulsion(
            margin=margin,
            weight=1.0,
            squared=squared,
            eps=eps,
            reduction=reduction,
        )

    def regularize(
        self,
        raw_rows: torch.Tensor,
        *,
        sampled_codes: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight <= 0.0:
            return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)

        features = None
        if self.source == "generated":
            features = generated
        elif self.source == "codes":
            features = sampled_codes
        elif self.source == "auto":
            features = generated if generated is not None else sampled_codes

        if features is None:
            return torch.zeros((), device=raw_rows.device, dtype=raw_rows.dtype)

        if self.feature_extractor is not None:
            features = self.feature_extractor(features)

        loss = self.repulsion(features)
        return (self.weight * loss).to(device=raw_rows.device, dtype=raw_rows.dtype)


def evaluate_tomographic_constraints(
    images: torch.Tensor,
    *,
    operator: Optional[Callable[[torch.Tensor], Any]] = None,
    target_modes: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    logits: Optional[torch.Tensor] = None,
    shift: Tuple[int, int] = (2, 2),
    patch_size: int = 8,
    secondary_patch_size: Optional[int] = 4,
    primary_query_index: int = 0,
    num_classes: Optional[Sequence[int]] = None,
    tau: float = 3.0,
) -> Dict[str, float]:
    """Evaluates the 6 quantitative tomographic constraint metrics defined in §AC3.

    Returns:
        Dict with keys:
            - "mean_target_prob": float (gate >= 0.80)
            - "primary_accuracy": float (gate >= 0.95)
            - "entropy_ratio": float (gate >= 0.85)
            - "feature_diversity": float (gate >= 0.12)
            - "patch_seam_ratio": float (gate <= 1.15)
            - "shift_sensitivity": float (gate <= 0.15)
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images must be a torch.Tensor, got {type(images)}")
    if images.ndim != 4:
        raise ValueError(f"images must be 4D (B, C, H, W), got {images.shape}")

    op_probs = None
    op_features = features
    op_logits = logits
    shifted_probs = None

    if operator is not None:
        out = operator(images)
        if isinstance(out, dict):
            op_probs = out.get("probs")
            if op_probs is None and "logits" in out:
                op_logits = out["logits"]
            if op_features is None:
                op_features = out.get("features", out.get("pooled_vision", out.get("v")))
        elif isinstance(out, (tuple, list)):
            if len(out) >= 1:
                first = out[0]
                if isinstance(first, torch.Tensor):
                    op_logits = first
            if len(out) >= 2 and op_features is None:
                second = out[1]
                if isinstance(second, torch.Tensor):
                    op_features = second
        elif isinstance(out, torch.Tensor):
            op_logits = out

        # Shift sensitivity evaluation under translation
        images_shifted = torch.roll(images, shifts=shift, dims=(-2, -1))
        out_shift = operator(images_shifted)
        if isinstance(out_shift, dict):
            shifted_probs = out_shift.get("probs")
            if shifted_probs is None and "logits" in out_shift:
                shifted_logits = out_shift["logits"]
                shifted_probs = F.softmax(shifted_logits, dim=-1)
        elif isinstance(out_shift, (tuple, list)) and len(out_shift) >= 1:
            if isinstance(out_shift[0], torch.Tensor):
                shifted_probs = F.softmax(out_shift[0], dim=-1)
        elif isinstance(out_shift, torch.Tensor):
            shifted_probs = F.softmax(out_shift, dim=-1)

    # Derive probabilities from logits if needed
    if op_probs is None and op_logits is not None:
        if op_logits.ndim == 3:
            op_probs = F.softmax(op_logits, dim=-1)
        elif op_logits.ndim == 2:
            if num_classes is not None:
                prob_parts = []
                idx = 0
                for K_q in num_classes:
                    prob_parts.append(F.softmax(op_logits[:, idx : idx + K_q], dim=-1))
                    idx += K_q
                op_probs = torch.stack(prob_parts, dim=1) if len(set(num_classes)) == 1 else prob_parts
            else:
                op_probs = F.softmax(op_logits, dim=-1).unsqueeze(1)

    if op_probs is None:
        raise ValueError("evaluate_tomographic_constraints requires operator outputting probs or logits")

    # 1. Mean Target Probability & 2. Primary Accuracy
    if isinstance(op_probs, torch.Tensor):
        if op_probs.ndim == 2:
            op_probs = op_probs.unsqueeze(1)  # (B, 1, K)
        Q = op_probs.shape[1]

        if target_modes is not None:
            if target_modes.ndim == 1:
                target_modes = target_modes.unsqueeze(-1)
            target_p = op_probs.gather(-1, target_modes.unsqueeze(-1)).squeeze(-1)
            mean_target_prob = float(target_p.mean().item())

            prim_idx = min(primary_query_index, Q - 1)
            pred_prim = op_probs[:, prim_idx].argmax(dim=-1)
            target_prim = target_modes[:, prim_idx]
            primary_accuracy = float((pred_prim == target_prim).float().mean().item())
        else:
            mean_target_prob = float(op_probs.max(dim=-1).values.mean().item())
            primary_accuracy = 1.0

        # 3. Marginal Entropy Ratio
        entropy_ratios = []
        for q in range(Q):
            p_q = op_probs[:, q]  # (B, K_q)
            if target_modes is not None:
                M_q = max(int(target_modes[:, q].max().item()) + 1, 2)
            else:
                M_q = p_q.shape[-1]
            p_sub = p_q[:, :M_q]
            p_bar = p_sub.mean(dim=0)
            p_bar = p_bar / p_bar.sum().clamp_min(1e-8)
            H = - torch.sum(p_bar * torch.log(p_bar.clamp_min(1e-8)))
            if M_q > 1:
                ratio = (H / math.log(M_q)).item()
                entropy_ratios.append(ratio)
        entropy_ratio = float(sum(entropy_ratios) / max(len(entropy_ratios), 1))
    else:
        # op_probs is a sequence of tensors [ (B, K_0), (B, K_1), ... ]
        Q = len(op_probs)
        target_p_list = []
        entropy_ratios = []
        for q in range(Q):
            p_q = op_probs[q]
            if target_modes is not None:
                tm_q = target_modes[:, q]
                t_p = p_q.gather(-1, tm_q.unsqueeze(-1)).squeeze(-1)
                target_p_list.append(t_p)
                M_q = max(int(tm_q.max().item()) + 1, 2)
            else:
                target_p_list.append(p_q.max(dim=-1).values)
                M_q = p_q.shape[-1]

            p_sub = p_q[:, :M_q]
            p_bar = p_sub.mean(dim=0)
            p_bar = p_bar / p_bar.sum().clamp_min(1e-8)
            H = - torch.sum(p_bar * torch.log(p_bar.clamp_min(1e-8)))
            if M_q > 1:
                entropy_ratios.append((H / math.log(M_q)).item())

        mean_target_prob = float(torch.stack(target_p_list, dim=-1).mean().item())
        if target_modes is not None:
            prim_idx = min(primary_query_index, Q - 1)
            pred_prim = op_probs[prim_idx].argmax(dim=-1)
            primary_accuracy = float((pred_prim == target_modes[:, prim_idx]).float().mean().item())
        else:
            primary_accuracy = 1.0
        entropy_ratio = float(sum(entropy_ratios) / max(len(entropy_ratios), 1))

    # 4. Feature Diversity
    if op_features is not None:
        rep = NormalizedFeatureCosineRepulsion()
        feature_diversity = float(rep.compute_diversity(op_features).item())
    else:
        feature_diversity = 1.0

    # 5. Patch Seam Ratio
    try:
        seam_calc = WavefrontManifoldPrior(
            patch_size=patch_size,
            secondary_patch_size=secondary_patch_size,
        )
        patch_seam_ratio = float(seam_calc.compute_patch_seam_ratio(images).item())
    except (RuntimeError, ValueError):
        patch_seam_ratio = 1.0

    # 6. Spatial Shift Sensitivity
    if shifted_probs is not None:
        if isinstance(op_probs, torch.Tensor) and isinstance(shifted_probs, torch.Tensor):
            if shifted_probs.ndim == 2 and op_probs.ndim == 3:
                shifted_probs = shifted_probs.unsqueeze(1)
            if target_modes is not None:
                if target_modes.ndim == 1:
                    target_modes = target_modes.unsqueeze(-1)
                p_orig = op_probs.gather(-1, target_modes.unsqueeze(-1)).squeeze(-1)
                p_s = shifted_probs.gather(-1, target_modes.unsqueeze(-1)).squeeze(-1)
                shift_sensitivity = float((p_s - p_orig).abs().mean().item())
            else:
                shift_sensitivity = float((shifted_probs - op_probs).abs().mean().item())
        else:
            shift_sensitivity = 0.0
    else:
        shift_sensitivity = 0.0

    return {
        "mean_target_prob": mean_target_prob,
        "primary_accuracy": primary_accuracy,
        "entropy_ratio": entropy_ratio,
        "feature_diversity": feature_diversity,
        "patch_seam_ratio": patch_seam_ratio,
        "shift_sensitivity": shift_sensitivity,
    }


__all__ = [
    "CosineRepulsionManipulator",
    "MarginalEntropyManipulator",
    "MarginalEntropyRegularizer",
    "NormalizedFeatureCosineRepulsion",
    "TomographicConstraintTarget",
    "evaluate_tomographic_constraints",
    "symmetric_log_odds",
]
