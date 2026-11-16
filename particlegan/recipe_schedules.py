"""Explicit recipe schedules and observation-only bookkeeping for plain Adam.

The schedules use completed optimizer updates against the declared recipe
horizon. Sampling, penalty calls and externally shortened execution budgets do
not advance or rescale that horizon. Plain Adam uses the declared update law;
its critic observer only counts steps for penalties and checkpoint receipts.
"""
from copy import copy
import math

import torch

from .grad_regularizers import CriticStepRecord


def cosine_value(completed_steps, start, end, anneal_end, total_steps):
    """Cosine interpolation over a declared initial fraction, then hold."""
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be a nonnegative integer")
    fraction = min(1.0, completed_steps / (anneal_end * total_steps))
    return float(end + (start - end) * .5 * (1.0 + math.cos(math.pi * fraction)))


def apply_optimizer_schedule(completed_steps, recipe, optimizer):
    """Apply configured beta2 to every group before its next Adam update."""
    if getattr(recipe, "beta2_end", None) is None:
        return
    for group in optimizer.param_groups:
        initial = group.get("_recipe_initial_betas")
        if initial is None:
            raise ValueError("scheduled beta2 requires an optimizer from the recipe factories")
        group["betas"] = (initial[0], cosine_value(completed_steps, initial[1], recipe.beta2_end,
                                                 recipe.beta2_anneal_end, recipe.total_steps))


def apply_penalty_schedule(completed_steps, recipe, penalty):
    """Set the fixed penalty coefficient before evaluating its loss."""
    if getattr(recipe, "reg_coeff_end", None) is None or penalty is None:
        return
    penalty.regularizer.coeff = cosine_value(
        completed_steps, penalty.initial_coeff, recipe.reg_coeff_end,
        recipe.reg_coeff_anneal_end, recipe.total_steps)


def apply_training_schedules(completed_steps, recipe, optimizers, penalty=None):
    """Shared schedule boundary for public policy and caller-owned loops."""
    for optimizer in optimizers:
        apply_optimizer_schedule(completed_steps, recipe, optimizer)
    apply_penalty_schedule(completed_steps, recipe, penalty)


def _observe_critic_step(optimizer, args, kwargs):
    optimizer.record.record_step(optimizer)


def _save_critic_observer(optimizer, state):
    state["regularizer"] = {"optimizer_family": "adam", "record": optimizer.record.state_dict(),
                            "ema": None, "guard": None}


def validate_plain_adam_state(optimizer, state):
    """Validate observer/schedule metadata without mutating an optimizer.

    Native Adam's deepcopy retains only its standard tensors and groups. Public
    checkpoint preflight therefore checks these extras on the live observer
    before making that temporary copy.
    """
    if getattr(optimizer, "recipe_optimizer_family", None) != "adam":
        return
    if not isinstance(state, dict):
        raise ValueError("invalid plain Adam checkpoint")
    if type(optimizer) is torch.optim.Adam and "tensorflow_v1" in state:
        raise ValueError("TensorFlow-v1 checkpoint cannot load into native PyTorch Adam")
    if hasattr(optimizer, "record"):
        extra = state.get("regularizer")
        if (not isinstance(extra, dict) or set(extra) != {"optimizer_family", "record", "ema", "guard"}
                or extra["optimizer_family"] != "adam" or extra["ema"] is not None or extra["guard"] is not None):
            raise ValueError("invalid plain Adam critic observer checkpoint")
        record = extra["record"]
        if hasattr(optimizer.record, "base_state"):
            record = optimizer.record.base_state(record)
        if not isinstance(record, dict) or set(record) != set(CriticStepRecord.KEYS):
            raise ValueError("invalid plain Adam critic step record")
        for key in ("calls", "observed_steps"):
            if type(record[key]) is not int or record[key] < 0:
                raise ValueError("invalid plain Adam critic step count")
        if record["anchor_started"] is not False:
            raise ValueError("plain Adam cannot restore a critic anchor")
        for key in ("lr_max", "lr_last"):
            value = record[key]
            if key == "lr_last" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid plain Adam critic LR record")
        copy(optimizer.record).load_state_dict(extra["record"])
    groups = state.get("param_groups", [])
    for actual, saved in zip(optimizer.param_groups, groups):
        if ("_recipe_initial_betas" in actual
                and tuple(saved.get("_recipe_initial_betas", ())) != actual["_recipe_initial_betas"]):
            raise ValueError("plain Adam beta2 schedule initialization differs from its recipe")


def _load_critic_before(optimizer, state):
    validate_plain_adam_state(optimizer, state)
    state = dict(state)
    optimizer._plain_pending_record = state.pop("regularizer")["record"]
    return state


def _load_native_before(optimizer, state):
    validate_plain_adam_state(optimizer, state)


def _load_critic_after(optimizer):
    optimizer.record.load_state_dict(optimizer.__dict__.pop("_plain_pending_record"))
    if hasattr(optimizer.record, "restore_device_state"):
        optimizer.record.restore_device_state(optimizer.param_groups[0]["params"][0].device)


def make_plain_adam(recipe, params, *, critic=None, **options):
    """Declared Adam law with disabled interventions and a critic observer."""
    if recipe.adam_variant == "tensorflow_v1":
        from .tensorflow_adam import TensorFlowV1Adam
        optimizer = TensorFlowV1Adam(params, **options)
    else:
        optimizer = torch.optim.Adam(params, **options)
    optimizer.recipe_optimizer_family = "adam"
    if type(optimizer) is torch.optim.Adam:
        optimizer.register_load_state_dict_pre_hook(_load_native_before)
    if recipe.beta2_end is not None:
        for group in optimizer.param_groups:
            group["_recipe_initial_betas"] = tuple(group["betas"])
    if critic is None:
        optimizer.latent_damping = optimizer.latent_history = None
        optimizer.direct_response = optimizer.direct_history = None
    else:
        optimizer.critic = critic
        optimizer.ema_critic = optimizer.anchor = optimizer.guard = None
        if recipe.effective_critic_formulation == "bispf":
            from .bispf import _BISPFObserverStepRecord
            optimizer.record = _BISPFObserverStepRecord()
        elif recipe.effective_critic_formulation == "empf":
            from .empf import _EMPFObserverStepRecord
            from ._empf.critic import _attach_empf_critic_hook
            optimizer.record = _EMPFObserverStepRecord()
            _attach_empf_critic_hook(critic, optimizer.record)
        elif recipe.effective_critic_formulation == "rhbi" or recipe.rhbi_weight > 0 or recipe.rhbi_critic_weight > 0:
            from .rhbi import _RHBIObserverStepRecord, _attach_rhbi_critic_hook
            optimizer.record = _RHBIObserverStepRecord()
            _attach_rhbi_critic_hook(critic, optimizer.record)
        else:
            optimizer.record = CriticStepRecord()
        optimizer.register_step_post_hook(_observe_critic_step)
        optimizer.register_state_dict_post_hook(_save_critic_observer)
        optimizer.register_load_state_dict_pre_hook(_load_critic_before)
        optimizer.register_load_state_dict_post_hook(_load_critic_after)
    return optimizer
