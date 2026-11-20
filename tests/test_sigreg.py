"""SIGReg software acceptance; two-update hosts confer no tier qualification."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import Recipe, SIGReg, get_recipe, init
from particlegan.sigreg import sigreg_statistic
from experiments.forge.api import FormulationContext, task_formulation_context
from experiments.forge.behavior_adapters import run_behavior
from experiments.forge.boundaries import recipe_field_owner
from experiments.forge.techniques import recipe_field_active, validate_same_technique

ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def context(device="cpu", **overrides):
    return FormulationContext(recipe_preset="bcap", seed=0, device=device,
        recipe_overrides={"num_particles": 12, "z_dim": 2, "batch_size": 4, "total_steps": 4,
                          "prior_regularizer": "sigreg", "prior_reg": .001, **overrides},
        prior=dict(kind="mog", sigma=.025, standardize=False), prior_contract="recipe_owned_v1")


def trainer(ctx):
    g = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2)), component="generator")
    d = ctx.construct(lambda: nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)), component="discriminator")
    return ctx.build_trainer(g.to(ctx.device), d.to(ctx.device))


def test_supplied_projection_formula_matches_explicit_trapezoid_and_gradient():
    rows = torch.tensor([[-1., .2], [.3, -.8], [1.1, 1.2]], dtype=torch.float64, requires_grad=True)
    directions = torch.tensor([[1., 0., .6], [0., 1., .8]], dtype=torch.float64)
    actual = sigreg_statistic(rows, directions, knots=17)
    expected = rows.new_zeros(())
    for direction in directions.T:
        for k in range(17):
            t = rows.new_tensor(3 * k / 16)
            phi = torch.exp(-t.square() / 2)
            x = (rows @ direction) * t
            error = (x.cos().mean() - phi).square() + x.sin().mean().square()
            expected = expected + len(rows) / 3 * error * phi * (3 / 16 if k in (0, 16) else 6 / 16)
    assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-13)
    a = torch.autograd.grad(actual, rows, retain_graph=True)[0]
    b = torch.autograd.grad(expected, rows)[0]
    assert torch.allclose(a, b, atol=1e-13, rtol=1e-13)
    assert torch.isfinite(a).all() and a.abs().max() > 0


def test_statistic_detects_mean_and_expansion_that_variance_floor_allows():
    prior = Recipe.from_dict(dict(num_particles=256)).make_prior(generator=torch.Generator().manual_seed(0))
    init.deterministic_orthogonal_(prior, seed=0)
    directions = torch.eye(2)
    normal = sigreg_statistic(prior.z, directions)
    shifted = sigreg_statistic(prior.z + 2, directions)
    expanded = sigreg_statistic(prior.z * 3, directions)
    assert shifted > 10 * normal and expanded > 10 * normal


def test_projection_draw_is_private_checkpointed_and_weight_applies_once():
    rows = torch.arange(24, dtype=torch.float32).reshape(12, 2).div(8).requires_grad_()
    module = SIGReg(weight=.25)
    state = deepcopy(module.state_dict())
    before = torch.get_rng_state().clone()
    a = module(rows)
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.equal(state["projection_rng_state"], module.projection_rng_state)
    unit = SIGReg(weight=1)
    unit.load_state_dict(state)
    assert torch.equal(a, .25 * unit(rows))
    state = deepcopy(module.state_dict())
    expected = module(rows)
    module.load_state_dict(state)
    assert torch.equal(expected, module(rows))


def test_zero_weight_consumes_no_rng_or_nonfinite_rows():
    module = SIGReg(weight=0)
    before = deepcopy(module.state_dict())
    rows = torch.full((12, 2), float("nan"), requires_grad=True)
    value = module(rows)
    value.backward()
    assert value == 0 and torch.equal(rows.grad, torch.zeros_like(rows))
    equal(before, module.state_dict())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=[pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")])])
def test_public_bcap_training_resume_replays_projection_rng_and_all_consumed_state(device):
    ctx = context(device)
    a = trainer(ctx)
    real = torch.arange(8, device=device, dtype=torch.float32).reshape(4, 2) / 4
    a.step(real)
    saved = ctx.state_dict()
    assert "prior_regularizer" in saved["trainer"]
    a.step(real)
    expected = ctx.state_dict()
    restored = context(device)
    b = trainer(restored)
    restored.load_state_dict(saved)
    b.step(real)
    equal(expected, restored.state_dict())
    before = b.state_dict()
    bad = deepcopy(before)
    bad["prior_regularizer"]["projection_rng_state"] = torch.ones(2, dtype=torch.uint8)
    with pytest.raises(ValueError, match="SIGReg"):
        b.load_state_dict(bad)
    equal(before, b.state_dict())


def test_disabled_sigreg_preserves_baseline_training_tensors_optimizers_and_streams():
    sig = context(prior_reg=0)
    plain = context(prior_reg=0, prior_regularizer="vicreg")
    a, b = trainer(sig), trainer(plain)
    before = deepcopy(a.prior_regularizer.state_dict())
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4
    for _ in range(2):
        x, y = a.step(real), b.step(real)
        assert torch.equal(x["loss_g"], y["loss_g"])
    for key in ("models", "optimizers", "streams", "initial_lrs", "completed_steps"):
        equal(a.state_dict()[key], b.state_dict()[key])
    equal(sig.streams.state_dict(), plain.streams.state_dict())
    equal(before, a.prior_regularizer.state_dict())
    assert "prior_regularizer" not in a.state_dict()


@pytest.mark.parametrize("particles,expected", [(12, 12), (1025, None)])
def test_active_sigreg_respects_existing_regularization_row_law(particles, expected):
    a = trainer(context(num_particles=particles))
    lengths = []
    hook = a.prior_regularizer.register_forward_pre_hook(lambda module, args: lengths.append(len(args[0])))
    a.step(torch.ones(4, 2))
    hook.remove()
    assert lengths == [expected] if expected is not None else 1 <= lengths[0] <= 4


@pytest.mark.parametrize("host", ["trajectory", "residual_student", "cover_leftover", "ae_gan_hold"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=[pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")])])
def test_behavior_hosts_apply_sigreg_and_archive_projection_state(tmp_path, monkeypatch, host, device):
    task = json.loads((ROOT / "configs/forge/tasks" / (host + ".json")).read_text())
    task["execution"]["steps"] = 2
    calls = []
    original = SIGReg.forward
    def observed(module, rows):
        calls.append((module.weight, len(rows)))
        return original(module, rows)
    monkeypatch.setattr(SIGReg, "forward", observed)
    candidate = dict(recipe_preset="bcap", recipe_overrides=dict(prior_regularizer="sigreg", prior_reg=.001))
    result = run_behavior(dict(candidate=candidate, protocol=dict(seed=0)), task, tmp_path, device=device)
    assert len(calls) == 2 and all(weight == .001 for weight, _ in calls)
    state = torch.load(tmp_path / "component-state.pt", weights_only=False)
    assert "projection_rng_state" in state["prior_regularizer"]
    assert result["evidence"]["guards"]["all_finite"]


def test_sigreg_fields_have_explicit_ownership_and_coefficient_search_preserves_mechanism():
    base = get_recipe("bcap", prior_regularizer="sigreg", prior_reg=.001)
    assert recipe_field_owner("prior_sigreg_projections") == "hyperparameter"
    assert recipe_field_active("prior_reg", base)
    assert recipe_field_active("prior_sigreg_projections", base)
    assert not recipe_field_active("prior_reg_target_std", base)
    validate_same_technique(base, base.replace(prior_reg=.01))
    with pytest.raises(ValueError, match="mechanism"):
        validate_same_technique(base, base.replace(prior_regularizer="vicreg"))


def test_word_public_component_checkpoint_replays_regularizer_and_named_streams():
    from benchmarks.toy_audit.api_images import WordFixture
    from experiments.forge.word_adapter import word_context
    task = json.loads((ROOT / "configs/forge/tasks/five_word_joint_smoke.json").read_text())
    candidate = json.loads((ROOT / "configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json").read_text())
    candidate["recipe_overrides"].update(prior_regularizer="sigreg", prior_reg=.001)
    request = {"candidate": candidate, "protocol": {"seed": 0}}
    def build():
        ctx = word_context(request, task, "cpu")
        fixture = WordFixture(device="cpu", seed=0, recipe_name=None, max_steps=4, components=ctx)
        return ctx, fixture
    ctx, a = build()
    a.step()
    saved, streams = deepcopy(a.state_dict()), ctx.streams.state_dict()
    assert "projection_rng_state" in saved["prior_regularizer"]
    a.step()
    expected, expected_streams = deepcopy(a.state_dict()), ctx.streams.state_dict()
    fresh, b = build()
    fresh.streams.load_state_dict(streams)
    b.policy.load_state_dict(saved["api_state"])
    b.recipe.restore_prior_regularizer(b.spread, saved["prior_regularizer"])
    b.data_generator.set_state(saved["data_generator"])
    b.step()
    equal(expected, b.state_dict())
    equal(expected_streams, fresh.streams.state_dict())


def test_all_current_tier1_tier2_bindings_admit_sigreg_after_source_refresh():
    from experiments.forge.adapters import adapter_preflight
    from recipe_entrypoint_compat import model_source_blockers
    candidate = json.loads((ROOT / "configs/forge/ideas/recipe-prior-refactor-bcap-pure-v1.json").read_text())
    candidate["recipe_overrides"].update(prior_regularizer="sigreg", prior_reg=.001)
    view = json.loads((ROOT / "configs/forge/views/discriminator_stability.json").read_text())
    # Stretch hosts that bind one declared formulation report BLOCKED for any
    # other recipe instead of substituting it; they are excluded with a reason
    # and must keep reporting the blocker (never a silent admission).
    excluded = {
        "wide_conditional_spread": "stretch host binds clean EMPF critic and EMPF prior regularizer only "
                                   "(experiments/forge/conditional_spread.py:context_for)",
    }
    for row in view["assignments"]:
        if row["task"] in excluded:
            assert row["importance"] == "stretch", row["task"]
            task = json.loads((ROOT / "configs/forge/tasks" / (row["task"] + ".json")).read_text())
            assert adapter_preflight(task, candidate, root=ROOT), excluded[row["task"]]
            continue
        if row["qualification_tier"] <= 2:
            task = json.loads((ROOT / "configs/forge/tasks" / (row["task"] + ".json")).read_text())
            # Source-bound model cards require their original checkout or a
            # separately qualified new cohort after restoration API migration.
            expected = model_source_blockers(ROOT, task) if task["adapter"] == "public_api_model_toy" else []
            assert adapter_preflight(task, candidate, root=ROOT) == expected, row["task"]
