"""Unit, problem-only audit, oracle PASS, destructive FAIL, bounded GIF, cross-recipe, and continuation tests."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from PIL import Image
import pytest
import torch

from benchmarks.toy_audit import api_contract
from benchmarks.toy_audit import api_four_laws_4 as fl4
from experiments.forge.state import state_digest
from particlegan import Recipe, unwrap_prior

_FORBIDDEN_INTERNAL_THRESHOLD_KEYS = frozenset({
    "stiff_gate",
    "stiff_gate_mean",
    "s_th",
    "s_therm",
    "d_max_s1",
    "D_max_s1",
    "mean_dz",
    "peak_rms",
    "prior_peak_rms",
    "critic_r_stable",
    "valley_sep",
    "realized_step_max",
    "ood_disp_ratio",
    "scale_parity_err",
    "post_shock_recovery",
})


def test_exports_specs_problem_only_thresholds_and_cases_contract() -> None:
    """Verify exact Interface Contract exports, strictly generated-output gates, and zero discovery regressions."""
    assert fl4.TASKS == (
        "fl_law41_trust_rare1pct_tanh",
        "fl_law42_swap_deep_critic_16l",
        "fl_law42_deep_resbn_and_conv1x1",
        "fl_law44_highd_sparse_prior_and_shock",
    )
    assert fl4.HOST_DEFINITIONS is fl4.SPECS
    assert fl4.Fixture is fl4.FourLawsFixture
    assert set(fl4.SPECS) == set(fl4.TASKS)
    assert set(fl4.THRESHOLDS) == set(fl4.TASKS)

    required_spec_keys = {
        "task_id",
        "case_id",
        "kind",
        "law",
        "defects",
        "title",
        "goal",
        "target",
        "resources",
        "prior",
        "model",
        "steps",
        "eval_samples",
        "observations",
        "minimum_stable_checks",
        "sampling_law",
        "view_kind",
    }
    for tid in fl4.TASKS:
        spec = fl4.SPECS[tid]
        assert required_spec_keys <= set(spec)
        assert spec["task_id"] == spec["case_id"] == spec["kind"] == tid
        assert spec["law"] in {"law4.1", "law4.2", "law4.4"}
        assert isinstance(spec["defects"], list) and len(spec["defects"]) >= 1
        assert "learnable" not in spec["prior"]
        assert spec["prior"]["kind"] == "mog" and float(spec["prior"]["sigma"]) > 0.0
        assert spec["prior"]["standardize"] is False
        assert int(spec["steps"]) >= 24
        assert int(spec["observations"]) == 24
        assert int(spec["minimum_stable_checks"]) == 5
        assert spec["sampling_law"] == "public_prior_without_output_noise"
        assert spec["view_kind"] == "vector"
        json.dumps(spec, allow_nan=False)

        rows = fl4.THRESHOLDS[tid]
        assert rows[0] == ["finite_fraction", "==", 1.0]
        names = [r[0] for r in rows]
        assert len(names) == len(set(names)), f"Duplicate metric in THRESHOLDS[{tid!r}]"
        # Problem-only / method-agnostic enforcement: zero internal optimizer/prior/critic metrics in THRESHOLDS!
        forbidden = set(names) & _FORBIDDEN_INTERNAL_THRESHOLD_KEYS
        assert not forbidden, f"Internal/solution metric(s) forbidden in THRESHOLDS[{tid!r}]: {forbidden}"
        for name, op, bound in rows:
            assert isinstance(name, str) and name and not name.startswith("diag_")
            assert op in {"<=", ">=", "=="}
            assert isinstance(bound, (int, float)) and not isinstance(bound, bool) and math.isfinite(float(bound))
        assert callable(getattr(fl4, f"score_{tid}"))

    cases = fl4.list_cases()
    assert [c["id"] for c in cases] == list(fl4.TASKS)
    for case in cases:
        api_contract.validate_case(case)
        assert case["protocol_seed"] == 0
        assert case["default_recipe"] == "unified_v14"
        assert case["legacy_ids"] == [f"four-laws-{case['id']}"]

    # Guardrail: all 9 Four Laws cases are registered alongside the 205 existing cases
    discovered = api_contract.discover()
    assert len(discovered) == 214
    assert set(fl4.TASKS) <= set(discovered)


def test_signed_margins_and_oracle_destructive_controls() -> None:
    """Verify exact signed-margin algebra plus fast CPU oracle PASS and destructive FAIL selectivity."""
    t0 = time.monotonic()
    controls = fl4.oracle_controls()
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"oracle_controls took too long on CPU: {elapsed:.2f}s"

    for prefix in ("fl_law41_", "fl_law42_swap_", "fl_law42_resbn_", "fl_law44_"):
        sub = {k: v for k, v in controls.items() if k.startswith(prefix)}
        pos = [k for k, v in sub.items() if v["expected_pass"]]
        neg = [k for k, v in sub.items() if not v["expected_pass"]]
        assert len(pos) >= 1, f"Missing positive oracle control for {prefix}"
        assert len(neg) >= 2, f"Need >=2 destructive controls for {prefix}"

    for name, ctrl in controls.items():
        assert ctrl["passed"] is ctrl["expected_pass"], f"Control {name} mismatch: {ctrl}"
        assert math.isfinite(ctrl["min_margin"])
        if ctrl["expected_pass"]:
            assert ctrl["min_margin"] >= 0.0, f"PASS control {name} had negative margin: {ctrl}"
            assert ctrl["failed_bounds"] == []
        else:
            assert ctrl["min_margin"] < 0.0, f"FAIL control {name} had non-negative margin: {ctrl}"
            assert len(ctrl["failed_bounds"]) >= 1

    m_missing = fl4.signed_margins({}, "fl_law41_trust_rare1pct_tanh")
    assert all(v == -10.0 for v in m_missing.values())
    m_nan = fl4.signed_margins({"finite_fraction": float("nan"), "modes": float("inf")}, "fl_law41_trust_rare1pct_tanh")
    assert m_nan["finite_fraction"] == -10.0 and m_nan["modes"] == -10.0
    m_eq = fl4.signed_margins({"finite_fraction": 1.0, "modes": 4}, "fl_law41_trust_rare1pct_tanh")
    assert m_eq["finite_fraction"] == 0.0
    assert m_eq["modes"] == pytest.approx(-0.2, rel=1e-6)


def test_resbn_linear_and_conv1x1_twin_mathematical_isomorphism() -> None:
    """Verify ResBNConv1x1Net matches ResBNLinearNet to FP32 machine precision at initialization."""
    torch.manual_seed(0)
    g_lin = fl4.ResBNLinearNet(32, 128, 128, 8)
    g_conv = fl4.ResBNConv1x1Net(32, 128, 128, 8)
    fl4.copy_resbn_linear_to_conv1x1(g_lin, g_conv)

    d_lin = fl4.ResBNLinearNet(128, 1, 128, 8)
    d_conv = fl4.ResBNConv1x1Net(128, 1, 128, 8)
    fl4.copy_resbn_linear_to_conv1x1(d_lin, d_conv)

    z = torch.randn(16, 32)
    with torch.no_grad():
        y_lin = g_lin(z)
        y_conv = g_conv(z)
        assert y_conv.shape == (16, 128, 1, 1)
        s_lin = d_lin(y_lin)
        s_conv = d_conv(y_conv)
    assert float((y_lin - y_conv.squeeze(-1).squeeze(-1)).abs().max().item()) < 1e-5
    assert float((s_lin - s_conv).abs().max().item()) < 1e-5


@pytest.mark.parametrize("task_id", fl4.TASKS)
def test_bounded_public_api_training_observe_purity_and_gif(task_id: str, tmp_path: Path) -> None:
    """Run bounded public GANTrainer updates, verify observe purity, and render multi-frame GIF."""
    steps = 8 if task_id in ("fl_law41_trust_rare1pct_tanh", "fl_law42_swap_deep_critic_16l") else 4
    fixture = fl4.build_case(task_id, device="cpu", seed=0, recipe_name="unified_v14", max_steps=steps)
    assert isinstance(fixture.recipe, Recipe)
    assert fixture.completed_steps == 0
    assert fixture.api_components

    if task_id == "fl_law42_deep_resbn_and_conv1x1":
        assert "conv1x1" in fixture.aux_trainers
    elif task_id == "fl_law44_highd_sparse_prior_and_shock":
        assert "scale_shock" in fixture.aux_trainers
    else:
        assert fixture.aux_trainers == {}

    g_param0 = next(p.detach().clone() for p in fixture.trainer.G.parameters() if p.requires_grad)
    rng_before = torch.get_rng_state().clone()
    state_before = state_digest(fixture.state_dict())
    obs0 = api_contract.validate_observation(fixture.observe(n=128, seed=10000))
    assert torch.equal(rng_before, torch.get_rng_state())
    assert state_digest(fixture.state_dict()) == state_before
    assert "margins" in obs0 and "min_margin" in obs0
    assert obs0["metrics"]["min_margin"] == pytest.approx(obs0["min_margin"], abs=1e-12)
    assert all(math.isfinite(float(v)) for v in obs0["margins"].values())
    json.dumps({"metrics": obs0["metrics"], "margins": obs0["margins"]}, allow_nan=False)

    mid = steps // 2
    records = [{**obs0, "step": 0}]
    for step in range(1, steps + 1):
        out = fixture.step()
        assert all(math.isfinite(float(v)) for v in out.values() if isinstance(v, (int, float)))
        if step in (mid, steps):
            obs = api_contract.validate_observation(fixture.observe(n=128, seed=10000))
            records.append({**obs, "step": step})

    assert fixture.completed_steps == steps
    g_param_after = next(p.detach().clone() for p in fixture.trainer.G.parameters() if p.requires_grad)
    assert not torch.equal(g_param0, g_param_after), f"{task_id}: generator weights did not move"
    assert len(fixture.batch_sequence_sha256) == 64

    gif_path = tmp_path / f"{task_id}.gif"
    fl4.render_task_gif(
        fixture.case,
        records,
        gif_path,
        full_budget=False,
        requested_steps=steps,
        final_verdict="FAIL",
    )
    assert gif_path.is_file() and gif_path.stat().st_size > 1024
    with Image.open(gif_path) as gif:
        assert gif.n_frames == len(records) == 3


@pytest.mark.parametrize("task_id", fl4.TASKS)
def test_method_agnostic_non_v14_recipe_execution(task_id: str) -> None:
    """Confirm every Law 4 task executes cleanly under non-v14 recipes (minimal & bispf) with zero AttributeError."""
    for recipe_name in ("minimal", "bispf"):
        fix = fl4.build_case(task_id, device="cpu", seed=0, recipe_name=recipe_name, max_steps=2)
        fix.step()
        obs = api_contract.validate_observation(fix.observe(n=64, seed=10000))
        assert obs["metrics"]["finite_fraction"] == 1.0
        assert all(math.isfinite(float(v)) for v in obs["margins"].values())


@pytest.mark.parametrize("task_id", fl4.TASKS)
def test_checkpoint_save_restore_exact_continuation(task_id: str) -> None:
    """Verify bit-exact continuation across state_dict / load_state_dict in a fresh fixture."""
    fix_a = fl4.build_case(task_id, device="cpu", seed=0, recipe_name="unified_v14", max_steps=4)
    for _ in range(2):
        fix_a.step()

    saved = fix_a.state_dict()
    fix_b = fl4.build_case(task_id, device="cpu", seed=0, recipe_name="unified_v14", max_steps=4)
    fix_b.load_state_dict(saved)

    for _ in range(2):
        fix_a.step()
        fix_b.step()

    for p_a, p_b in zip(fix_a.trainer.G.parameters(), fix_b.trainer.G.parameters()):
        assert torch.equal(p_a, p_b)
    for p_a, p_b in zip(fix_a.trainer.D.parameters(), fix_b.trainer.D.parameters()):
        assert torch.equal(p_a, p_b)
    assert torch.equal(unwrap_prior(fix_a.trainer.prior).z, unwrap_prior(fix_b.trainer.prior).z)
    for k in fix_a.aux_trainers:
        for p_a, p_b in zip(fix_a.aux_trainers[k].G.parameters(), fix_b.aux_trainers[k].G.parameters()):
            assert torch.equal(p_a, p_b)

    obs_a = fix_a.observe(n=128, seed=10000)
    obs_b = fix_b.observe(n=128, seed=10000)
    assert obs_a["metrics"] == obs_b["metrics"]
    assert fix_a.batch_sequence_sha256 == fix_b.batch_sequence_sha256
    assert fix_a.seed == fix_b.seed

    # Mismatched recipe checkpoint must be rejected fail-closed before mutating state
    fix_other = fl4.build_case(task_id, device="cpu", seed=0, recipe_name="minimal", max_steps=4)
    with pytest.raises(ValueError, match="recipe mismatch"):
        fix_other.load_state_dict(saved)


@pytest.mark.parametrize("task_id", fl4.TASKS)
def test_forge_formulation_context_integration_and_rng_audit(task_id: str) -> None:
    """Verify FourLawsFixture binds cleanly to Forge FormulationContext with 0 unintended eval RNG deviations."""
    from experiments.forge.api import FormulationContext

    spec = fl4.SPECS[task_id]
    res = spec["resources"]
    ctx = FormulationContext(
        recipe_preset="unified_v14",
        recipe_overrides={
            "num_particles": int(res["num_particles"]),
            "z_dim": int(res["z_dim"]),
            "batch_size": int(res["batch_size"]),
        },
        prior=dict(spec["prior"]),
        seed=0,
        device="cpu",
        prior_contract="recipe_owned_v1",
    )
    fix = fl4.build_case(task_id, device="cpu", seed=0, max_steps=4, components=ctx)
    fix.step()

    before = ctx.streams.audit()
    global_before = torch.get_rng_state().clone()
    obs = api_contract.validate_observation(fix.observe(n=64, seed=10000))
    assert obs["metrics"]["finite_fraction"] == 1.0
    assert torch.equal(global_before, torch.get_rng_state())

    allowed = {
        key
        for key, binding in ctx.streams.manifest()["bindings"].items()
        if binding["family"] == "eval"
    }
    after = ctx.streams.audit()
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    unintended = changed - allowed
    assert not unintended, f"{task_id}: observe() perturbed non-eval FormulationContext streams: {unintended}"

