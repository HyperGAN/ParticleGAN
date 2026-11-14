"""Forge boundary review (2026-10-25): cache keys, explicit caches, trainer hook, graders,
fixed leaderboard denominators, selection/holdout claims and calibration bounds."""
import hashlib
import inspect
import subprocess
import sys
import threading
from pathlib import Path
from types import MethodType

import pytest
import torch
from torch import nn

from experiments.forge import (calibration, configuration_search, decision_contracts, knowledge, planning,
                               resolution_scope, tier1_policy, views)
from experiments.forge.contracts import atomic_json
from experiments.forge.leaderboard_contract import (budget_text, family_search_budget, fixed_outcomes,
                                                    holdout_claim, load_fixed_test_list, selection_lineage)

ROOT = Path(__file__).resolve().parents[1]


# 1. Finding #23: decision-contract binding memo keys include override values.

def _candidate(lr):
    return {"id": "same-id", "recipe_preset": "k3p", "recipe_overrides": {"lr": lr},
            "resolved_recipe": {"lr": lr}, "prior": {"kind": "particle"}}


def test_binding_signature_distinguishes_tunable_override_values():
    low, high = _candidate(1e-3), _candidate(2e-3)
    # Preflight sharing keeps its historical tunable-flag contract ...
    assert planning._blocker_sig(low) == planning._blocker_sig(high)
    # ... but bindings report effective values, so their key must differ.
    assert planning._binding_sig(low) != planning._binding_sig(high)


def test_cached_bindings_never_reuse_a_different_override_value(tmp_path, monkeypatch):
    calls = []

    def resolve(candidate, tasks, protocol, root):
        calls.append(candidate["recipe_overrides"]["lr"])
        return {key: {name: {"lr": candidate["recipe_overrides"]["lr"]} for name in tasks}
                for key in ("recipe", "prior", "initialization", "budget", "sampling", "host",
                            "evaluation", "task_identity")}

    monkeypatch.setattr(decision_contracts, "_resolve_bindings", resolve)
    tasks = {"t1": {"id": "t1"}}
    with planning.resolution_cache(tmp_path):
        first = decision_contracts._bindings(_candidate(1e-3), tasks, {}, tmp_path)
        second = decision_contracts._bindings(_candidate(2e-3), tasks, {}, tmp_path)
        again = decision_contracts._bindings(_candidate(1e-3), tasks, {}, tmp_path)
    assert first["recipe"]["t1"] == {"lr": 1e-3} and second["recipe"]["t1"] == {"lr": 2e-3}
    assert again == first and calls == [1e-3, 2e-3]


# 2. resolution_cache binds an explicit cache object; no module globals are rebound.

def test_resolution_cache_rebinds_no_module_globals_and_is_context_local():
    before = (views.load_tasks, tier1_policy.validate, decision_contracts._bindings, planning.load_tasks)
    seen = {}
    with planning.resolution_cache(ROOT) as cache:
        assert (views.load_tasks, tier1_policy.validate, decision_contracts._bindings, planning.load_tasks) == before
        assert resolution_scope.active(ROOT) is cache and resolution_scope.active("/elsewhere") is None
        with planning.resolution_cache(ROOT) as inner:
            assert inner is cache
        first, second = views.load_tasks(ROOT), views.load_tasks(ROOT)
        assert first is not second and first == second
        name = next(iter(first))
        assert first[name] is second[name]  # memoized catalogue
        worker = threading.Thread(target=lambda: seen.update(active=resolution_scope.active(ROOT)))
        worker.start()
        worker.join()
    assert seen == {"active": None}  # another thread never observes this cache
    assert resolution_scope.active(ROOT) is None
    assert views.load_tasks(ROOT)[name] is not first[name]


# 3. Public GANTrainer serving hook replaces the exec-patched live-storage step.

def _trainer(**options):
    torch.manual_seed(0)
    recipe = __import__("particlegan").get_recipe("e22", num_particles=12, z_dim=3, batch_size=6,
                                                  total_steps=None, serve_average=4.0)
    generator = nn.Sequential(nn.Linear(3, 8), nn.LeakyReLU(.2), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(.2), nn.Linear(8, 1))
    streams = {name: torch.Generator().manual_seed(seed) for seed, name in enumerate(
        ("latent_generator", "penalty_generator", "noise_generator", "input_noise_generator", "eval_generator"), 1)}
    trainer = __import__("particlegan").GANTrainer(recipe, generator, critic, seed=0, max_steps=30,
                                                   **streams, **options)
    # Force the served average to settle early so the serving swap is exercised.
    trainer.policy._serve_settled = lambda: trainer.completed_steps >= 5
    return trainer


def _trajectory(trainer, steps=15):
    data, digest = torch.Generator().manual_seed(99), hashlib.sha256()
    for _ in range(steps):
        out = trainer.step(torch.randn(6, 2, generator=data))
        for key in sorted(k for k in out if torch.is_tensor(out[k])):
            digest.update(out[key].numpy().tobytes())
        for module in (trainer.G, trainer.D, trainer.prior):
            for key, value in sorted(module.state_dict().items()):
                digest.update(key.encode() + value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_serve_after_step_false_is_bit_identical_to_the_legacy_live_step_overlay():
    from particlegan import training
    from experiments.forge.canonical_image_adapter import derive_live_step
    source = Path(training.__file__).read_text()
    hooked = "        if self.serve_after_step:\n            self._serve_apply()\n"
    assert source.count(hooked) == 2  # end of _step and end of _load_state_dict
    legacy = source.replace(hooked, "        self._serve_apply()\n")  # the overlay's original input form
    namespace = dict(vars(training))
    exec(compile(derive_live_step(legacy), "<legacy-live-step>", "exec"), namespace)
    overlay = _trainer()
    overlay._step = MethodType(namespace["_step"], overlay)
    live = _trajectory(_trainer(serve_after_step=False))
    assert live == _trajectory(overlay)
    assert live != _trajectory(_trainer())  # the default still serves the settled average


def test_serve_after_step_is_validated_and_canonical_image_owner_no_longer_execs_step():
    with pytest.raises(TypeError, match="serve_after_step"):
        _trainer(serve_after_step=1)
    from experiments.forge import canonical_image_adapter
    owner_source = inspect.getsource(canonical_image_adapter.construct_owner)
    assert "serve_after_step=False" in owner_source
    assert "derive_live_step(" not in owner_source and "MethodType" not in owner_source
    assert "_step = " not in owner_source



def test_serve_after_step_false_also_leaves_the_training_iterate_after_resume():
    def params(trainer):
        return [value.clone() for value in trainer.G.state_dict().values()]

    source = _trainer(serve_after_step=False)
    _trajectory(source, steps=10)
    live, state = params(source), source.state_dict()
    # The fixture forces settling through a patched ``_serve_settled``; the checkpoint's
    # recorded source is re-derived from table-tester state, which never settled here.
    state["policy"]["served_source"] = "fast"
    resumed = _trainer(serve_after_step=False)
    resumed.load_state_dict(state)
    assert resumed._fast is None
    assert all(torch.equal(a, b) for a, b in zip(params(resumed), live))
    served = _trainer()
    served.load_state_dict(state)  # the default still serves the settled average on resume
    assert served._fast is not None
    assert not all(torch.equal(a, b) for a, b in zip(params(served), live))

# 4. Graders live in benchmarks/, byte-identical to the archived evidence source.

def test_ring_grader_imports_from_benchmarks_with_archived_identity():
    receipt = views.CONVERGENCE_GATE_RECEIPT
    gate = views._convergence_class()
    module = sys.modules[gate.__module__]
    assert gate.__module__ == receipt["module"] == "benchmarks.toy100.k3p_convergence_gate"
    assert Path(module.__file__).resolve() == (ROOT / receipt["source"]).resolve()
    new, archived = (ROOT / receipt["source"]).read_bytes(), (ROOT / receipt["archived_source"]).read_bytes()
    assert new == archived and hashlib.sha256(new).hexdigest() == receipt["sha256"]
    blob = subprocess.run(["git", "hash-object", str(ROOT / receipt["source"])], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert blob == receipt["archived_git_blob"]
    # Only the grader's own module and the retired exec-overlay name matter here;
    # other tests legitimately load reports/ copies under unrelated names.
    assert not Path(module.__file__).resolve().is_relative_to(ROOT / "reports")
    legacy = sys.modules.get("_forge_original_convergence_gate")
    assert legacy is None or not Path(legacy.__file__).resolve().is_relative_to(ROOT / "reports")
    for name in ("ring_hold", "ring_extension"):
        task = views.load_tasks(ROOT)[name]
        assert task["evaluation"]["sources"][receipt["archived_source"]] == receipt["sha256"]


# 5. Fixed, dated leaderboard denominators and declared tuning budgets.

def test_fixed_denominator_counts_held_and_missing_tests():
    tests = ["a", "b", "held_fail", "dropped"]
    counts = fixed_outcomes({"a": "PASS", "b": "BLOCKED", "held_fail": "FAIL"}, tests, ["held_fail", "dropped"])
    assert counts == {"total": 4, "passed": 1, "failed": 1, "not_measured": 1, "other": {"BLOCKED": 1},
                      "held": {"total": 2, "passed": 0, "failed": 1, "not_measured": 1}}


def test_committed_inventory_reports_against_the_fixed_dated_list():
    fixed = load_fixed_test_list(ROOT)
    prior_list = load_fixed_test_list(ROOT, "configs/forge/leaderboards/technique-inventory-full-suite-2026-11-06.json")
    previous = load_fixed_test_list(ROOT, "configs/forge/leaderboards/technique-inventory-full-suite-2026-10-25.json")
    assert previous and len(previous["tests"]) == 49
    assert prior_list and prior_list["dated"] == "2026-11-06" and len(prior_list["tests"]) == 61
    assert fixed and fixed["dated"] == "2026-11-12" and len(fixed["tests"]) == 64
    view = views.load_view(ROOT, "discriminator_stability")
    assert set(fixed["tests"]) - set(previous["tests"]) == {a["task"] for a in view["assignments"] if a["importance"] == "stretch"}
    text = (ROOT / "reports/forge/technique-inventory.md").read_text()
    section = text.split("## Full-suite results — failures included", 1)[1].split("## ", 1)[0]
    assert f"fixed test list `{fixed['id']}`" in section
    rows = [line for line in section.splitlines() if line.startswith("| [")]
    assert rows and all(line.split(" | ")[6] == str(len(fixed["tests"])) for line in rows)
    assert all("declared candidate card" in line for line in rows)


def test_search_budget_labels_untuned_and_searched_families(tmp_path):
    atomic_json(tmp_path / "configs/forge/trainer-families.json", {"families": [
        {"id": "base", "candidates": ["base-v1"]}, {"id": "variant", "reporting_family": "base", "candidates": []},
        {"id": "external", "candidates": ["external-v1"]}], "historical_families": []})
    atomic_json(tmp_path / "configs/forge/searches/s1.json",
                {"id": "s1", "trainer_family": "variant", "tuning_through_tier": 1,
                 "grid": {"lr": [1, 2], "d_lr_mult": [1, 2, 3]}})
    atomic_json(tmp_path / "configs/forge/searches/s2.json",
                {"id": "s2", "trainer_family": "base", "tuning_through_tier": 2, "grid": [{"lr": [1]}, {"lr": [2]}]})
    atomic_json(tmp_path / "configs/forge/configurations/cfg.json",
                {"id": "cfg", "trainer_family": "variant", "search_study_id": "s1"})
    atomic_json(tmp_path / "configs/forge/ideas/child.json", {"id": "child", "parent": "cfg"})
    base = family_search_budget(tmp_path, "base")
    assert (base["searches"], base["configurations"], base["declared_candidates"]) == (2, 8, 2)
    lineage = selection_lineage(tmp_path, "child")
    assert lineage == {"search_id": "s1", "configuration": "cfg", "tuning_through_tier": 1, "configurations": 6}
    assert "descends from `s1` (tuned on Tier ≤1)" in budget_text(base, lineage)
    external = budget_text(family_search_budget(tmp_path, "external"), selection_lineage(tmp_path, "external-v1"))
    assert external.startswith("untuned: no registered configuration search; 1 declared candidate card")


# 7. Selection versus qualification: tuning-tier PASS is never independent confirmation.

def test_holdout_claim_separates_tuning_and_untuned_tiers():
    passed = holdout_claim({1: ["PASS", "PASS"], 2: ["PASS"], 3: ["PASS"]}, 1)
    assert passed["status"] == "PASS" and passed["holdout_tiers"] == [2, 3]
    assert passed["independent_confirmation"] == "not_performed" and "not independent confirmation" in passed["claim"]
    failed = holdout_claim({"1": ["PASS"], "2": ["PASS", "FAIL"]}, 1)
    assert failed["status"] == "FAIL" and "did not pass the untuned holdout" in failed["claim"]
    assert holdout_claim({1: ["PASS"], 2: ["UNKNOWN"]}, 1)["status"] == "not_measured"
    only_tuning = holdout_claim({1: ["PASS"]}, 1)
    assert only_tuning["status"] == "none_declared" and "no untuned holdout" in only_tuning["claim"]



def test_holdout_treats_invalid_like_the_fixed_denominator_does():
    # fixed_outcomes reports INVALID under "other", never "failed"; the holdout agrees.
    assert fixed_outcomes({"a": "INVALID"}, ["a"], [])["failed"] == 0
    assert holdout_claim({1: ["PASS"], 2: ["PASS", "INVALID"]}, 1)["status"] == "not_measured"


def test_holdout_tier_statuses_count_each_required_task_once():
    from experiments.forge.family_reports import _cohort_tier_statuses
    shared = {"task": "shared", "qualification_tier": 2, "importance": "required"}
    progress = {"diagnostic_views": ["diag"], "views": [
        {"id": "a", "assignments": [shared, {"task": "early", "qualification_tier": 1, "importance": "required"}]},
        {"id": "b", "assignments": [shared, {"task": "early", "qualification_tier": 3, "importance": "required"},
                                    {"task": "held", "qualification_tier": 2, "importance": "held"}]},
        {"id": "diag", "assignments": [{"task": "diag_only", "qualification_tier": 2, "importance": "required"}]}]}
    cohort = {"tasks": {"shared": {"status": "PASS"}, "early": {"status": "PASS"}, "held": {"status": "FAIL"}},
              "tiers": {"2": {"counts": {"PASS": 2}}}}  # summed per-view counts double-count "shared"
    assert _cohort_tier_statuses(progress, cohort) == {1: ["PASS"], 2: ["PASS"]}
    claim = holdout_claim(_cohort_tier_statuses(progress, cohort), 1)
    assert claim["required_total"] == 1 and claim["required_passed"] == 1

def test_search_progression_states_the_selected_holdout():
    tasks = [{"task": "smoke", "qualification_tier": 1, "importance": "required", "gate_status": "PASS"},
             {"task": "quality", "qualification_tier": 2, "importance": "required", "gate_status": "FAIL"}]
    summary = {"tuning_through_tier": 1, "trials": [{"candidate_id": "c", "tasks": tasks}],
               "selection": {"selected_candidate_id": "c", "selection_complete": True, "qualified": True}}
    configuration_search._annotate_progression(summary)
    claim = summary["progression"]["selected_holdout"]
    assert summary["progression"]["outcome"] == "tuning_only_winner"
    assert claim["status"] == "FAIL" and claim["independent_confirmation"] == "not_performed"


def test_readout_claim_for_search_descendant(tmp_path):
    atomic_json(tmp_path / "configs/forge/searches/s1.json", {"id": "s1", "tuning_through_tier": 1, "grid": {"lr": [1]}})
    view = {"assignments": [{"task": "t1", "qualification_tier": 1, "importance": "required"},
                            {"task": "t2", "qualification_tier": 2, "importance": "required"},
                            {"task": "t3", "qualification_tier": 2, "importance": "held"}]}
    candidate = {"id": "cfg", "search_study_id": "s1"}
    claim = knowledge.readout_selection_claim(tmp_path, candidate, view,
                                              [{"task_id": "t1", "gate_status": "PASS"}])
    assert claim["search_id"] == "s1" and claim["status"] == "not_measured" and claim["required_total"] == 1
    assert knowledge.readout_selection_claim(tmp_path, {"id": "plain"}, view, []) is None


# 6. Calibration readouts carry exact Clopper-Pearson bounds next to point estimates.

def test_clopper_pearson_bounds_and_stats():
    assert calibration.clopper_pearson(0, 2)["upper"] == pytest.approx(1 - .025 ** .5, abs=1e-9)
    interval = calibration.clopper_pearson(1, 3)
    assert (interval["lower"], interval["upper"]) == pytest.approx((0.0084038, 0.9057007), abs=1e-6)
    assert calibration.clopper_pearson(0, 0) is None
    rows = [{"classification": kind, "reference": {"decision": ref}, "smoke": {"blocked_tasks": [],
             "error_or_incomplete_tasks": [], "decision": "PASS"}}
            for kind, ref in (("true_accept", "PASS"), ("true_reject", "FAIL"), ("true_reject", "FAIL"))]
    for row in rows:
        row["reference"].update(blocked_tasks=[], error_or_incomplete_tasks=[])
    stats = calibration._stats(rows)
    assert stats["false_accept_fraction"] == 0 and stats["false_accept_ci95"]["upper"] == pytest.approx(.841886, abs=1e-6)
    lines = calibration.bounds_lines(stats, {"maximum_false_accept_fraction": .1, "maximum_false_reject_fraction": 0})
    assert "0/2 = 0.000 (95% Clopper-Pearson [0.000, 0.842])" in lines[0] and "threshold ≤ 0.1" in lines[0]
