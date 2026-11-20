"""Current questions retain initial conditions and gates after ownership moves."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from pr_integration_compat import before_pr_integration


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "03466efa4de8271b9a2c964406dc6f4f0f260792"
MIGRATION = "9a0b16887798e6006d25b4d98e77f5754a49c09b"


def test_current_prior_questions_change_only_declared_ownership_and_source_bindings():
    migrated = []
    # Audit the exact 33-card migration; later registered questions own their
    # separate protocols and were never part of this before/after comparison.
    paths = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", MIGRATION,
         "--", "configs/forge/tasks"], cwd=ROOT, text=True).splitlines()
    for relative in paths:
        path = ROOT / relative
        current = before_pr_integration(ROOT, json.loads(path.read_text()), relative)
        original = json.loads(subprocess.check_output(
            ["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT))
        expected = deepcopy(original)
        assert expected["execution"]["prior"].pop("learnable") is True
        expected["execution"]["prior_contract"] = "recipe_owned_v1"
        expected["requires_capabilities"] = [name for name in expected["requires_capabilities"]
                                              if name != "learned_locations"]
        observed = deepcopy(current)
        if (current["evaluation"].get("evaluator_revision") or {}).get("id") == "tier2-grid-triad-clean-sampling-calibration-v1":
            assert current["id"] in {"grid100", "rotated100", "staggered100"}
            assert observed["evaluation"]["coverage_thresholds"]["min_precision"] == 0.75
            expected["evaluation"]["coverage_thresholds"]["min_precision"] = 0.75
        # Source rebinding authorizes current software only. Every architecture,
        # target, initial prior, sampling law, budget, cadence and bound stays.
        for card in (expected, observed):
            card["evaluation"].pop("sources", None)
            card["evaluation"].pop("evaluator_revision", None)
        assert observed == expected, current["id"]
        migrated.append(current["id"])
    assert len(migrated) == 33

