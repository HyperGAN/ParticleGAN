"""Check the transparent metadata-only adaptation without running training."""
import ast
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helper():
    spec = importlib.util.spec_from_file_location("compatibility_adapter_test", ROOT / "reports/forge/bcap-three-phase/verify_compatibility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_preserves_every_assertion_and_adds_only_checked_inactive_defaults():
    module = helper()
    old = (ROOT / module.INACTIVE).read_bytes()
    new, diff = module.adapter_source(old)
    additions = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removals = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert additions == ['+    "critic_step_mode": "none",', '+    "optimizer_svd_backend": "native",']
    assert not removals
    assert new.count(b"assert ") == old.count(b"assert ")
    assert (ROOT / module.INACTIVE).read_bytes() == old


def test_adapter_refuses_modified_original_assertions():
    module = helper()
    old = (ROOT / module.INACTIVE).read_bytes().replace(b"assert torch.equal(left, right)", b"assert True")
    with pytest.raises(ValueError, match="immutable original comparator changed"):
        module.adapter_source(old)


def test_scoped_adapter_reaches_subprocess_program_and_restores_on_failure(tmp_path):
    from conftest import _archived_inactive_comparison_scope
    from tests.archived_forge_contracts import COMMITS
    module = helper()
    original_path = ROOT / module.INACTIVE
    original = original_path.read_bytes()
    fields = {"constraint_geometry_mode": "none"}
    imported = SimpleNamespace(__file__=str(original_path), ROOT=ROOT,
        DEVELOP=COMMITS["bcap_legacy_questions"], NEW_INACTIVE_FIELDS=fields)
    with pytest.raises(RuntimeError, match="fixture teardown control"):
        with _archived_inactive_comparison_scope(imported, tmp_path):
            program = Path(imported.__file__)
            assert program.parent == tmp_path
            tree = ast.parse(program.read_text())
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and getattr(node.targets[0], "id", None) == "NEW_INACTIVE_FIELDS")
            defaults = ast.literal_eval(assignment.value)
            assert {name: defaults[name] for name in module.INACTIVE_DEFAULTS} == module.INACTIVE_DEFAULTS
            receipt = json.loads((tmp_path / "adapter-receipt.json").read_text())
            checked = receipt["checked_inactive_defaults"]
            # 19 archived/review defaults, two later scalar defaults, plus 33 inactive RHBI defaults.
            assert len(checked) == 54
            assert checked["ebi_beta"] == 30.0 and checked["kappa_radial"] == 1.0
            assert checked["training_precision"] == "float32"
            assert checked["bispf_gate_overrides"] == [] and defaults["bispf_gate_overrides"] == ()
            as_json = lambda value: json.loads(json.dumps(value))  # the receipt is JSON
            assert as_json({name: defaults[name] for name in checked}) == checked
            assert as_json(imported.NEW_INACTIVE_FIELDS) == {**fields, **checked}
            # The subprocess keeps current code but asks the original task
            # questions in a private checkout; no live declarations are edited.
            assert imported.ROOT == tmp_path / "legacy-question-source"
            assert (imported.ROOT / "particlegan/recipes.py").read_bytes() == (ROOT / "particlegan/recipes.py").read_bytes()
            source_binding = json.loads((imported.ROOT / "legacy-software-source-bindings.json").read_text())
            assert source_binding["original_task_source_commit"] == imported.DEVELOP
            assert receipt["task_question_source_commit"] == imported.DEVELOP
            assert receipt["original_sha256"] == hashlib.sha256(original).hexdigest()
            assert receipt["adapted_sha256"] == hashlib.sha256(program.read_bytes()).hexdigest()
            raise RuntimeError("fixture teardown control")
    assert imported.__file__ == str(original_path)
    assert imported.ROOT == ROOT
    assert imported.NEW_INACTIVE_FIELDS is fields
    assert original_path.read_bytes() == original


@pytest.mark.parametrize("field,value", [("ebi_beta", 15.0), ("kappa_radial", 2.0)])
def test_scoped_adapter_refuses_nondefault_metadata(tmp_path, monkeypatch, field, value):
    from dataclasses import replace
    import particlegan
    from conftest import _archived_inactive_comparison_scope
    from tests.archived_forge_contracts import COMMITS

    changed = replace(particlegan.Recipe.from_dict({}), **{field: value})
    monkeypatch.setattr(particlegan, "Recipe", SimpleNamespace(from_dict=lambda fields: changed))
    imported = SimpleNamespace(__file__=str(ROOT / helper().INACTIVE), ROOT=ROOT,
        DEVELOP=COMMITS["bcap_legacy_questions"], NEW_INACTIVE_FIELDS={})
    with pytest.raises(AssertionError, match=field):
        with _archived_inactive_comparison_scope(imported, tmp_path):
            pytest.fail("nondefault recipe metadata was discarded")
