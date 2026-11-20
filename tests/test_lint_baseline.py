"""Exercise the actual lint gate without changing repository sources."""
import json
from pathlib import Path
import subprocess

import pytest

import scripts.check_lint as check_lint
from scripts.check_lint import FROZEN_F821_FALSE_POSITIVES, PATHS, findings, main


def _workspace(root):
    for name in PATHS:
        (root / name).mkdir()
    return root / "scripts" / "probe.py"


def test_lint_gate_rejects_new_findings_and_requires_resolved_exemptions_to_shrink(tmp_path, capsys):
    probe = _workspace(tmp_path)
    args = ["--root", str(tmp_path)]
    assert main([*args, "--update-baseline"]) == 0
    probe.write_text("import math\n")
    assert main(args) == 1
    assert "NEW:" in capsys.readouterr().out
    assert main([*args, "--update-baseline"]) == 0
    assert main(args) == 0
    probe.unlink()
    assert main(args) == 1
    assert "RESOLVED:" in capsys.readouterr().out
    assert main([*args, "--update-baseline"]) == 0
    assert main(args) == 0


def test_lint_baseline_survives_unrelated_line_moves(tmp_path):
    probe = _workspace(tmp_path)
    source = "from math import sin\nfrom math import sin\n"
    probe.write_text(source)
    args = ["--root", str(tmp_path)]
    assert main([*args, "--update-baseline"]) == 0
    probe.write_text("\n" * 10 + source)
    assert main(args) == 0


def test_undefined_names_cannot_be_accepted_by_updating_a_baseline(tmp_path, capsys):
    probe = _workspace(tmp_path)
    args = ["--root", str(tmp_path)]
    assert main([*args, "--update-baseline"]) == 0
    baseline = tmp_path / ".ruff-baseline.json"
    original = baseline.read_bytes()
    probe.write_text("answer = undefined_name\n")
    assert main(args) == 1
    assert "STRICT:" in capsys.readouterr().out
    assert main([*args, "--update-baseline"]) == 1
    assert baseline.read_bytes() == original


def test_forged_undefined_name_exemption_is_rejected(tmp_path, capsys):
    _workspace(tmp_path)
    args = ["--root", str(tmp_path)]
    assert main([*args, "--update-baseline"]) == 0
    baseline = tmp_path / ".ruff-baseline.json"
    document = json.loads(baseline.read_text())
    document["findings"] = {"scripts/probe.py": {"F821": {"Undefined name `unknown`": 1}}}
    baseline.write_text(json.dumps(document))
    assert main(args) == 1
    assert "baseline contains a strict code" in capsys.readouterr().out


def test_frozen_false_positive_exception_cannot_hide_a_changed_source(tmp_path, capsys):
    _workspace(tmp_path)
    root = Path(__file__).resolve().parents[1]
    frozen_path = next(iter(FROZEN_F821_FALSE_POSITIVES))
    target = tmp_path / frozen_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((root / frozen_path).read_bytes())
    args = ["--root", str(tmp_path)]
    assert main([*args, "--update-baseline"]) == 0
    assert main(args) == 0
    baseline = (tmp_path / ".ruff-baseline.json").read_bytes()
    target.write_text(target.read_text() + "\nnew_result = new_undefined_name\n")
    assert main([*args, "--update-baseline"]) == 1
    assert "new_undefined_name" in capsys.readouterr().out
    assert (tmp_path / ".ruff-baseline.json").read_bytes() == baseline


def test_missing_ruff_or_empty_output_raises_clear_runtime_error(tmp_path, monkeypatch):
    _workspace(tmp_path)
    monkeypatch.setattr(
        check_lint.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout="", stderr="No module named ruff\n"
        ),
    )
    with pytest.raises(RuntimeError, match="No module named ruff"):
        findings(tmp_path)
