"""Verify non-scientific reporting/publishing modules stay isolated from Forge source manifests."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import shutil

import pytest

from experiments.forge import sources

ROOT = Path(__file__).resolve().parents[1]


def _parse_imports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8", errors="ignore"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level > 0:
                parts = list(py_file.relative_to(ROOT).parent.parts)
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                base = ".".join(parts)
                full = f"{base}.{mod}" if mod else base
                imported.add(full)
                for alias in node.names:
                    imported.add(f"{full}.{alias.name}")
            else:
                imported.add(mod)
                for alias in node.names:
                    imported.add(f"{mod}.{alias.name}")
    return imported


def test_non_scientific_source_paths_are_never_imported_by_scientific_code():
    """Semantic control: excluded reporting/publishing modules cannot affect training or grading."""
    excluded = set(sources.NON_SCIENTIFIC_SOURCE_PATHS)
    assert excluded, "NON_SCIENTIFIC_SOURCE_PATHS must be non-empty"
    for rel in excluded:
        assert (ROOT / rel).is_file(), f"excluded path missing: {rel}"

    excluded_mods = {rel[:-3].replace("/", ".") for rel in excluded}
    excluded_short = {mod.split(".")[-1] for mod in excluded_mods}

    # Check every Python file inside particlegan/, benchmarks/, lib/, and every
    # non-excluded scientific module inside experiments/forge/ (excluding CLI __main__.py).
    scientific_files: list[Path] = []
    for folder in ("particlegan", "benchmarks", "lib"):
        scientific_files.extend(sorted((ROOT / folder).rglob("*.py")))
    for path in sorted((ROOT / "experiments/forge").glob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in excluded or rel == "experiments/forge/__main__.py":
            continue
        # Only core execution/evaluation/planning/queue/contract/adapter modules.
        if path.stem in {
            "api",
            "contracts",
            "execution_policy",
            "hostprofiles",
            "knowledge",
            "planning",
            "queue",
            "runner",
            "sources",
            "studies",
            "taskrecipes",
            "views",
            "worker",
        } or path.stem.endswith("_adapter"):
            scientific_files.append(path)

    violations: list[str] = []
    for py_file in scientific_files:
        deps = _parse_imports(py_file)
        for dep in deps:
            if dep in excluded_mods or any(dep.startswith(m + ".") for m in excluded_mods):
                violations.append(f"{py_file.relative_to(ROOT)} imports {dep}")
            elif dep.startswith("experiments.forge.") and dep.split(".")[-1] in excluded_short:
                violations.append(f"{py_file.relative_to(ROOT)} imports {dep}")
    assert not violations, "Scientific modules must not import excluded reporting files:\n" + "\n".join(violations)


def test_editing_non_scientific_file_preserves_source_digest(tmp_path):
    """Editing hf_publish.py / leaderboard_media.py must not invalidate scientific source_digest."""
    for folder in ("particlegan", "experiments/forge", "benchmarks", "lib"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    (tmp_path / "particlegan/core.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "experiments/forge/hf_publish.py").write_text("V = 1\n", encoding="utf-8")
    (tmp_path / "experiments/forge/leaderboard_media.py").write_text("M = 1\n", encoding="utf-8")

    before = sources.inspect_source(tmp_path)
    assert "particlegan/core.py" in before["files"]
    assert "experiments/forge/hf_publish.py" not in before["files"]
    assert "experiments/forge/leaderboard_media.py" not in before["files"]

    # Mutate non-scientific reporting files -> digest MUST stay identical.
    (tmp_path / "experiments/forge/hf_publish.py").write_text("V = 999\n", encoding="utf-8")
    (tmp_path / "experiments/forge/leaderboard_media.py").write_text("M = 999\n", encoding="utf-8")
    after_non_sci = sources.inspect_source(tmp_path)
    assert after_non_sci["digest"] == before["digest"]

    # Mutate scientific file -> digest MUST change.
    (tmp_path / "particlegan/core.py").write_text("X = 2\n", encoding="utf-8")
    after_sci = sources.inspect_source(tmp_path)
    assert after_sci["digest"] != before["digest"]


def test_verify_source_snapshot_rejects_tampered_file(tmp_path):
    """Semantic control: deliberate damage to any snapshot file must fail verification."""
    spec = importlib.util.spec_from_file_location(
        "regen_mod_test", ROOT / "reports/forge/regenerate_technique_inventory.py"
    )
    regen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen)

    snap = tmp_path / "snap"
    (snap / "particlegan").mkdir(parents=True)
    f = snap / "particlegan/sample.py"
    f.write_text("A = 1\n", encoding="utf-8")
    from experiments.forge.contracts import file_hash, stable_hash

    files = {"particlegan/sample.py": file_hash(f)}
    manifest = {"schema_version": 1, "digest": stable_hash(files), "files": files}
    (snap / "forge-source.json").write_text(
        regen._json_text(manifest), encoding="utf-8"
    )

    # Deliberately tamper with the file -> verification MUST raise ValueError.
    tampered = tmp_path / "snap_tampered"
    shutil.copytree(snap, tampered)
    (tampered / "particlegan/sample.py").write_text("A = 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        regen._verify_source_snapshot(
            ROOT,
            tampered,
            {manifest["digest"]: manifest},
            origin_commit="HEAD",
            study_commit="HEAD",
        )
