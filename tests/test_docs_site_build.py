"""Tests for the ParticleGAN MkDocs Material Research Portal builder and assets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

from scripts import build_docs_site as portal
from scripts import check_repo_bloat

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_docs_extra():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)
    docs_deps = data["project"]["optional-dependencies"]["docs"]
    joined = " ".join(docs_deps)
    assert "mkdocs>=" in joined
    assert "mkdocs-material>=" in joined
    assert "pymdown-extensions>=" in joined


def test_convert_github_alerts_and_preserve_code_fences():
    sample = (
        "Intro paragraph.\n"
        "> [!IMPORTANT]\n"
        "> Calibration remains BLOCKED.\n"
        "> Failures stay visible.\n"
        "\n"
        "```markdown\n"
        "> [!NOTE]\n"
        "> Inside code fence — must stay untouched.\n"
        "```\n"
        "\n"
        "> [!TIP]\n"
        "> Use `get_recipe()`.\n"
    )
    out = portal.convert_github_alerts(sample)
    assert "!!! important\n    Calibration remains BLOCKED.\n    Failures stay visible." in out
    assert "!!! tip\n    Use `get_recipe()`." in out
    assert "> [!NOTE]\n> Inside code fence — must stay untouched." in out


def test_transform_markdown_rewrites_staged_and_external_links():
    repo_to_staged = {
        "reports/forge/technique-inventory.md": "technique-inventory.md",
        "reports/forge/families/unified.md": "families/unified.md",
        "reports/forge/experiments/assets/grid100.svg": "experiments/assets/grid100.svg",
        "docs/four-laws.md": "laws/four-laws.md",
    }
    staged_paths = set(repo_to_staged.values())
    md = (
        '<a name="cohort-cuda-abc"></a>\n\n'
        "See [Unified](../families/unified.md#cohort-cuda-abc) and "
        "[`docs/four-laws.md`](../../../docs/four-laws.md#law-1) and "
        "[`four-laws.md#L60-L77`](../../../docs/four-laws.md#L60-L77).\n"
        "SVG: ![grid](assets/grid100.svg)\n"
        "Code: [`particlegan/recipes.py`](../../../particlegan/recipes.py#L67) and "
        "[receipt](../records/sample.json).\n"
        "Inline code `[keep](../../../particlegan/recipes.py)` stays literal.\n"
        "```\n"
        "[keep-fenced](../../../particlegan/recipes.py)\n"
        "```\n"
    )
    out = portal.transform_markdown(
        md,
        src_repo_rel="reports/forge/experiments/grid100.md",
        dst_docs_rel="experiments/grid100.md",
        repo_to_staged=repo_to_staged,
        staged_paths=staged_paths,
        root=ROOT,
        git_ref="develop",
    )
    assert '<a id="cohort-cuda-abc"></a>' in out
    assert "[Unified](../families/unified.md#cohort-cuda-abc)" in out
    assert "[`docs/four-laws.md`](../laws/four-laws.md#law-1)" in out
    assert "[`four-laws.md#L60-L77`](https://github.com/HyperGAN/ParticleGAN/blob/develop/docs/four-laws.md#L60-L77)" in out
    assert "![grid](assets/grid100.svg)" in out
    assert "[`particlegan/recipes.py`](https://github.com/HyperGAN/ParticleGAN/blob/develop/particlegan/recipes.py#L67)" in out
    assert "[receipt](https://github.com/HyperGAN/ParticleGAN/blob/develop/reports/forge/records/sample.json)" in out
    assert "`[keep](../../../particlegan/recipes.py)`" in out
    assert "[keep-fenced](../../../particlegan/recipes.py)" in out


def test_build_docs_site_skip_mkdocs_stages_portal_and_leaves_repo_clean(tmp_path):
    build_dir = tmp_path / "build"
    site_dir = tmp_path / "site"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_docs_site.py"),
            "--build-dir",
            str(build_dir),
            "--site-dir",
            str(site_dir),
            "--skip-mkdocs",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Staged ParticleGAN Research Portal" in result.stdout

    docs = build_dir / "docs"
    expected_files = [
        docs / "index.md",
        docs / "leaderboard.md",
        docs / "technique-inventory.md",
        docs / "laws/index.md",
        docs / "laws/four-laws.md",
        docs / "laws/audit/index.md",
        docs / "families/index.md",
        docs / "families/unified.md",
        docs / "experiments/index.md",
        docs / "studies/index.md",
        docs / "studies/problems.md",
        docs / "reference/index.md",
        docs / "reference/api.md",
        docs / "reference/glossary.md",
        docs / "reference/experimentation.md",
        docs / "assets/images/particlegan-hero.png",
        docs / "assets/stylesheets/portal.css",
        docs / "assets/javascripts/mathjax.js",
        docs / "assets/javascripts/mermaid-init.js",
        docs / "assets/javascripts/portal-leaderboard.js",
        docs / "assets/data/leaderboard.json",
        build_dir / "mkdocs.yml",
    ]
    for path in expected_files:
        assert path.is_file(), f"missing staged portal file: {path}"

    # Verify leaderboard.json structure
    payload = json.loads((docs / "assets/data/leaderboard.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "particlegan_forge_leaderboard_v1"
    assert len(payload["families"]) >= 10
    assert len(payload["tasks"]) > 100
    assert len(payload["energy_benchmarks"]) > 0
    assert len(payload["laws"]) >= 4
    assert payload["laws_meta"]["epistemic_status"] == "provisional_evolving"
    assert payload["caveats"]

    # Verify resolved mkdocs.yml preserves mermaid custom fence and has dynamic nav
    mkdocs_text = (build_dir / "mkdocs.yml").read_text(encoding="utf-8")
    assert "!!python/name:pymdownx.superfences.fence_code_format" in mkdocs_text
    assert "pymdownx.arithmatex:" in mkdocs_text
    assert '"Families":' in mkdocs_text
    assert '"Active Selected Families":' in mkdocs_text
    assert '"Historical & Ablation Families":' in mkdocs_text

    # Verify link rewriting on staged family page (no raw ../../../particlegan links left)
    unified_staged = (docs / "families/unified.md").read_text(encoding="utf-8")
    assert "../../../particlegan/" not in unified_staged
    assert "https://github.com/HyperGAN/ParticleGAN/blob/develop/particlegan/recipes.py" in unified_staged
    assert "<a name=" not in unified_staged

    # Verify repo bloat guard stays clean
    bloat = check_repo_bloat.check_repo_bloat(ROOT)
    assert bloat["passed"], bloat["new_violations"]


def test_stdlib_leaderboard_fallback_matches_hf_publish():
    stdlib_payload = portal.build_portal_leaderboard_payload(ROOT, prefer_stdlib=True)
    full_payload = portal.build_portal_leaderboard_payload(ROOT, prefer_stdlib=False)
    assert stdlib_payload["schema_version"] == 1
    assert stdlib_payload["summary"]["selected_families"] == full_payload["summary"]["selected_families"]
    assert stdlib_payload["summary"]["counted_view_requirements"] == full_payload["summary"]["counted_view_requirements"]
    assert stdlib_payload["summary"]["distinct_required_tests"] == full_payload["summary"]["distinct_required_tests"]
    assert stdlib_payload["summary"]["full_suite_tests"] == full_payload["summary"]["full_suite_tests"]
    by_id_std = {f["id"]: f for f in stdlib_payload["families"]}
    by_id_full = {f["id"]: f for f in full_payload["families"]}
    assert set(by_id_std) == set(by_id_full)
    for fid, f_full in by_id_full.items():
        f_std = by_id_std[fid]
        assert f_std["full_suite"]["pass"] == f_full["full_suite"]["pass"], fid
        assert f_std["full_suite"]["fail"] == f_full["full_suite"]["fail"], fid
        assert f_std["total_view_requirements"]["pass"] == f_full["total_view_requirements"]["pass"], fid
    assert len(stdlib_payload["energy_benchmarks"]) > 0
    assert any(t.get("gif_relpath") for t in stdlib_payload["tasks"])
    assert len(stdlib_payload["laws"]) >= 4


def test_portal_javascript_syntax_and_helpers():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    js_files = [
        ROOT / "docs/portal/assets/javascripts/mathjax.js",
        ROOT / "docs/portal/assets/javascripts/mermaid-init.js",
        ROOT / "docs/portal/assets/javascripts/portal-leaderboard.js",
    ]
    for path in js_files:
        subprocess.run([node, "--check", str(path)], check=True)

    helper_check = (
        f"const m = require({json.dumps(str(js_files[2]))});"
        "const f = m.computeParetoFrontier(["
        "  {idx:0, x:20, y:80}, {idx:1, x:10, y:70}, {idx:2, x:15, y:60}, {idx:3, x:12, y:90}"
        "]);"
        "if (f.map(p => p.idx).join(',') !== '1,3') throw new Error('bad pareto: ' + JSON.stringify(f));"
        "const u = m.hfGifUrl('HyperGAN/particlegan-forge-evidence', 'main', 'media/forge/a.gif');"
        "if (!u.endsWith('/resolve/main/media/forge/a.gif')) throw new Error('bad gif url: ' + u);"
    )
    subprocess.run([node, "-e", helper_check], check=True)


def test_mkdocs_strict_build_when_installed(tmp_path):
    mkdocs_bin = os.environ.get("MKDOCS_BIN") or shutil.which("mkdocs")
    if not mkdocs_bin:
        pytest.skip("mkdocs is not installed in the current environment")
    build_dir = tmp_path / "mkdocs-build"
    site_dir = tmp_path / "mkdocs-site"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_docs_site.py"),
            "--build-dir",
            str(build_dir),
            "--site-dir",
            str(site_dir),
            "--strict",
            "--mkdocs-bin",
            mkdocs_bin,
        ],
        cwd=str(ROOT),
        check=True,
    )
    assert (site_dir / "index.html").is_file()
    assert (site_dir / "leaderboard/index.html").is_file()
    assert (site_dir / "laws/four-laws/index.html").is_file()
    assert (site_dir / "families/unified/index.html").is_file()
    assert (site_dir / "assets/data/leaderboard.json").is_file()
