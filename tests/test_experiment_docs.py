"""Tests for Forge experiment docs, target-geometry SVGs, AST extraction, and Laws Hub."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import pytest

from experiments.forge.experiment_docs import (
    EXPERIMENTS_DIR_REL,
    LAWS_HUB_REL,
    TASK_DOCS_DIR_REL,
    _check_mermaid_balance,
    build_experiment_catalog,
    extract_python_annotations,
    load_laws_registry,
    write_experiment_docs,
)
from experiments.forge.experiment_svg import BADGE_TEXT, render_task_svg
from experiments.forge.views import task_evaluation_fingerprint, task_execution_fingerprint
from scripts.check_repo_bloat import check_repo_bloat

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "configs/forge/tasks"
EXPERIMENTS_DIR = ROOT / EXPERIMENTS_DIR_REL
ASSETS_DIR = EXPERIMENTS_DIR / "assets"
LAWS_HUB = ROOT / LAWS_HUB_REL

LINK_RE = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)\)")
RAW_LATEX_CMD_RE = re.compile(r"\\(?:mathbb|nabla|operatorname|frac|lVert|rVert|lvert|rvert|sum|prod|int)\b")


def _all_base_task_paths() -> list[Path]:
    return sorted(TASKS_DIR.glob("*.json"))


def _md_anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    seen: dict[str, int] = {}
    anchors: set[str] = set()
    for heading in re.findall(r"^#{1,6} (.*)$", text, flags=re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    for html_id in re.findall(r'<a\s+(?:name|id)="([^"]+)"', text):
        anchors.add(html_id)
    return anchors


def test_all_base_tasks_preserve_fingerprints_and_bytes_vs_develop():
    """Every base task keeps identical execution/evaluation fingerprints (and raw bytes) vs develop."""
    base_ref = None
    for candidate_ref in ("develop", "origin/develop"):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{candidate_ref}^{{commit}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            base_ref = candidate_ref
            break
    if base_ref is None:
        pytest.skip("develop ref unavailable in detached/shallow CI checkout")
    task_paths = _all_base_task_paths()
    assert len(task_paths) == 163
    for path in task_paths:
        rel = path.relative_to(ROOT).as_posix()
        current_bytes = path.read_bytes()
        current = json.loads(current_bytes.decode("utf-8"))
        dev_bytes = subprocess.check_output(["git", "show", f"{base_ref}:{rel}"], cwd=ROOT)
        dev_task = json.loads(dev_bytes.decode("utf-8"))
        assert task_execution_fingerprint(current) == task_execution_fingerprint(dev_task), rel
        assert task_evaluation_fingerprint(current) == task_evaluation_fingerprint(dev_task), rel
        assert hashlib.sha256(current_bytes).hexdigest() == hashlib.sha256(dev_bytes).hexdigest(), rel


def test_all_base_pages_svgs_and_sidecars_exist_and_are_valid():
    task_ids = [json.loads(p.read_text(encoding="utf-8"))["id"] for p in _all_base_task_paths()]
    assert len(task_ids) == 163
    assert (EXPERIMENTS_DIR / "README.md").is_file()
    assert LAWS_HUB.is_file()

    sizes: list[int] = []
    for tid in task_ids:
        md_path = EXPERIMENTS_DIR / f"{tid}.md"
        svg_path = ASSETS_DIR / f"{tid}.svg"
        sidecar_path = ROOT / TASK_DOCS_DIR_REL / f"{tid}.json"
        assert md_path.is_file(), f"missing page for {tid}"
        assert svg_path.is_file(), f"missing svg for {tid}"
        assert sidecar_path.is_file(), f"missing task-documentation sidecar for {tid}"

        svg_text = svg_path.read_text(encoding="utf-8")
        sizes.append(len(svg_text.encode("utf-8")))
        root_el = ET.fromstring(svg_text)
        assert root_el.tag.endswith("svg")
        assert BADGE_TEXT in "".join(root_el.itertext()), f"{tid}.svg missing schematic badge"

        md_text = md_path.read_text(encoding="utf-8")
        assert f"assets/{tid}.svg" in md_text
        assert "## Visual Schematic" in md_text
        assert "## Protocol & Architecture Diagram" in md_text
        assert "## Task Contract & Numerical Gates" in md_text
        assert "## Recorded Family Results" in md_text
        assert "## Annotation Provenance" in md_text
        assert not _check_mermaid_balance(md_text), f"mermaid error in {tid}.md"

    # Keep SVGs crisp and lightweight (median <= 6.5 KB, max <= 12 KB)
    sizes.sort()
    assert sizes[len(sizes) // 2] <= 6500, f"median SVG too large: {sizes[len(sizes) // 2]}"
    assert max(sizes) <= 12000, f"max SVG too large: {max(sizes)}"


def test_svg_generation_is_deterministic():
    task = json.loads((TASKS_DIR / "fl_law1_aniso_unequal_modes.json").read_text(encoding="utf-8"))
    doc = json.loads((ROOT / TASK_DOCS_DIR_REL / "fl_law1_aniso_unequal_modes.json").read_text(encoding="utf-8"))
    first = render_task_svg(task, doc)
    second = render_task_svg(task, doc)
    assert first == second
    assert BADGE_TEXT in "".join(ET.fromstring(first).itertext())


def test_generated_markdown_has_safe_mermaid_math_and_resolving_links():
    md_files = sorted(EXPERIMENTS_DIR.glob("*.md")) + [LAWS_HUB]
    assert len(md_files) == 165
    anchor_cache: dict[Path, set[str]] = {}
    broken_links: list[str] = []
    math_issues: list[str] = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        # Check Mermaid balance & no raw brackets inside quoted node labels
        assert not _check_mermaid_balance(text), md_path.name
        for block in re.findall(r"```mermaid\n(.*?)```", text, flags=re.S):
            for quoted in re.findall(r'"([^"]*)"', block):
                assert "[" not in quoted and "]" not in quoted, f"raw brackets in Mermaid label in {md_path.name}: {quoted}"

        # Strip fenced code blocks and inline code spans before checking math & links
        stripped = re.sub(r"```.*?```", "", text, flags=re.S)
        no_code = re.sub(r"`[^`\n]+`", "", stripped)
        if r"\(" in no_code or r"\[" in no_code:
            math_issues.append(f"{md_path.name}: unsupported \\( or \\[ delimiter")
        if RAW_LATEX_CMD_RE.search(no_code):
            # Only flag if outside $...$
            without_math = re.sub(r"\$\$[\s\S]*?\$\$|\$[^$\n]+\$", "", no_code)
            if RAW_LATEX_CMD_RE.search(without_math):
                math_issues.append(f"{md_path.name}: raw LaTeX command outside math")
        for line in no_code.splitlines():
            if line.strip().startswith("|") and "$" in line:
                for span in re.findall(r"\$([^$\n]+)\$", line):
                    if "|" in span:
                        math_issues.append(f"{md_path.name}: pipe inside table math")

        # Verify every relative link and anchor resolves
        for target in LINK_RE.findall(stripped):
            if re.match(r"^[a-z]+:", target):
                continue
            file_part, _, anchor = target.partition("#")
            dest = (md_path.parent / file_part).resolve() if file_part else md_path.resolve()
            if not dest.exists():
                broken_links.append(f"{md_path.name} -> {target}")
                continue
            if anchor and dest.suffix == ".md":
                if dest not in anchor_cache:
                    anchor_cache[dest] = _md_anchors(dest)
                if anchor not in anchor_cache[dest]:
                    broken_links.append(f"{md_path.name} -> {target} (missing anchor #{anchor})")

    assert not math_issues, math_issues[:10]
    assert not broken_links, broken_links[:10]


def test_ast_extractor_reads_experiment_doc_and_docstring_without_executing(tmp_path: Path):
    fixture = tmp_path / "sample_host.py"
    fixture.write_text(
        '"""Synthetic host fixture for read-only AST extraction."""\n'
        "SCALE_VAL = 3.0\n"
        'EXPERIMENT_DOC = {"title": "Synthetic Probe", "laws": ["law1"], "edge_cases": ["demo"], '
        '"intuition": "Tests AST extraction.", "failure_mode": "Executes module."}\n'
        'SPECS = {"demo_task": {"bound": SCALE_VAL, "steps": 100}}\n'
        'THRESHOLDS = {"demo_task": [["hq", ">=", 0.9]]}\n'
        'raise RuntimeError("AST extractor must never execute Python files")\n',
        encoding="utf-8",
    )
    info = extract_python_annotations(fixture)
    assert info["docstring"] == "Synthetic host fixture for read-only AST extraction."
    assert info["experiment_doc"]["title"] == "Synthetic Probe"
    assert info["specs"]["demo_task"]["bound"] == 3.0
    assert info["thresholds"]["demo_task"] == [["hq", ">=", 0.9]]
    assert set(info["extracted_symbols"]) >= {"__doc__", "EXPERIMENT_DOC", "SPECS", "THRESHOLDS"}

    # Also verify real SHA-256-pinned repo file parses read-only cleanly
    real_info = extract_python_annotations(ROOT / "benchmarks/toy_audit/api_four_laws_123.py")
    assert "SPECS" in real_info["extracted_symbols"]
    assert "THRESHOLDS" in real_info["extracted_symbols"]
    assert "fl_law1_aniso_unequal_modes" in real_info["specs"]


def test_dynamic_laws_registry_and_inline_task_documentation(tmp_path: Path):
    """Laws Hub and catalog dynamically read configs/forge/laws.json and inline task['documentation']."""
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs/four-laws.md").write_text("# Laws\n\n### Law X — Evolving Principle\n", encoding="utf-8")
    (tmp_path / "configs/forge/tasks").mkdir(parents=True)
    (tmp_path / "configs/forge/family-documentation").mkdir(parents=True)
    (tmp_path / "reports/forge/families").mkdir(parents=True)
    (tmp_path / "reports/forge/families/demo.md").write_text("# Demo\n", encoding="utf-8")

    laws_payload = {
        "schema_version": 1,
        "epistemic_status": "provisional_evolving",
        "epigraph": "Test evolving epigraph.",
        "epistemic_note": "Laws may be refined or replaced.",
        "doc_path": "docs/four-laws.md",
        "laws": [
            {
                "id": "law_x",
                "short_label": "Law X",
                "title": "Evolving Principle",
                "domain_level": "Evolving Level",
                "question": "Does the evolving registry work without hardcoding?",
                "doc_anchor": "law-x--evolving-principle",
                "status": "provisional",
                "sub_laws": [{"id": "law_x1", "short_label": "Law X.1", "title": "Sub-Principle", "doc_anchor": "law-x--evolving-principle"}],
            }
        ],
    }
    (tmp_path / "configs/forge/laws.json").write_text(json.dumps(laws_payload), encoding="utf-8")
    (tmp_path / "configs/forge/family-documentation/demo.json").write_text(
        json.dumps({
            "family": "demo",
            "title": "Demo Family",
            "law_mechanisms": {
                "law_x": {"status": "addressed", "mechanism": "Adaptive Chart", "known_limits": "None yet"}
            },
        }),
        encoding="utf-8",
    )
    inline_task = {
        "schema_version": 1,
        "id": "inline_demo_task",
        "adapter": "transfer_vector",
        "execution": {"steps": 100, "prior": {"kind": "mog", "sigma": 0.05, "standardize": False}},
        "evaluation": {"kind": "transfer_sustained", "thresholds": [["hq", ">=", 0.9]]},
        "resources": {"gpus": 0},
        "requires_capabilities": [],
        "dependencies": [],
        "documentation": {
            "title": "Inline Documented Future Task",
            "laws": ["law_x1"],
            "edge_cases": ["future-inline"],
            "intuition": "Verifies inline task documentation support for new unpinned tasks.",
            "failure_mode": "Fails when inline docs are ignored.",
            "protocol_notes": "Inline fixture.",
            "diagram_kind": "gaussian_mixture_2d",
            "custom_mermaid": None,
        },
    }
    (tmp_path / "configs/forge/tasks/inline_demo_task.json").write_text(json.dumps(inline_task), encoding="utf-8")

    loaded = load_laws_registry(tmp_path)
    assert loaded["laws"][0]["id"] == "law_x"

    summary = write_experiment_docs(tmp_path)
    assert summary["experiment_count"] == 1
    assert summary["svg_count"] == 1

    hub_text = (tmp_path / LAWS_HUB_REL).read_text(encoding="utf-8")
    assert "Law X — Evolving Principle" in hub_text
    assert "Law X.1: Sub-Principle" in hub_text
    assert "Adaptive Chart" in hub_text
    assert "inline_demo_task" in hub_text
    assert "Test evolving epigraph." in hub_text


def test_dcgan_r256_renders_scaling_metrics_and_repo_bloat_passes():
    catalog = build_experiment_catalog(ROOT)
    assert catalog["experiment_count"] == 163
    assert catalog["full_suite_count"] == 59
    assert catalog["unimplemented_backlog_count"] == 12
    r256 = (EXPERIMENTS_DIR / "dcgan_performance_r256.md").read_text(encoding="utf-8")
    assert "Median ms/step" in r256
    assert "Peak Allocated MiB" in r256
    hub = LAWS_HUB.read_text(encoding="utf-8")
    assert "## Four Laws Adherence Diagnostic Leaderboard (NVIDIA L4 Cohort)" in hub
    assert "`axiom-v4`" in hub
    fl_page = (EXPERIMENTS_DIR / "fl_law41_trust_rare1pct_tanh.md").read_text(encoding="utf-8")
    assert "### Four Laws Adherence Diagnostic Receipts (Separate Cohort)" in fl_page
    assert "NVIDIA L4" in fl_page
    assert "NVIDIA RTX A6000" in fl_page
    bloat = check_repo_bloat(ROOT)
    assert bloat["passed"], bloat["new_violations"]
