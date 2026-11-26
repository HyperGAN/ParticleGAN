"""Offline unit tests for the Hugging Face Static Space assets and deploy helper."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ops.hf_space import deploy

SPACE_DIR = Path(__file__).resolve().parents[1] / "ops" / "hf_space"


def test_committed_space_assets_validate_and_exclude_deploy_script():
    frontmatter = deploy.validate_space_dir(SPACE_DIR)
    assert frontmatter["sdk"] == "static"
    assert frontmatter["pinned"] is False
    assert frontmatter["license"] == "mit"
    assert frontmatter["title"]
    assert frontmatter["emoji"]
    assert frontmatter["short_description"]

    files = deploy.collect_upload_files(SPACE_DIR)
    paths = [item["path"] for item in files]
    assert set(deploy.REQUIRED_FILES).issubset(paths)
    assert "deploy.py" not in paths
    assert all("__pycache__" not in path for path in paths)


def test_index_and_app_reference_dataset_and_caveats():
    html = (SPACE_DIR / "index.html").read_text(encoding="utf-8")
    js = (SPACE_DIR / "app.js").read_text(encoding="utf-8")

    assert 'href="styles.css"' in html
    assert 'src="app.js"' in html
    assert "HyperGAN/particlegan-forge-evidence" in html
    assert "HyperGAN/ParticleGAN" in html
    assert "technique-inventory.md" in html
    assert "Scientific Scope" in html
    assert 'id="energy-section"' in html
    assert 'id="energy-table"' in html
    assert 'id="kpi-energy"' in html
    assert 'id="four-laws-l4-block"' in html
    assert 'id="four-laws-l4-toggle"' in html
    assert "four_laws_adherence_l4" in js
    assert "four_laws_adherence_v2_l4" in js
    for col_label in (
        "Candidate",
        "Suite Pass",
        "r256 ms/step",
        "r256 J/step",
        "Watts",
        "State B/param",
        "ring16 J-to-PASS",
        "grid100 J-to-PASS",
        "GPU Cohort",
        "Provenance",
    ):
        assert col_label in html
    assert "HyperGAN/particlegan-forge-evidence" in js
    assert "data/leaderboard.json" in js
    assert "energy_benchmarks" in js
    assert "J-to-PASS" in js


def test_app_js_passes_node_syntax_check_if_available():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    done = subprocess.run([node, "--check", str(SPACE_DIR / "app.js")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_app_js_normalizes_v1_leaderboard_schema_if_node_available():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    script = r"""
const assert = require("assert");
const app = require(process.argv[1]);
const sample = {
  schema_version: 1,
  schema: "particlegan_forge_leaderboard_v1",
  generated_at: "2026-11-22T06:06:12Z",
  git_commit: "87d72c39b9f79c2d9702897c2d7d693afb2a236d",
  dataset_repo: "HyperGAN/particlegan-forge-evidence",
  view: "discriminator_stability",
  view_revision: 9,
  declared_view_revisions: { adaptation: 4, discriminator_stability: 19 },
  full_suite_list: {
    id: "technique-inventory-full-suite-2026-11-18",
    dated: "2026-11-18",
    sha256: "e03d3154ff77a8bcdc684e7e88d446d702132cb14971035aa095e9d65bf0f2a0",
    count: 69
  },
  summary: {
    selected_families: 1,
    counted_view_requirements: 115,
    distinct_required_tests: 36,
    full_suite_tests: 69,
    stretch_assignments: 68,
    stretch_distinct_tests: 33,
    held_assignments: 48,
    held_distinct_tests: 13,
    removed_assignments: 4,
    removed_distinct_tests: 1,
    attempt_coverage: { attempted: 1109, total: 1109 }
  },
  caveats: ["Screening evidence only."],
  families: [{
    id: "unified",
    label: "Unified",
    selected_recipe: "Unified",
    candidate_id: "unified-v11",
    recipe_sha256: "5a885f9e0dc2",
    source_digest: "e132cf1f36af",
    cohort_anchor: "cohort-cuda-42a130a78fc3",
    family_page: "reports/forge/families/unified.md",
    stale: false,
    tiers: {
      "1": { pass: 27, total: 27, not_run: 0, blocked: 0, diagnostic: 0, incompatible: 0 },
      "2": { pass: 68, total: 78, not_run: 1, blocked: 0, diagnostic: 0, incompatible: 0 },
      "3": { pass: 0, total: 10, not_run: 10, blocked: 0, diagnostic: 0, incompatible: 0 },
      "4": { pass: 0, total: 0, not_run: 0, blocked: 0, diagnostic: 0, incompatible: 0 }
    },
    total_view_requirements: { pass: 95, total: 115, not_run: 11, blocked: 0, diagnostic: 0, incompatible: 0 },
    full_suite: { pass: 40, fail: 6, not_measured: 11, other: 12, total: 69 },
    stretch: { total: { pass: 50, total: 68, not_run: 4, incompatible: 12, diagnostic: 0 } }
  }],
  tasks: [{
    family_id: "unified",
    family_label: "Unified",
    task_id: "ae_gan_hold",
    tier: 1,
    role: "required",
    views: ["discriminator_stability (Tier 1)"],
    status: "PASS",
    recorded_status: "PASS",
    reason: "ok",
    recipe_hash: "5a885f9e0dc2",
    source_hash: "e132cf1f36af",
    wall_seconds: 12.5,
    energy: {
      joules_total: 4566.565,
      joules_per_step: 2.854,
      avg_power_watts: 91.8,
      first_pass_step: 217,
      sustained_pass_step: 284,
      joules_to_first_pass: 619.34,
      joules_to_sustained_pass: 810.56,
      energy_backend: "nvml_energy_counter"
    },
    gif_relpath: "media/forge/pure-bcap/media/5ea5bdbb2d71/ae_gan_hold.gif"
  }, {
    family_id: "unified",
    family_label: "Unified",
    task_id: "two_pole",
    tier: 1,
    role: "required",
    views: ["discriminator_stability (Tier 1)"],
    status: "PASS",
    recorded_status: "PASS",
    reason: "ok",
    energy: null
  }],
  energy_benchmarks: [{
    candidate_id: "law4-energy-v3",
    family_id: "unified",
    optimizer_family: "law4_energy_v3",
    hardware_gpu: "NVIDIA RTX A6000",
    contended: false,
    source_commit: "0da787744",
    source_file: "reports/forge/studies/law4-phase2-energy-benchmark.json",
    r256_ms_per_step: 45.58,
    r256_watts: 287.7,
    r256_joules_per_step: 13.12,
    state_bytes_per_param: 0.00000017,
    ring16_first_pass_step: 217,
    ring16_j_to_pass: 619.34,
    grid100_first_pass_step: 1500,
    grid100_sustained_pass_step: 2500,
    grid100_j_to_pass: 7643.07,
    grid100_j_to_sustained_pass: 12738.46,
    occam_pass: 14,
    occam_executed: 16,
    evidence_scope: "study_energy_benchmark"
  }],
  task_definitions: [{ task_id: "ae_gan_hold", tier: 1, role: "required", summary: "Hold AE-GAN" }]
};
app.normalizeDataset(sample, "fee560f60b75df2bb0c286ae2dfb7ec5770403a9");
assert.strictEqual(app.state.meta.hfRevision, "fee560f60b75df2bb0c286ae2dfb7ec5770403a9");
assert.strictEqual(app.state.meta.fullSuiteList, "technique-inventory-full-suite-2026-11-18");
assert.strictEqual(app.state.meta.backfillAttempted, 1109);
assert.strictEqual(app.state.families.length, 1);
assert.strictEqual(app.state.families[0].total_pass, 95);
assert.strictEqual(app.state.families[0].total_req, 115);
assert.strictEqual(app.state.families[0].stretch_pass, 50);
assert.strictEqual(app.state.families[0].stretch_total, 68);
assert.strictEqual(app.state.families[0].full_pass, 40);
assert.strictEqual(app.state.families[0].r256_joules_per_step, null);
assert.strictEqual(app.state.receipts.length, 2);
assert.strictEqual(app.state.receipts[0].energy.joules_to_first_pass, 619.34);
assert.strictEqual(app.state.receipts[0].energy.energy_backend, "nvml_energy_counter");
assert.strictEqual(app.state.receipts[1].energy, null);
assert.strictEqual(app.state.energyBenchmarks.length, 1);
assert.strictEqual(app.state.energyBenchmarks[0].candidate_id, "law4-energy-v3");
assert.strictEqual(app.state.energyBenchmarks[0].grid100_j_to_pass, 7643.07);
assert.strictEqual(app.state.energyBenchmarks[0].contended, false);
assert.strictEqual(app.formatJoules(7643.07), "7,643 J");
assert.strictEqual(app.formatJoules(null), "—");
assert.strictEqual(
  app.hfResolveUrl(app.state.receipts[0].gif_path),
  "https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/fee560f60b75df2bb0c286ae2dfb7ec5770403a9/media/forge/pure-bcap/media/5ea5bdbb2d71/ae_gan_hold.gif"
);
"""
    done = subprocess.run([node, "-e", script, str(SPACE_DIR / "app.js")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_dry_run_makes_zero_network_calls_and_prints_json(capsys):
    rc = deploy.main(["--dry-run", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["space_id"] == deploy.DEFAULT_SPACE_ID
    assert payload["file_count"] >= 4
    uploaded_paths = {item["path"] for item in payload["files"]}
    assert set(deploy.REQUIRED_FILES).issubset(uploaded_paths)
    assert "deploy.py" not in uploaded_paths


def test_deploy_space_invokes_hf_api_upload_folder():
    calls = []

    class FakeHfApi:
        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                oid="abc123def456",
                commit_url="https://huggingface.co/spaces/HyperGAN/particlegan-leaderboard/commit/abc123def456",
            )

    result = deploy.deploy_space(
        source_dir=SPACE_DIR,
        space_id="HyperGAN/particlegan-leaderboard",
        commit_message="Test deploy",
        dry_run=False,
        api=FakeHfApi(),
    )
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "HyperGAN/particlegan-leaderboard"
    assert calls[0]["repo_type"] == "space"
    assert calls[0]["commit_message"] == "Test deploy"
    assert "deploy.py" in calls[0]["ignore_patterns"]
    assert "style.css" in calls[0]["delete_patterns"]
    assert result["commit_oid"] == "abc123def456"
    assert result["space_url"] == "https://huggingface.co/spaces/HyperGAN/particlegan-leaderboard"


def test_validate_space_dir_rejects_missing_files_and_bad_frontmatter(tmp_path):
    with pytest.raises(ValueError, match="Missing required Space file"):
        deploy.validate_space_dir(tmp_path)

    for name in ("index.html", "app.js", "styles.css"):
        (tmp_path / name).write_text("ok\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# No frontmatter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML frontmatter"):
        deploy.validate_space_dir(tmp_path)

    (tmp_path / "README.md").write_text(
        "---\ntitle: Test\nemoji: x\nsdk: gradio\npinned: false\nlicense: mit\nshort_description: desc\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sdk='static'"):
        deploy.validate_space_dir(tmp_path)

    (tmp_path / "README.md").write_text(
        "---\ntitle: Test\nemoji: x\nsdk: static\npinned: false\nlicense: mit\n"
        f"short_description: {'a' * 61}\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds 60 chars"):
        deploy.validate_space_dir(tmp_path)

    (tmp_path / "README.md").write_text(
        "---\ntitle: Test\nemoji: x\ncolorTo: cyan\nsdk: static\npinned: false\nlicense: mit\nshort_description: ok\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="colorTo"):
        deploy.validate_space_dir(tmp_path)


def test_index_includes_upgraded_research_sections_and_dynamic_laws():
    html = (SPACE_DIR / "index.html").read_text(encoding="utf-8")
    js = (SPACE_DIR / "app.js").read_text(encoding="utf-8")

    # Research Portal link, Laws Matrix, Pareto Scatter, Law/Edge Filters, and Experiment Deep-Dive Modal
    assert 'id="link-pages-portal"' in html
    assert "https://hypergan.github.io/ParticleGAN/" in html
    assert 'id="laws-section"' in html
    assert 'id="laws-epistemic-banner"' in html
    assert 'id="pareto-card"' in html
    assert 'id="filter-law"' in html
    assert 'id="filter-edge-case"' in html
    assert 'id="gif-law-filter"' in html
    assert 'id="gif-quick-tasks"' in html
    assert 'id="gif-family-strip"' in html
    assert 'id="gif-lightbox"' in html
    assert 'id="btn-gif-sync"' in html
    assert 'id="btn-gif-top2"' in html
    assert 'id="btn-gif-contrast"' in html
    assert 'data-gif-layout="top4"' in html
    assert 'id="experiment-modal"' in html
    # Never hardcode "Four Laws" in UI components — laws are loaded dynamically from data.laws
    assert "Four Laws" not in html
    assert "Four Laws" not in js


def test_app_js_enriched_laws_pareto_filters_and_local_mode_if_node_available():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    script = r"""
const assert = require("assert");
const app = require(process.argv[1]);

// 1. Local mode and data URL resolution
assert.strictEqual(app.isLocalMode({ localMode: false, datasetRepo: "HyperGAN/particlegan-forge-evidence", dataPath: "data/leaderboard.json" }), false);
assert.strictEqual(app.isLocalMode({ localMode: true, datasetRepo: "HyperGAN/particlegan-forge-evidence", dataPath: "data/leaderboard.json" }), true);
assert.strictEqual(app.isLocalMode({ localMode: false, datasetRepo: "local", dataPath: "data/leaderboard.json" }), true);
assert.strictEqual(app.resolveDataUrl({ localMode: true, datasetRepo: "HyperGAN/particlegan-forge-evidence", dataPath: "data/leaderboard.json" }), "data/leaderboard.json");
assert.strictEqual(
  app.resolveDataUrl({ localMode: false, datasetRepo: "HyperGAN/particlegan-forge-evidence", requestedRev: "main", dataPath: "data/leaderboard.json" }),
  "https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/main/data/leaderboard.json"
);

// 2. Hash routing & SVG sanitizer
assert.strictEqual(app.parseExperimentHash("#experiment/ring16_acquisition"), "ring16_acquisition");
assert.strictEqual(app.parseExperimentHash("#laws-section"), null);
assert.strictEqual(app.sanitizeInlineSvg('<svg viewBox="0 0 10 10"><circle r="3"/></svg>').startsWith("<svg"), true);
assert.strictEqual(app.sanitizeInlineSvg('<svg><script>alert(1)</script></svg>'), null);

// 3. Enriched payload with dynamic 3-law registry, family mechanisms, task docs & multi-cohort Pareto rows
const enriched = {
  schema_version: 1,
  schema: "particlegan_forge_leaderboard_v1",
  pages_base_url: "https://hypergan.github.io/ParticleGAN",
  laws_meta: {
    schema_version: 1,
    epistemic_status: "provisional_evolving",
    epigraph: "Discover the laws of learning.",
    epistemic_note: "Living framework note.",
    doc_path: "docs/four-laws.md"
  },
  laws: [
    { id: "law1", short_label: "Law 1", title: "Representational Geometry", question: "Q1?", sub_laws: [] },
    {
      id: "law3",
      short_label: "Law 3",
      title: "Transport Identifiability",
      question: "Q3?",
      sub_laws: [{ id: "law3a", short_label: "Law 3A", title: "Ray Screening" }]
    }
  ],
  families: [
    {
      id: "unified",
      label: "Unified",
      candidate_id: "unified-v11",
      overview: "Unified formulation",
      law_mechanisms: {
        law1: { status: "addressed", mechanism: "Prototype prior", how_it_works: "Adaptive cells", known_limits: "Fixed M" }
      },
      equations: [{ label: "Gate", latex: "r(y) = 1", explanation: "Demo" }],
      mermaid_diagram: 'flowchart LR\n  A["P"] --> B["G"]',
      quickstart: { preset: "unified_v11", snippet: ["from particlegan import get_recipe"] },
      tags: ["constant-learning-rate"],
      full_suite: { pass: 40, fail: 6, not_measured: 11, other: 14, total: 71 },
      total_view_requirements: { pass: 95, total: 115 }
    }
  ],
  task_definitions: [
    {
      task_id: "ring16_acquisition",
      title: "Ring-16 Mode Acquisition",
      tier: 1,
      role: "required",
      laws: ["law1", "law3a"],
      edge_cases: ["disconnected-support", "16-mode-ring"],
      intuition: "Acquire 16 modes.",
      thresholds: { min_modes: 16 },
      svg_inline: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>'
    },
    {
      task_id: "two_pole",
      title: "Two Pole",
      tier: 1,
      role: "required",
      laws: ["law1"],
      edge_cases: ["bimodal"]
    }
  ],
  tasks: [
    { family_id: "unified", task_id: "ring16_acquisition", tier: 1, role: "required", status: "PASS" },
    { family_id: "unified", task_id: "two_pole", tier: 1, role: "required", status: "PASS" }
  ],
  energy_benchmarks: [
    {
      candidate_id: "cand-fast",
      family_id: "unified",
      hardware_gpu: "NVIDIA RTX A6000",
      contended: false,
      r256_ms_per_step: 40.0,
      r256_joules_per_step: 10.0,
      grid100_j_to_pass: 4500.0,
      occam_pass: 15,
      occam_executed: 16
    },
    {
      candidate_id: "cand-dominated",
      family_id: "unified",
      hardware_gpu: "NVIDIA RTX A6000",
      contended: false,
      r256_ms_per_step: 60.0,
      r256_joules_per_step: 15.0,
      grid100_j_to_pass: 7000.0,
      occam_pass: 14,
      occam_executed: 16
    },
    {
      candidate_id: "cand-best",
      family_id: "unified",
      hardware_gpu: "NVIDIA RTX A6000",
      contended: false,
      r256_ms_per_step: 48.0,
      r256_joules_per_step: 11.4,
      grid100_j_to_pass: 5000.0,
      occam_pass: 16,
      occam_executed: 16
    },
    {
      candidate_id: "cand-contended",
      family_id: "unified",
      hardware_gpu: "NVIDIA RTX A6000",
      contended: true,
      r256_ms_per_step: 20.0,
      r256_joules_per_step: 5.0,
      grid100_j_to_pass: 2000.0,
      occam_pass: 16,
      occam_executed: 16
    }
  ]
};

app.normalizeDataset(enriched, "abc123");
assert.strictEqual(app.state.laws.length, 2);
assert.strictEqual(app.state.lawsMeta.epistemic_note, "Living framework note.");
assert.strictEqual(app.state.families[0].law_mechanisms.law1.status, "addressed");
assert.strictEqual(app.state.families[0].quickstart.preset, "unified_v11");

// Parent law "law3" matches task tagged with sub-law "law3a"
const ringDef = app.state.taskDefinitions.get("ring16_acquisition");
assert.strictEqual(app.taskMatchesLaw(ringDef, "law3"), true);
assert.strictEqual(app.taskMatchesLaw(ringDef, "law3a"), true);
assert.strictEqual(app.taskMatchesLaw(app.state.taskDefinitions.get("two_pole"), "law3"), false);

// Explorer law and edge-case filters
app.state.expFilters.law = "law3";
assert.strictEqual(app.getFilteredReceipts().length, 1);
assert.strictEqual(app.getFilteredReceipts()[0].task_id, "ring16_acquisition");
app.state.expFilters.law = "";
app.state.expFilters.edgeCase = "bimodal";
assert.strictEqual(app.getFilteredReceipts().length, 1);
assert.strictEqual(app.getFilteredReceipts()[0].task_id, "two_pole");
app.state.expFilters.edgeCase = "";

// Pareto points separate cohorts & exclude contended points from Pareto frontier
const pareto = app.buildParetoPoints(app.state.energyBenchmarks, app.state.families, "grid100_j_to_pass", "occam");
assert.strictEqual(pareto.points.length, 4);
assert.strictEqual(pareto.cohorts.length, 2);
const frontierIds = pareto.points.filter((p) => p.isFrontier).map((p) => p.candidate_id).sort();
assert.deepStrictEqual(frontierIds, ["cand-best", "cand-fast"]);
const contendedPt = pareto.points.find((p) => p.candidate_id === "cand-contended");
assert.strictEqual(contendedPt.isFrontier, false);
assert.strictEqual(contendedPt.shape, "diamond");

// Switch to full_suite Y-axis and r256_ms_per_step X-axis
const paretoFull = app.buildParetoPoints(app.state.energyBenchmarks, app.state.families, "r256_ms_per_step", "full_suite");
assert.strictEqual(paretoFull.points.length, 4);
assert.ok(paretoFull.points[0].y > 50 && paretoFull.points[0].y < 60);

// Deduplicate study_energy_benchmark over receipt row for same (candidate_id, cohort) and flag latest family version
const dedupRows = [
  {
    key: "u15-rec",
    candidate_id: "unified-v15",
    family_id: "unified",
    hardware_gpu: "NVIDIA RTX A6000",
    contended: false,
    evidence_scope: "published_summary",
    grid100_j_to_pass: 5400,
    occam_pass: 16,
    occam_executed: 16
  },
  {
    key: "u15-study",
    candidate_id: "unified-v15",
    family_id: "unified",
    hardware_gpu: "NVIDIA RTX A6000",
    contended: false,
    evidence_scope: "study_energy_benchmark",
    grid100_j_to_pass: 5200,
    r256_joules_per_step: 12.1,
    occam_pass: 16,
    occam_executed: 16
  },
  {
    key: "u12-study",
    candidate_id: "unified-v12",
    family_id: "unified",
    hardware_gpu: "NVIDIA RTX A6000",
    contended: false,
    evidence_scope: "study_energy_benchmark",
    grid100_j_to_pass: 4512,
    r256_joules_per_step: 11.4,
    occam_pass: 16,
    occam_executed: 16
  }
];
const paretoDedup = app.buildParetoPoints(dedupRows, app.state.families, "grid100_j_to_pass", "occam");
assert.strictEqual(paretoDedup.points.length, 2);
const u15Pt = paretoDedup.points.find((p) => p.candidate_id === "unified-v15");
assert.strictEqual(u15Pt.x, 5200);
assert.strictEqual(u15Pt.isLatestFamily, true);
assert.strictEqual(paretoDedup.points.find((p) => p.candidate_id === "unified-v12").isLatestFamily, false);

// Best-performing family ranking & smart GIF defaults
const sampleFams = [
  { id: "atlas", label: "Atlas", full_pass: 7, total_pass: 17, tier1_pass: 17, full_fail: 8, stale: false },
  { id: "axiom", label: "Axiom", full_pass: 39, total_pass: 95, tier1_pass: 27, full_fail: 6, stale: false },
  { id: "unified", label: "Unified", full_pass: 40, total_pass: 95, tier1_pass: 27, full_fail: 6, stale: false },
  { id: "rhbi", label: "RHBI", full_pass: 33, total_pass: 83, tier1_pass: 27, full_fail: 7, stale: false },
];
const ranked = app.rankFamiliesByPerformance(sampleFams);
assert.deepStrictEqual(ranked.map((f) => f.id), ["unified", "axiom", "rhbi", "atlas"]);

const gifReceipts = [
  { family_id: "atlas", task_id: "ring16_acquisition", status: "FAIL", gif_path: "media/forge/a.gif" },
  { family_id: "unified", task_id: "ring16_acquisition", status: "PASS", gif_path: "media/forge/u.gif" },
  { family_id: "axiom", task_id: "ring16_acquisition", status: "PASS", gif_path: "media/forge/ax.gif" },
  { family_id: "rhbi", task_id: "ring16_acquisition", status: "FAIL", gif_path: "media/forge/r.gif" },
  { family_id: "atlas", task_id: "two_pole", status: "FAIL", gif_path: "media/forge/a2.gif" },
];
assert.deepStrictEqual(app.pickDefaultGifFamilies("ring16_acquisition", ranked, gifReceipts, 2), ["unified", "axiom"]);
assert.deepStrictEqual(app.pickDefaultGifFamilies("ring16_acquisition", ranked, gifReceipts, 4), ["unified", "axiom", "rhbi", "atlas"]);
assert.deepStrictEqual(app.pickContrastGifPair("ring16_acquisition", ranked, gifReceipts), ["unified", "rhbi"]);
assert.strictEqual(
  app.pickDefaultGifTask(
    [{ taskId: "two_pole", gifCount: 1 }, { taskId: "ring16_acquisition", gifCount: 4 }],
    ranked,
    gifReceipts
  ),
  "ring16_acquisition"
);
"""
    done = subprocess.run([node, "-e", script, str(SPACE_DIR / "app.js")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
