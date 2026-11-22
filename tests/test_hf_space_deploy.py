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
    assert "HyperGAN/particlegan-forge-evidence" in js
    assert "data/leaderboard.json" in js


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
    gif_relpath: "media/forge/pure-bcap/media/5ea5bdbb2d71/ae_gan_hold.gif"
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
assert.strictEqual(app.state.receipts.length, 1);
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
