# Leaderboard Refresh Runbook

One-command workflow for keeping **all four** ParticleGAN Forge leaderboard surfaces synchronized:

| Surface | Location | Source of truth |
| --- | --- | --- |
| **GitHub Markdown Leaderboard** | [`reports/forge/technique-inventory.md`](../reports/forge/technique-inventory.md) & [`reports/forge/families/`](../reports/forge/families/) | [`configs/forge/selections/family-current-v1.json`](../configs/forge/selections/family-current-v1.json) + [`reports/forge/technique-evidence/manifest.json`](../reports/forge/technique-evidence/manifest.json) |
| **Hugging Face Dataset** | [`HyperGAN/particlegan-forge-evidence`](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence) (`data/leaderboard.json`, Parquet tables, GIFs) | Exported by `python -m experiments.forge publish --inventory` |
| **Hugging Face Static Space** | [`HyperGAN/particlegan-leaderboard`](https://huggingface.co/spaces/HyperGAN/particlegan-leaderboard) | Static frontend in [`ops/hf_space/`](../ops/hf_space/) fetching `data/leaderboard.json` from the HF Dataset |
| **GitHub Pages Research Portal** | [`https://hypergan.github.io/ParticleGAN/`](https://hypergan.github.io/ParticleGAN/) | Auto-built on merge to `develop` by [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) via [`scripts/build_docs_site.py`](../scripts/build_docs_site.py) |

All four surfaces are orchestrated by [`ops/refresh_leaderboard.py`](../ops/refresh_leaderboard.py).

---

## Quick Start (1-Command Refresh)

### 1. Audit what is stale or missing (offline, ~2 s)

```bash
python ops/refresh_leaderboard.py status --check-pr
```

This checks:

- Git branch, `HEAD`, and whether the working tree is clean.
- Whether newer candidates (e.g. `unified-v15`, `axiom-v5`) exist beyond the current pins in [`configs/forge/selections/family-current-v1.json`](../configs/forge/selections/family-current-v1.json).
- Whether any 16-task Occam-complete candidate is missing an uncontended NVML energy benchmark study (needed for the **Cost vs. Pass-Rate Pareto Frontier**).
- Whether [`reports/forge/publications/inventory-latest.json`](../reports/forge/publications/inventory-latest.json) is behind `HEAD`.
- Whether staged actual-training GIFs are waiting to upload to Hugging Face.
- Whether the standing `develop -> master` PR is open and links to `technique-inventory.md`.

### 2. Local dry-run rebuild & validation (zero network uploads)

```bash
python ops/refresh_leaderboard.py sync
```

Runs, in order:

1. `python -m experiments.forge compile --summaries-only`
2. `python reports/forge/regenerate_technique_inventory.py --refresh-publication`
3. `python -m experiments.forge.experiment_docs`
4. `python scripts/build_docs_site.py --skip-mkdocs` (add `--check-mkdocs` to run `mkdocs build --strict` locally)
5. `python scripts/check_repo_bloat.py` and `python scripts/check_lint.py`
6. `python -m experiments.forge publish --inventory --dry-run`
7. `python ops/hf_space/deploy.py --dry-run`

Preview the exact command plan without executing anything via `--dry-run`.

### 3. Publish to Hugging Face (Dataset + Static Space) and verify live

```bash
python ops/refresh_leaderboard.py sync --publish
```

This runs all local generation/validation steps, then:

1. Uploads staged actual-training GIFs when `--upload-media`, `--collect-local-media`, or `--ingest-media` is passed.
2. Publishes `data/leaderboard.json`, `data/*.parquet`, and the Dataset Card to `HyperGAN/particlegan-forge-evidence`, updating [`reports/forge/publications/inventory-latest.json`](../reports/forge/publications/inventory-latest.json).
3. Deploys [`ops/hf_space/`](../ops/hf_space/) to `HyperGAN/particlegan-leaderboard`.
4. Runs live HTTP + payload verification across HF Dataset, HF Space, GitHub Pages, and the `develop -> master` PR.

### 4. Push, PR, merge to `develop`, and verify GitHub Pages

GitHub Pages automatically rebuilds and deploys whenever a PR is merged into `develop`:

```bash
git add -A
git commit -m "forge: refresh leaderboard reports and HF publication receipt"
git push -u origin HEAD
gh pr create --base develop --fill
# After CI passes & PR merges to develop:
python ops/refresh_leaderboard.py verify --watch-pages
```

> [!IMPORTANT]
> Always keep an open `develop -> master` PR whose description links to
> `https://github.com/HyperGAN/ParticleGAN/blob/develop/reports/forge/technique-inventory.md`.
> `verify` checks this automatically.

---

## Updating Specific Surfaces

### A. Updating the Energy & J-to-PASS Pareto Frontier

The **Cost vs. Pass-Rate Pareto Frontier** (on both the HF Space and GitHub Pages) pairs:

- **Y-axis**: 16-task `r4-occam-ablation-v1` Occam pass rate (`occam_pass / occam_executed`, excluding `INCOMPLETE`/`TIMEOUT` tasks from pass credit).
- **X-axis**: Uncontended NVML `grid100` J-to-PASS, `ring16` J-to-PASS, `dcgan_performance_r256` J/step, or `dcgan_performance_r256` ms/step.

Wait until all GPUs are idle (`0%` util), then run:

```bash
python ops/refresh_leaderboard.py energy \
  --ideas axiom-v5 unified-v15 \
  --output rtx-a6000-new-candidates-energy-benchmark.json
```

`energy` automatically:

1. Refuses to run if any GPU is busy (unless `--allow-busy` is passed, which tags `contended: true` so contended numbers never pollute clean Pareto cohorts).
2. Runs [`experiments/law4_optimizer_profiler.py`](../experiments/law4_optimizer_profiler.py) (`--mode energy-benchmark`) and saves raw logs under gitignored `runs/forge/`.
3. Attaches `family_id`, `source_commit`, `contended`, and the matching committed 16-task `occam_receipt_ids` (or `official_forge_readout`) into `reports/forge/studies/<output>.json`.

Then run `python ops/refresh_leaderboard.py sync --publish` and merge into `develop`.

---

### B. Promoting a New Candidate in `technique-inventory.md`

[`reports/forge/technique-inventory.md`](../reports/forge/technique-inventory.md) and [`reports/forge/families/`](../reports/forge/families/) display the exact whole-row cohorts pinned in [`configs/forge/selections/family-current-v1.json`](../configs/forge/selections/family-current-v1.json).

> [!NOTE]
> **Unrelated edits do not invalidate experiments:** Pure post-run reporting, docs, Hugging Face
> publishing, media export, and Colab helper scripts (`experiments/forge/NON_SCIENTIFIC_SOURCE_PATHS`)
> are excluded from the scientific `source_digest` (enforced by
> [`tests/test_forge_sources_isolation.py`](../tests/test_forge_sources_isolation.py)). If an older
> run was recorded before that split (or across a multi-GPU commit), pass `--source-snapshot`,
> `--study-commit`, and `--cuda-model` to `promote` to cryptographically verify every recorded file
> hash against the exact frozen snapshot without re-running GPUs.

1. **Run ordinary screening from a clean commit on `develop`:**

   ```bash
   git status --short  # must be empty
   CLEAN_SHA=$(git rev-parse HEAD)
   python -m experiments.forge run unified-v15 --study unified-v15-disc-stab-rtx-study --gpus 0,1
   python -m experiments.forge run axiom-v5    --study axiom-v5-disc-stab-rtx-study    --gpus 0,1
   ```

2. **Preview the promotion without modifying tracked files:**

   ```bash
   python ops/refresh_leaderboard.py promote \
     --source-commit "$CLEAN_SHA" \
     --family unified=unified-v15 \
     --family axiom=axiom-v5 \
     --dry-run
   ```

3. **Apply the promotion:**

   ```bash
   python ops/refresh_leaderboard.py promote \
     --source-commit "$CLEAN_SHA" \
     --family unified=unified-v15 \
     --family axiom=axiom-v5
   ```

   `promote` enforces all Forge provenance rules automatically:
   - Verifies hydrated receipts at `$CLEAN_SHA` match the exact Git blobs at `$CLEAN_SHA`.
   - Archives the prior `configs/forge/selections/family-current-v1.json` byte-for-byte into `configs/forge/selections/history/family-current-<sha256>.json`.
   - Writes verified `family_row_pin(...)` entries (`default_adoption: false`, `selection_kind: "current_measurement"`).
   - Registers the new candidate(s) in [`configs/forge/trainer-families.json`](../configs/forge/trainer-families.json) and [`reports/forge/technique-evidence/manifest.json`](../reports/forge/technique-evidence/manifest.json).
   - Regenerates `technique-inventory.md`, `technique-inventory.json`, and `reports/forge/families/*.md`.

4. **Update narrative docs & pinned tests (if needed):**
   - Review [`configs/forge/family-documentation/`](../configs/forge/family-documentation/) so descriptions and honest failure modes match the promoted candidate.
   - Run `pytest -q tests/test_forge_hf_publish.py tests/test_forge_trainer_families.py tests/test_forge_current_technique_inventory.py tests/test_docs_site_build.py tests/test_hf_space_deploy.py` and update any tests that explicitly pinned the prior candidate ID.
   - Finish with `python ops/refresh_leaderboard.py sync --publish` and merge into `develop`.

---

### C. Refreshing Actual-Training GIFs

GIFs are never committed to Git; only the compact manifest [`reports/forge/leaderboard-media-index.json`](../reports/forge/leaderboard-media-index.json) is tracked, while binary `.gif` files live on Hugging Face under `media/forge/leaderboard/`.

```bash
# Export GIFs from local selected-cohort attempts, update index, and upload to HF:
python ops/refresh_leaderboard.py sync --collect-local-media --publish

# Or ingest externally rendered GIF bundles (e.g., from Colab):
python ops/refresh_leaderboard.py sync --ingest-media /path/to/rendered_gifs --publish
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Receipts recorded at <SHA> do not match the clean Git tree` | The Forge study ran while tracked files were modified locally. | Commit all changes (`git status --short` empty), re-run the Forge study from that clean commit, and pass that exact SHA to `promote`. |
| `Refusing uncontended energy benchmark while GPU(s) are busy` | Another process is using a GPU (`util > 5%`). | Wait for GPUs to go idle (`nvidia-smi`), or pass `--allow-busy` only for explicitly tagged contended diagnostics. |
| Candidate appears in `energy.parquet` but is missing from the Pareto chart | Candidate does not yet have all 16 `r4-occam-ablation-v1` tasks in a single `(candidate_id, source_commit, source_digest, gpu)` cohort. | Run the complete `r4-occam-ablation-v1` view on that GPU (or link the official 16-task `occam_receipt_ids` in the study JSON). |
| GitHub Pages build fails in CI but worked locally | `.github/workflows/pages.yml` builds **without PyTorch** using `build_leaderboard_json_stdlib` in `scripts/build_docs_site.py`. | Test the PyTorch-free path locally: `python scripts/build_docs_site.py --stdlib-leaderboard --strict`. |
| `scripts/check_repo_bloat.py` fails | A file `> 500 KB` or a raw log/GIF entered `reports/`. | Keep raw logs, traces, checkpoints, and GIFs under gitignored `runs/forge/` or upload media to Hugging Face. |
