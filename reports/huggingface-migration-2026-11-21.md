# Hugging Face Data Migration & Public Publishing Plan (2026-11-21)

- **Date:** 2026-11-21
- **Baseline branch / commit:** `develop` @ `8a412dcaf` (Phase 1 branched off `87d72c39b`)
- **Status:** Phase 1 in progress — live Hugging Face repos created (`HyperGAN/particlegan-forge-evidence` Dataset, `HyperGAN/particlegan-leaderboard` Static Space)

## 1. Problem Statement & Baseline Inventory

Two problems share the same root cause:

1. **Stray research data in Git:** `HEAD` tracks **2.01 GB across 25,510 files**, of which **1.96 GB (97.5%, 21,425 files)** sits under `reports/`. The library, Forge framework, tests, configs, and docs together are only ~50 MB. `.git` packfiles have grown to **2.10 GB**, and `.github/workflows/tests.yml` currently runs `fetch-depth: 0` across all 8 CPU test shards.
2. **Unpublished results outside Git:** **348 GB** of raw execution envelopes, checkpoints, and artifacts (`runs/` = 196 GB, `artifacts/` = 152 GB) exist only on local disk. Archive cards referenced by `experiments/forge/artifact_resolver.py` resolve as `MISSING` on any other machine, and external readers have no searchable/interactive way to inspect ParticleGAN's benchmark results outside raw GitHub Markdown.

### Measured Tracked Footprint (`HEAD` @ `8a412dcaf`)

| Category | Tracked Size | File Count | Primary Locations |
| --- | ---: | ---: | --- |
| `.json` | 756.95 MB | 14,478 | `reports/forge/records/` (254 MB), `bcap-prior-search/` (97 MB), `technique-evidence/` (77 MB), `technique-inventory.json` (21.6 MB) |
| `.gif` | 631.86 MB | 2,195 | `reports/forge/` (408 MB), `reports/toy_audit/` (158 MB), `reports/toy100/` (47 MB) |
| `.npz` | 264.40 MB | 1,459 | `reports/toy100/` (228 MB), `reports/gym/` (28 MB) |
| `.html` | 100.65 MB | 37 | `reports/transition/demo/index.html` (8.7 MB), interactive report dumps |
| `.png` / `.mp4` | 83.30 MB | 426 | `reports/toy100/`, `reports/forge/`, `reports/cifar-ddgan/` |
| `.gz` / `.pt` / `.zip` | 78.53 MB | 740 | `reports/toy100/` (42 MB `.gz`, 30 MB `.pt`) |
| Code, configs, tests, docs, `.md` | ~92 MB | ~6,175 | `configs/` (12.6 MB), `docs/` (11.6 MB), `tests/` (7.2 MB), `experiments/` (3.7 MB), `particlegan/` (3.0 MB), `.md` reports |
| **Total tracked in `HEAD`** | **2,007.89 MB** | **25,510** | — |

### Non-Negotiable Provenance Constraints (from `AGENTS.md` & `docs/forge-artifact-resolver.md`)

- **Do not rewrite Git history.** Historical receipts and tests rely on exact pinned Git commits and blob hashes (and `tests.yml` replays published ancestor bytes). Evicting files from `HEAD` cleans future checkouts and stops new bloat while preserving historical commit/blob identities.
- **No silent network hydration.** `experiments/forge/artifact_resolver.py` must remain offline-by-default; Hugging Face retrieval must be explicitly opted into (via flag or environment variable) and verified against committed SHA-256 digests and declared byte counts before extraction.
- **Keep `reports/forge/technique-inventory.md` live in Git.** The `develop -> master` PR links directly to `https://github.com/HyperGAN/ParticleGAN/blob/develop/reports/forge/technique-inventory.md`.
- **Never substitute summaries for original evidence.** Parquet tables and Space dashboards are derived publication views; byte-exact verification always points back to content-addressed `sha256/<digest>.tar.gz` archives and pinned commits.

---

## 2. Account-Level Setup Checklist (Hugging Face + GitHub)

Perform these one-time steps before Phase 1 implementation begins:

### A. Hugging Face Organization & Billing / Storage
1. **Create or claim the Hugging Face Organization** matching GitHub: **`HyperGAN`** (or confirm the exact org handle you want public URLs to use — e.g., `https://huggingface.co/HyperGAN`).
   - Using an org instead of a personal account keeps URLs stable (`HyperGAN/particlegan-...`) and decouples publishing permissions from a single personal profile.
2. **Verify storage tier & request an open-research grant if needed:**
   - Free HF accounts/orgs have storage limits on large datasets (typically ~100 GB per dataset soft tier / ~300 GB org-wide depending on current HF plan rules).
   - For the initial migration (~2–20 GB of curated reports, GIFs, Parquet tables, and published study archives), a free org is sufficient.
   - If you plan to archive >100 GB of full checkpoints/envelopes from `runs/` + `artifacts/`, either request a free Hugging Face **Open Source / Academic Storage Grant** for the org or upgrade the org plan before uploading bulk checkpoints.
3. **Choose license metadata** on the repos: align with `LICENSE` (MIT) or CC-BY-4.0 for benchmark datasets/media.

### B. Repositories to Create Under the Org
Create these three repos (all **Public** unless you want a separate private staging dataset first):

| Repo Type | Suggested Name | Purpose |
| --- | --- | --- |
| **Dataset** | `HyperGAN/particlegan-forge-evidence` | Primary data store: Parquet leaderboard/receipts tables, training GIFs/media, content-addressed `archives/sha256/*.tar.gz`, and migrated legacy tarballs |
| **Space** | `HyperGAN/particlegan-leaderboard` | Public interactive leaderboard & training-GIF viewer (Static HTML/DuckDB-Wasm or Gradio) backed by the Dataset repo |
| **Model** *(optional, later)* | `HyperGAN/particlegan-<task>-<recipe>` | Only when publishing curated pretrained checkpoints (e.g., CIFAR DDGAN, Lunar Lander) with standalone model cards |

### C. Access Tokens (Least-Privilege Fine-Grained Tokens)
In **Hugging Face → Settings → Access Tokens**, create **two separate fine-grained tokens** (avoid classic account-wide write tokens):

1. **Local developer publishing token (`particlegan-forge-publish-local`)**
   - Scope: **Write** access *only* to `HyperGAN/particlegan-forge-evidence` (and `HyperGAN/particlegan-leaderboard` if pushing Space updates manually).
   - Store locally via `hf auth login` (or `HF_TOKEN` in your local shell environment — never commit).
2. **CI publishing token (`particlegan-forge-ci`)** *(optional, for automated Space/Parquet refresh on `develop`/`master` or release tags)*
   - Scope: **Write** access *only* to `HyperGAN/particlegan-forge-evidence` and `HyperGAN/particlegan-leaderboard`.
   - Add to GitHub repository **Settings → Secrets and variables → Actions** as `HF_TOKEN`.

### D. Local Environment Verification
Install `huggingface_hub` (with Xet transfer support enabled by default in modern versions) and verify authentication:

```bash
python -m pip install -U "huggingface_hub>=0.30" "pyarrow>=17"
hf auth login
hf auth whoami
```

---

## 3. Target Architecture & Ownership Split

### GitHub (`HyperGAN/ParticleGAN`) — Target `< 100 MB` in `HEAD`
- Source code (`particlegan/`, `particlegan_forge/`, `experiments/`, `benchmarks/`, `examples/`, `tests/`, `scripts/`, `ops/`)
- Declarations (`configs/forge/`)
- Docs, narrative study writeups (`.md`), `reports/forge/technique-inventory.md`, and `reports/forge/problems.md`
- Compact **archive & publication manifest cards** recording:
  - archive `sha256`, byte size, and per-member SHA-256s
  - original Git commit/blob provenance (for migrated tracked files)
  - pinned HF locator: `hf_repo`, `hf_revision` (40-char commit SHA), and `hf_path`
- Minimal synthetic fixtures in `tests/fixtures/` and README hero image

### Hugging Face Dataset (`HyperGAN/particlegan-forge-evidence`)
Structured to avoid huge loose-file counts while keeping media directly linkable:

```text
README.md                                    # HF Dataset Card (YAML metadata + dataset description)
data/
  leaderboard.parquet                        # Selected recipes, tier totals, full-suite & stretch metrics
  receipts.parquet                           # Consolidated Forge technique-receipts & readouts
  attempts.parquet                           # Backfill / attempt coverage records
media/
  forge/<family>/<task>/<artifact>.gif       # Current training GIFs referenced by Markdown & Space
  toy_audit/<test>/<artifact>.gif            # Public-API toy test GIFs
  docs/<name>.gif                            # Docs/walkthrough animations
archives/
  sha256/<64-char-sha256>.tar.gz             # Exact content-addressed archives for artifact_resolver.py
legacy/
  toy100-artifacts-8a412dcaf.tar.gz          # One-time archive of reports/toy100 (.npz/.pt/.gz/.gif)
  toy-audit-media-8a412dcaf.tar.gz           # One-time archive of reports/toy_audit (.gif)
  reports-demos-8a412dcaf.tar.gz             # One-time archive of reports/{trajectory,transition,gym}
  forge-records-8a412dcaf.tar.gz             # One-time archive of heavy historical Forge JSONs
```

- **Markdown links** always pin an immutable HF revision SHA, never `main`:
  `https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/<hf_rev>/media/...gif`
- **Dataset Viewer** automatically renders `data/*.parquet` with full SQL search and filtering on the HF dataset page.

### Hugging Face Space (`HyperGAN/particlegan-leaderboard`)
- Reads `data/*.parquet` and `media/` from `HyperGAN/particlegan-forge-evidence` (pinned or latest tagged release).
- Provides:
  - Interactive filter by family (`Unified`, `RHBI`, `EMPF`, `BISPF`, `BCAP`, `Atlas`, `E22`, `GAN v3`, `K3P`, `KA2`, `R1/R2`), tier (1–4), and status (`required`, `stretch`, `held`, `diagnostic`)
  - Side-by-side training GIF inspection per task/recipe
  - Direct provenance links back to the Git commit, study readout `.md`, and HF `sha256` archive

---

## 4. Forge Tooling & Code Changes

1. **Optional dependency extra in `pyproject.toml`:**
   - Add `hf = ["huggingface_hub>=0.30,<3", "pyarrow>=17"]` so core library installs stay lean (`torch` + `numpy` only).
2. **Opt-in HF backend in `experiments/forge/artifact_resolver.py`:**
   - Support an explicit `--hf-repo <org/repo>` + `--hf-rev <sha>` (or `PARTICLEGAN_FORGE_HF_REPO` / manifest `hf` locator when `--allow-hf` is passed).
   - Never fetch over the network unless explicitly requested.
   - Download `archives/sha256/<sha256>.tar.gz` into the local content-addressed cache (`runs/forge/artifacts/sha256/<sha256>.tar.gz`), then validate through the existing `_verified_archive` SHA-256, byte-count, path-safety, and member-hash checks.
   - Update `docs/forge-artifact-resolver.md` and `tests/test_forge_artifact_resolver.py` (using a mocked/local hub cache fixture—no live network in unit tests).
3. **New `python -m experiments.forge publish` subcommand:**
   - `--study <id>` or `--inventory`:
     - Packages specified envelopes/artifacts into `sha256/<sha256>.tar.gz` (validating all member hashes).
     - Exports `data/leaderboard.parquet`, `data/receipts.parquet`, and `data/attempts.parquet`.
     - Uploads changed files in a single atomic HF commit (`HfApi.create_commit` / `upload_large_folder` for large drops).
     - Writes a small publication receipt in Git (`reports/forge/publications/<id>.json`) recording the returned immutable `hf_revision` SHA, uploaded file digests, and Git source commit.
4. **CI Bloat Guard (`scripts/check_repo_bloat.py` wired into `.github/workflows/tests.yml`):**
   - Rejects *newly added* tracked files under `reports/` matching `.gif`, `.mp4`, `.npz`, `.pt`, `.pth`, `.ckpt`, `.gz`, `.zip`, `.html`, or any file `> 500 KB` (with an explicit allowlist file for pre-existing tracked paths during the phased migration so Phase 1 never breaks existing branches).

---

## 5. Phased Execution Plan

### Phase 0 — Account Setup (Owner Action)
- Complete Section 2 (create `HyperGAN` HF org, create Dataset + Space repos, mint fine-grained tokens, configure `hf auth login` and optional GitHub `HF_TOKEN` secret).

### Phase 1 — Stop the Bleeding & Publish Live Results (Zero Breakage)
- Add `hf` optional extra to `pyproject.toml`.
- Implement `forge publish` (Parquet export + archive/media upload + publication receipt).
- Implement opt-in HF hydration in `experiments/forge/artifact_resolver.py` + unit tests.
- Run initial publish of:
  - Current Parquet tables (`leaderboard`, `receipts`, `attempts`)
  - Existing valid archive tarballs from local storage (resolving current `MISSING` archive gaps)
  - Featured training GIFs for `technique-inventory.md`
- Deploy minimal `HyperGAN/particlegan-leaderboard` Space reading the Dataset.
- Enable the CI bloat guard in allowlist mode so **zero new binaries or mega-JSONs** enter Git from this point forward.

### Phase 2 — Evict Stray Tracked Binaries from `HEAD` (~1.2 GB+ Reduction)
Migrate in small, reviewable PRs while preserving original Git commit/blob provenance per `AGENTS.md`:

1. **`reports/toy100/` (497 MB):** Bundle `.npz`/`.pt`/`.gz`/`.gif`/`.mp4` into `legacy/toy100-artifacts-<commit>.tar.gz` + upload linked media to `media/toy100/`; keep `.md`/`.py`/small config JSONs + pointer manifest in Git; repair Markdown links.
2. **`reports/toy_audit/` (175 MB):** Upload 345 `.gif` files to `media/toy_audit/` + archive bundle; update public-API toy test contract links to pinned HF URLs.
3. **`reports/trajectory/`, `reports/transition/`, `reports/gym/`, `reports/cifar-ddgan/`, `reports/denoising-toy/` (~218 MB):** Upload `.html`, `.npz`, `.gif`, `.png`, `.pt` to HF; keep `.md` writeups and pointer manifests in Git.
4. **`reports/forge/` media (408 MB across 1,803 `.gif` files):** Upload to `media/forge/` and update Forge's Markdown renderer (`tier1_media.py` / `family_reports.py` / `technique_inventory.py`) to emit pinned HF URLs (with a local-path fallback flag for offline rendering).

### Phase 3 — Decouple Heavy Forge JSONs & Optimize CI (~700 MB Reduction)
- Audit which tests (`test_refresh_forge_readouts.py`, `forge compile`, `forge board`) read `reports/forge/records/` (254 MB), `bcap-prior-search/` (97 MB), `technique-evidence/` (77 MB), and `technique-inventory.json` (21.6 MB).
- Keep compact normalized receipts in Git (or hydrate full historical readouts on demand from HF) so `HEAD` drops below ~100 MB.
- Scope `fetch-depth: 0` in `.github/workflows/tests.yml` to only the dedicated ancestor-replay test job, letting the other 7 shards use fast shallow clones (`fetch-depth: 1`).

### Phase 4 — Curated Triage of Local `runs/` & `artifacts/` (348 GB)
- Inventory local `runs/` (196 GB) and `artifacts/` (152 GB) against committed study/archive cards:
  - **Published / cited studies:** verify SHA-256, package as `archives/sha256/<sha256>.tar.gz`, upload to HF, verify round-trip hydration with `python -m experiments.forge artifacts hydrate --allow-hf`.
  - **Superseded scratch / worktree / debug runs:** verify not referenced by any committed card or active study, then prune locally.

---

## 6. Success Criteria

- `HEAD` tracked size drops from **2.01 GB (25,510 files)** to **< 150 MB** (Phase 2) and **< 100 MB** (Phase 3).
- CI blocks any new stray `.gif`/`.npz`/`.pt`/`.html` or `> 500 KB` JSON committed under `reports/`.
- `python -m experiments.forge artifacts inspect/hydrate` can verify and restore published archives on a fresh machine using pinned HF revisions with byte-exact SHA-256 checks.
- External users can browse ParticleGAN's leaderboard, filter all 1,100+ recipe/test outcomes in the HF Dataset Viewer, and inspect training GIFs in `HyperGAN/particlegan-leaderboard`.
