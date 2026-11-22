# Local Storage Triage & Historical Archive Migration Report (2026-11-21)

- **Workspace audited:** `/nvme/dev/hypergan/ParticleGAN` (`develop` @ `87d72c39b9f79c2d9702897c2d7d693afb2a236d`)
- **Action taken:** Read-only audit + atomic HF upload of verified historical archives. **Zero tracked Git files edited; zero files deleted in `runs/` or `artifacts/`.**
- **Target HF dataset:** [`HyperGAN/particlegan-forge-evidence`](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence)
- **Pinned HF revision for archived tarballs:** `3cac9836b867e1a1f49050e68d64255c6a0f7e6d` ([commit](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/commit/3cac9836b867e1a1f49050e68d64255c6a0f7e6d))

---

## Executive Summary

| Scope | Allocated (`du -sh`) | Apparent Bytes (`du -sb`) | Key Finding |
|---|---:|---:|---|
| `runs/` | **196 GB** | 208,593,904,597 B (194.27 GiB) | 81% of `runs/forge` non-worktree data is raw `*.pt` checkpoints (97.91 GiB); 45 GB is 15 local Git worktrees; 47 GB is `queues/`; 14 GB is `tmp/`. |
| `artifacts/` | **152 GB** | 161,358,916,951 B (150.28 GiB) | **99.93% (`152 GB`)** is 7 Git worktrees in `artifacts/worktrees/` (4 of which hold 132 GB of nested `runs/forge/dcgan-performance-cloud-*`). The 7 non-worktree experiment dirs total only **113 MB**. |
| **Combined local repo** | **348 GB** | **369,952,821,548 B (344.55 GiB)** | **~217 GB** recoverable immediately (Tier 1: merged worktrees, `tmp/`, `frozen-checkouts/`, unreferenced/duplicate scaling clouds); **+76 GB** in Tier 2 (committed scaling-cloud `state.pt` checkpoints). Total safe recovery: **~293 GB (84%)**. |
| Host mirrors discovered | **102.7 GB** | — | `/mnt/ml7tb/ParticleGAN-forge` (76 GB), `/mnt/ml7tb/experiments/ParticleGAN` (668 MB), `/home/martyn/dev/particlegan/.worktree-cleanup/20261018T045452Z` (26 GB on `/home`, currently **97% full / 17 GB free**). |
| **Verified archives uploaded to HF** | **2.49 GiB** | **2,672,339,174 B** | **22/22** byte-exact SHA-256 + `tar.gz` verified historical Forge archives uploaded atomically to `archives/sha256/<sha256>.tar.gz` at revision `3cac9836b867e1a1f49050e68d64255c6a0f7e6d` and remote-verified byte-for-byte. |

---

## 1. Committed Archive Card Inventory & Local Verification

### 1.1 Strict `artifact_resolver.archive_spec` Cards (16 Unique Cards)

Running `experiments.forge.artifact_resolver.archive_spec` + `inspect_archive` across all 13,384 JSON files in `reports/` yields **16 schema-compliant archive cards**:

- **`AVAILABLE` (byte-exact verified via `python -m experiments.forge artifacts inspect`):** **1 card** (`126,880,703` bytes = 121.0 MiB, 3,269 declared members verified)
  - `reports/forge/tier1-completion/artifact-inventory.json` -> `/mnt/ml7tb/experiments/ParticleGAN/tier1-completion-v1/artifacts.tar.gz` (`sha256: 2041cfe2e9975b533089d4c74b0970164235c0d0b35af46a3ebbd7b7168b9c66`)
- **`MISSING` locally:** **15 cards** (`705,909,346` bytes = 673.2 MiB declared)
  - 10 point to `/ml2/hypergan/...` (volume `/ml2` is not mounted on this host)
  - 4 point to `runs/forge/...` (previously deleted locally)
  - 1 points to `/home/martyn/dev/ParticleGAN/artifacts/forge/tier1-existing-configs-v1.tar.gz` (directory no longer present)
- **`INVALID`:** **0**

### 1.2 Full Cross-Schema Audit (Custom Archive Cards + Provenance Manifests + Local Mirrors)

Expanding the audit beyond `archive_spec`'s 3 schemas to all committed `reports/forge/` archive cards, provenance manifests, and readouts — and scanning local mirrors (`/mnt/ml7tb/ParticleGAN-forge/`, `/mnt/ml7tb/experiments/ParticleGAN/`, and `/home/martyn/dev/particlegan/.worktree-cleanup/20261018T045452Z/`) — recovered **22 unique byte-exact verified tarballs** whose full 64-character SHA-256 digests are committed in `reports/forge/`:

- **Total verified & uploaded to HF:** **22 tarballs** (`2,672,339,174` bytes = **2.489 GiB**, 40,278 total tar members)
  - 1 strict `archive_spec` card (`tier1-completion/artifact-inventory.json`)
  - 9 custom archive/provenance cards (`bcap-tier2-search/archive.json`, `dualnorm-pacing-v2/artifact-inventory.json`, `five-word-tier-split/archive.json`, `fivewords-dynamics/archive.json`, `gaussian-smoke-inventory/archive-v4.json`, `k3p-two-pole-horizon-v1/archive.json`, `records/archive-pr320-network-spectral-cap-c4627fb14619.json`, `tier1-prior-duration/artifact-provenance.json`, `tier1-prior-smoke/artifact-provenance.json`)
  - 12 readout/provenance/admission manifests (`gaussian-smoke-inventory/provenance-interruption-v3.json`, `word-split-inventory/readout.json`, `bcap-tier2/readout.json`, `bcap-six/readout.json`, `gaussian-smoke-inventory/v2-interruption/provenance.json`, `tier1-batch-size/provenance.json`, `gaussian-shallow/provenance.json`, `bcap-convolution/readout.json`, `records/gaussian-no-fourier-v1-external-readout.json`, `gaussian-smoke-tier-split/provenance.json`, `gaussian1d-diagnosis/provenance.json`, `gaussian-smoke-inventory/admission-v1.json`)
- **Non-tarball directory archive verified on `/mnt/ml7tb`:**
  - `reports/forge/bcap-develop-integration/artifact-archive.json` -> unpacked directory `/mnt/ml7tb/ParticleGAN-forge/bcap-develop-integration-next`; `certificates/manifest.json` verified byte-exact (`sha256: f8789425fdf31965f1e83dad0f4c3cbc0dc83f847a46763ec756098090a5001a`).
- **Unreferenced local mirror tarballs (NOT uploaded; zero SHA-256 matches in `reports/`):** **10 tarballs** (`327,160,354` bytes = 312.0 MiB), including `/mnt/ml7tb/experiments/ParticleGAN/bcap-tier1-repair-v1/artifacts.tar.gz` (100.5 MB, `ad67afc34e86...`) and 9 superseded/draft tarballs in `.worktree-cleanup/20261018T045452Z/saved-files/`.

---

## 2. Verified Archives Uploaded to Hugging Face (`HyperGAN/particlegan-forge-evidence`)

All 22 archives were verified for both exact 64-char SHA-256 match against `reports/forge/` and clean `tarfile` gzip integrity, uploaded in one atomic commit, and verified remotely via `HfApi.list_repo_tree(..., revision=hf_revision)` for exact byte counts.

- **Repo:** `HyperGAN/particlegan-forge-evidence` (dataset, public)
- **Immutable `hf_revision`:** `3cac9836b867e1a1f49050e68d64255c6a0f7e6d`
- **Commit URL:** `https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/commit/3cac9836b867e1a1f49050e68d64255c6a0f7e6d`
- **Total uploaded:** `22` tarballs, `2,672,339,174` bytes (`2.489 GiB`), `0` skipped, `22/22` remote byte-exact verified.

| # | Primary Committed Card / Manifest | SHA-256 | Bytes | Members | HF Path (`archives/sha256/...`) | Local Source Mirror |
|---:|---|---|---:|---:|---|---|
| 1 | `reports/forge/bcap-tier2-search/archive.json` | `c6bd92e0300bba7665f96c28df00bd015e364f1b3c71957506def336e94dccc6` | 1,609,755,839 | 17,672 | `archives/sha256/c6bd92e0300bba7665f96c28df00bd015e364f1b3c71957506def336e94dccc6.tar.gz` | `/mnt/ml7tb/ParticleGAN-forge/archives/bcap-search-96-v1.tar.gz` |
| 2 | `reports/forge/gaussian-smoke-inventory/archive-v4.json` | `1531a5de3ee93e1bb216060eb25967010a2381c76ff0bc873267580851ba5c21` | 289,084,194 | 3,982 | `archives/sha256/1531a5de3ee93e1bb216060eb25967010a2381c76ff0bc873267580851ba5c21.tar.gz` | `.worktree-cleanup/.../gaussian-smoke-inventory-v4-final.tar.gz` |
| 3 | `reports/forge/dualnorm-pacing-v2/artifact-inventory.json` | `136c907e4a46cdcfffc7b2a904d2f1b684d06cdd44eabbba537ec64a0a680741` | 231,319,616 | 4,406 | `archives/sha256/136c907e4a46cdcfffc7b2a904d2f1b684d06cdd44eabbba537ec64a0a680741.tar.gz` | `.worktree-cleanup/.../bcap-dualnorm-pacing-v2.tar.gz` |
| 4 | `reports/forge/tier1-completion/artifact-inventory.json` (`archive_spec`) | `2041cfe2e9975b533089d4c74b0970164235c0d0b35af46a3ebbd7b7168b9c66` | 126,880,703 | 3,269 | `archives/sha256/2041cfe2e9975b533089d4c74b0970164235c0d0b35af46a3ebbd7b7168b9c66.tar.gz` | `/mnt/ml7tb/experiments/ParticleGAN/tier1-completion-v1/artifacts.tar.gz` |
| 5 | `reports/forge/gaussian-smoke-inventory/provenance-interruption-v3.json` | `dd0aaeca0f50fe1590d7d6b7a101d677610ca2966cd6544f2a697e059f1b907a` | 94,909,147 | 2,246 | `archives/sha256/dd0aaeca0f50fe1590d7d6b7a101d677610ca2966cd6544f2a697e059f1b907a.tar.gz` | `.worktree-cleanup/.../gaussian-smoke-inventory-v3-provenance-interruption.tar.gz` |
| 6 | `reports/forge/word-split-inventory/readout.json` | `a2519b3f68d2cbf8bf96e929242c21876db301df3df48d51cfd3b4e1957410fc` | 76,400,055 | 1,842 | `archives/sha256/a2519b3f68d2cbf8bf96e929242c21876db301df3df48d51cfd3b4e1957410fc.tar.gz` | `.worktree-cleanup/.../word-split-inventory-v1.tar.gz` |
| 7 | `reports/forge/bcap-tier2/readout.json` | `3e494a4525c4aa2d9b05085237c5f7e4d0069b8ff4e9146f60182abe33d53990` | 65,064,799 | 1,805 | `archives/sha256/3e494a4525c4aa2d9b05085237c5f7e4d0069b8ff4e9146f60182abe33d53990.tar.gz` | `.worktree-cleanup/.../bcap-tier2-smoothing-v1.tar.gz` |
| 8 | `reports/forge/bcap-six/readout.json` | `44de286fe6adfe3bde97f27a927fe224328391025083c3d15759c2fc89a16ab8` | 37,033,845 | 1,587 | `archives/sha256/44de286fe6adfe3bde97f27a927fe224328391025083c3d15759c2fc89a16ab8.tar.gz` | `.worktree-cleanup/.../bcap-six-smoothing-v1.tar.gz` |
| 9 | `reports/forge/records/archive-pr320-network-spectral-cap-c4627fb14619.json` | `f8c15e2eda12c906c5996c7a9bdb1a7c30d281cc5c2257f93bcf9dd71886cdc9` | 33,779,339 | 180 | `archives/sha256/f8c15e2eda12c906c5996c7a9bdb1a7c30d281cc5c2257f93bcf9dd71886cdc9.tar.gz` | `.worktree-cleanup/.../gaussian-network-magnitude-v1.tar.gz` |
| 10 | `reports/forge/gaussian-smoke-inventory/v2-interruption/provenance.json` | `220b4310532c73a483827310201d11852089430d34079d03f93d33d7dfb5d488` | 21,781,946 | 1,528 | `archives/sha256/220b4310532c73a483827310201d11852089430d34079d03f93d33d7dfb5d488.tar.gz` | `.worktree-cleanup/.../gaussian-smoke-inventory-v2-interruption.tar.gz` |
| 11 | `reports/forge/tier1-batch-size/provenance.json` | `32580b2591ec6750b2e3c62b9ad8e446f8525072d4a0f46b9918b49453808588` | 19,072,998 | 49 | `archives/sha256/32580b2591ec6750b2e3c62b9ad8e446f8525072d4a0f46b9918b49453808588.tar.gz` | `.worktree-cleanup/.../tier1-batch-size-v1.tar.gz` |
| 12 | `reports/forge/tier1-prior-smoke/artifact-provenance.json` | `0ab6a34f0dbb27b7bd6647ef44918a1f022abc2732826c0d50190064d96ccd7d` | 13,716,517 | 158 | `archives/sha256/0ab6a34f0dbb27b7bd6647ef44918a1f022abc2732826c0d50190064d96ccd7d.tar.gz` | `.worktree-cleanup/.../tier1-prior-smoke-v1.tar.gz` |
| 13 | `reports/forge/k3p-two-pole-horizon-v1/archive.json` | `68ca93892345822ff2b4c3c549afbe3831549cc753aee243dfaa16ea154b5bd0` | 12,411,184 | 1,271 | `archives/sha256/68ca93892345822ff2b4c3c549afbe3831549cc753aee243dfaa16ea154b5bd0.tar.gz` | `.worktree-cleanup/.../k3p-two-pole-horizon-v1-v1.tar.gz` |
| 14 | `reports/forge/fivewords-dynamics/archive.json` | `f72449c02c76952657983df5c926cf2f3fe85efc9a4680e1d05ae8e4560ce6a2` | 11,437,506 | 115 | `archives/sha256/f72449c02c76952657983df5c926cf2f3fe85efc9a4680e1d05ae8e4560ce6a2.tar.gz` | `.worktree-cleanup/.../fivewords-dynamics-v1.tar.gz` |
| 15 | `reports/forge/five-word-tier-split/archive.json` | `e2944bc031f96e9bbdd09fe8c8be24dff95a9d92fd8dd07c45a5ad9b220e0d0b` | 9,025,125 | 38 | `archives/sha256/e2944bc031f96e9bbdd09fe8c8be24dff95a9d92fd8dd07c45a5ad9b220e0d0b.tar.gz` | `.worktree-cleanup/.../five-word-tier-split-v1.tar.gz` |
| 16 | `reports/forge/tier1-prior-duration/artifact-provenance.json` | `18e339a2e78b73d95798b871914ac7a9301057e8b49a89c610f34dd7992d2480` | 8,012,588 | 29 | `archives/sha256/18e339a2e78b73d95798b871914ac7a9301057e8b49a89c610f34dd7992d2480.tar.gz` | `.worktree-cleanup/.../tier1-prior-duration-v1.tar.gz` |
| 17 | `reports/forge/gaussian-shallow/provenance.json` | `319cbec56850ed05711a983d336ee204834c933ec34d6168fd1e5098e8e36837` | 4,531,512 | 51 | `archives/sha256/319cbec56850ed05711a983d336ee204834c933ec34d6168fd1e5098e8e36837.tar.gz` | `.worktree-cleanup/.../gaussian-shallow-v1.tar.gz` |
| 18 | `reports/forge/bcap-convolution/readout.json` | `5d39d93b81b3101e02a8be2c97c8951905e9943ed9445188918efee542f82806` | 4,282,895 | 1,277 | `archives/sha256/5d39d93b81b3101e02a8be2c97c8951905e9943ed9445188918efee542f82806.tar.gz` | `.worktree-cleanup/.../bcap-convolution-images-v1.tar.gz` |
| 19 | `reports/forge/records/gaussian-no-fourier-v1-external-readout.json` | `8c5515862f5afd51fb09fcf8bee963b013014aaaa04a80846a26974f9c44e812` | 3,575,254 | 24 | `archives/sha256/8c5515862f5afd51fb09fcf8bee963b013014aaaa04a80846a26974f9c44e812.tar.gz` | `.worktree-cleanup/.../gaussian-no-fourier-v1.tar.gz` |
| 20 | `reports/forge/gaussian-smoke-tier-split/provenance.json` | `29f4886e32aa150bd88a688a3c5cf859b1e127818b39acb8a459f189c197ff96` | 224,325 | 10 | `archives/sha256/29f4886e32aa150bd88a688a3c5cf859b1e127818b39acb8a459f189c197ff96.tar.gz` | `.worktree-cleanup/.../gaussian-smoke-tier-split-v1.tar.gz` |
| 21 | `reports/forge/gaussian1d-diagnosis/provenance.json` | `4dc3e21b0643ae59f869c8569af52bfcb7e1ad83f53e0130d3c6b22765688c61` | 25,982 | 10 | `archives/sha256/4dc3e21b0643ae59f869c8569af52bfcb7e1ad83f53e0130d3c6b22765688c61.tar.gz` | `.worktree-cleanup/.../gaussian1d-diagnosis-v1.tar.gz` |
| 22 | `reports/forge/gaussian-smoke-inventory/admission-v1.json` | `6642dc64b49ae07706ec5a44cc0555722a4dcb751bf5976546358d3cba363f6b` | 13,805 | 4 | `archives/sha256/6642dc64b49ae07706ec5a44cc0555722a4dcb751bf5976546358d3cba363f6b.tar.gz` | `.worktree-cleanup/.../gaussian-smoke-inventory-v1-admission.tar.gz` |

*(Full local path prefix for `.worktree-cleanup/...` is `/home/martyn/dev/particlegan/.worktree-cleanup/20261018T045452Z/saved-files/<worktree>/files/artifacts/...`.)*

### 2.1 Declared Archive Cards Currently `MISSING` on This Host (29 Unique SHAs, 1.869 GiB)

These 29 declared archive tarballs (`2,007,199,015` bytes) were not found in `/nvme/dev/hypergan/ParticleGAN`, `/mnt/ml7tb`, or `/home/martyn/dev/particlegan`:

- **Unmounted `/ml2/` host volume (12 archives, 606.4 MB):**
  - `continuous-baseline-20261003/archive.json` (`e160acbcc6b9`, 159.3 MB)
  - `generator-step-20261003/archive.json` (`271cf1c21a34`, 111.3 MB)
  - `critic-balance-20261003/archive.json` (`292061adad59`, 64.4 MB)
  - `family-winner-round1/{phase1,policy-atlas-initial-launch,policy-capacity,policy-round1..4}` (7 cards, 315.8 MB)
  - `g-coherent-mean-{paired-research,seed0}/artifact-archive.json` (2 cards, 55.7 MB)
- **Historical `/home/martyn/dev/ParticleGAN/...` worktree paths (10 archives, 959.6 MB):**
  - `dualnorm-tier1/artifact-inventory.json` (`e4ee9b7aab8b`, 539.7 MB)
  - `pure-bcap/{archive,archive-initial}.json` (`9f2aa3f21b74` 100.1 MB, `3266966bc504` 44.3 MB)
  - `word-root-cause/archive.json` (`f67ffff63dc7`, 59.4 MB)
  - `tier1-refresh/archive.json` (`1ea74b7b92cd`, 43.0 MB)
  - `k3p-global-tier1-{v3,v2}/archive.json` (`567d5b09ed53` 33.3 MB, `218cb718e61f` 25.2 MB)
  - `family-wide-word-repairs/archive.json` (`c9943e536341`, 12.2 MB)
  - `ring16-truncation/archive.json` (`a7f7a8eb26d4`, 7.1 MB)
- **Relative `runs/forge/...` & `artifacts/...` paths (8 archives, 441.2 MB):**
  - `gaussian-smoke-inventory/archive-v6.json` (`87e445f1b07e`, 288.8 MB)
  - `gaussian-smoke-inventory/interrupted-v5/archive.json` (`9897f0d1f5c8`, 105.4 MB)
  - `ring16-failure/archive-reproduction.json` (`c6c452ef8dc7`, 29.9 MB)
  - `technique-inventory-archive.json` (`0574554f0b98`, 5.5 MB)
  - `r1r2-modern-toy-archive.json` (`96e1cb72b9ea`, 3.8 MB)
  - `release07-task-adaptation-archive.json` (`d2d228c91b2d`, 1.5 MB)
  - `r3gan-baseline-archive.json` (`a0446d48bfb7`, 1.4 MB)
  - `ring16-followup/archive.json` (`30f7ab06e201`, 0.2 MB)

If `/ml2` is ever remounted or another host backup is located, those tarballs can be uploaded to the same content-addressed `archives/sha256/<sha256>.tar.gz` path without modifying any existing cards.

---

## 3. Full Directory Triage: `runs/` (196 GB) and `artifacts/` (152 GB)

### 3.1 Top-Level Breakdown (`runs/` and `artifacts/`)

| Directory | `du -sh` | Category | Summary & Evidence Status |
|---|---:|---|---|
| **`artifacts/worktrees/`** | **152 G** | **Mixed (124 G Tier-1 Prune / 28 G Review)** | 7 Git worktrees. 3 are clean & merged in `develop` (124 G); 4 are unmerged branches (28 G, of which 21 G is `empf-v8` scaling checkpoints already summarized on `feat/empf-v8`). |
| `artifacts/soft-cut-v1` | 21 M | **Review / Archive** | Referenced by unmerged branch `research/first-principles-memory` (10 refs). |
| `artifacts/rank-matching-v1` | 20 M | **Review / Archive** | Referenced by `research/first-principles-memory` (7 refs). |
| `artifacts/unit-laplace-energy-v1` | 18 M | **Review / Archive** | Referenced by `research/first-principles-memory` (2 refs). |
| `artifacts/support-distance-v1` | 18 M | **Review / Archive** | Referenced by `research/first-principles-memory` (6 refs). |
| `artifacts/energy-score-v1` | 18 M | **Review / Archive** | Referenced by `research/first-principles-memory` (7 refs). |
| `artifacts/all-pairs-energy-v1` | 18 M | **Review / Archive** | Referenced by `research/first-principles-memory` (2 refs). |
| `artifacts/prior-routing-v1` | 304 K | **Review / Archive** | Referenced by `research/first-principles-memory` (5 refs). |
| **`runs/forge/`** | **170 G** | **Mixed (see Section 3.2)** | 47 G `queues/`, 45 G `worktrees/`, 52 G `dcgan-performance-cloud-*`, 14 G `tmp/`, 3.7 G `frozen-checkouts/`, ~8.3 G individual study/queue dirs. |
| `runs/forge-rhbi-v7/` | 13 G | **Tier-2 Prune (12 G) + Review (1.4 G)** | 12 G is `v8-scaling-queue/dcgan-performance-cloud-679a26b257b6` (receipts committed in `reports/forge/scaling-receipts/`); 1.4 G is `qual-queue`, `v8-queue`, and `screen*` studies. |
| `runs/forge-rhbi-refresh/` | 12 G | **Tier-2 Prune** | 12 G is `dcgan-performance-cloud-dc48d630d646` (60 cells, 11.14 GiB `.pt`; all receipts committed in `reports/forge/scaling-receipts/dcgan-performance-cloud-dc48d630d646/`). |
| `runs/v11_laws_research/` | 635 M | **Review / Archive** | Recent Nov 20 research (`forge_queue_final` 634 M + CIFAR law2/3/4 probes); 0 committed refs yet. |
| `runs/forge-rhbi-v9/` | 427 M | **Review / Archive** | Nov 13 RHBI-v9 calibration & waves 1–6 (741 `.pt`, 90 `.gif`); 0 committed refs. |
| `runs/forge-rhbi-v12/` | 310 M | **Review / Archive** | Nov 14 RHBI-v12 probes 5–11 (267 `.pt`, 77 `.gif`); 0 committed refs. |
| `runs/forge-rhbi-v10/` | 282 M | **Review / Archive** | Nov 14 RHBI-v10 probes & verification runs (348 `.pt`, 68 `.gif`); 0 committed refs. |
| `runs/forge-stage1/` | 57 M | **Review / Archive** | Referenced by `reports/forge/technique-inventory.json` and `attempt-coverage.json` (`attempt-coverage-empf-v8` 39 M + logs). |
| `runs/cifar_toy_bench/` | 29 M | **Review / Archive** | Recent Nov 20 CIFAR toy benchmarks (`joint_2420a3232`, `joint_13775ca3c`, `baseline_v1`). |
| `runs/v11_opt_research/` | 18 M | **Review / Archive** | Recent Nov 20 v11 optimization ablations (`readme_100g_v2` 15 M, scripts/logs). |
| `runs/readme-hero-v0110/` | 15 M | **Review / Archive** | Referenced by `reports/readme-100gaussians/summary-current.json` (hero GIF source run). |
| `runs/hf_cache/` & `runs/tmp/` | 1.8 M | **Active Migration Cache** | Created 2026-11-21 for HF migration (`HF_HUB_CACHE` / `HF_XET_CACHE` / `TMPDIR`). **Do NOT prune while HF migration runs are active.** |

---

### 3.2 Deep Dive: `runs/forge/` (170 GB)

File-type audit of `runs/forge/` (excluding `worktrees/` and `frozen-checkouts/`, 120.89 GiB):
- `*.pt`: **97.91 GiB (81.0%)** across 10,615 files
- `*.json`: 14.85 GiB (96,718 files)
- `*.npz`: 3.14 GiB (11,950 files)
- `*.gif`: 2.88 GiB (8,599 files)

| Subdirectory under `runs/forge/` | `du -sh` | Category | Details |
|---|---:|---|---|
| `runs/forge/queues/` | **47 G** | **Mixed (Tier-2 Prune / Review)** | 17 queue dirs: `release-v0110-scaling` (13 G; 12 G is committed `dcgan-performance-cloud-e132cf1f36af`), `unified-v4` (8.3 G), `theory2-critic` (5.1 G), `r4-razor` (4.4 G), `nash-v3` (3.6 G), `r4-complete-v11` (1.8 G), `r4-minimal-v2` (1.6 G), `r4-complete-final` (1.5 G), `r4-minimal-final` (1.4 G), `theory3-rigor` (1.2 G), `cond-ddgan-final` (1.1 G), `cond-ddgan` (1.1 G), `calibration-holdout` (1.1 G), `r4-complete-v2` (970 M), `r4-complete` (962 M), `release-v0110-unified-v11` (658 M), `r4-minimal` (121 M). |
| `runs/forge/worktrees/` | **45 G** | **Tier-1 Prune (42.9 G) / Review (2.1 G)** | 15 Git worktrees: **14 are clean & merged in `develop`** (`theory1-coupling` 6.3 G, `theory3-dynamics` 5.5 G, `unified-v4` 3.4 G, `r4-solution-complete` 3.1 G, `shepherd-unify` 2.7 G, `theory2-critic` 2.6 G, `cond-ddgan` 2.6 G, `theory4-arch` 2.5 G, `theory3-rigor` 2.5 G, `r4-ablation-razor` 2.5 G, `nash-complete` 2.5 G, `r4-minimal-core` 2.4 G, `theory4-imbalance` 2.3 G, `calibration-holdout` 2.3 G). **1 is unmerged:** `readme-hero-gif` (2.1 G, 1 commit ahead of `develop`). |
| `runs/forge/dcgan-performance-cloud-2edfe08a4bf5` | **16 G** | **Tier-2 Prune** | 76 cells / 76 `state.pt`; all scaling receipts committed in `reports/forge/scaling-receipts/dcgan-performance-cloud-2edfe08a4bf5/`. |
| `runs/forge/tmp/` | **14 G** | **Tier-1 Prune** | `pytest-of-martyn` (6.8 G, incl. 3 detached pytest worktrees), `t4_forge_queue` (3.2 G), diagnostic probes (`diag_*`, `prof_*`, `smoke_*`). Scratch only. |
| `runs/forge/dcgan-performance-cloud-ee2635996fe6` | **12 G** | **Tier-2 Prune** | 60 cells / 60 `state.pt`; receipts committed in `reports/forge/scaling-receipts/dcgan-performance-cloud-ee2635996fe6/`. |
| `runs/forge/dcgan-performance-cloud-184aa7597be3` | **12 G** | **Tier-2 Prune** | 60 cells / 60 `state.pt`; receipts committed in `reports/forge/scaling-receipts/dcgan-performance-cloud-184aa7597be3/`. |
| `runs/forge/dcgan-performance-cloud-a989784d6f16` | **12 G** | **Tier-1 Prune** | 60 cells / 60 `state.pt`; **0 references** in `reports/` or `scaling-receipts/` (superseded run). |
| `runs/forge/frozen-checkouts/` | **3.7 G** | **Tier-1 Prune** | 2 reproducible Git checkouts (`50398cb034...` 1.9 G, `c2c73f2082...` 1.8 G); 0 references. |
| `runs/forge/queue-theory1-prior`, `queue-shepherd-unify{,-v2,-v3}`, `queue`, `smpf-v{1,2,3}-queue` | **5.8 G** | **Review / Archive** | Referenced by committed Forge readouts & leaderboards (`formulation_comparison.json`, `readout-*.json`). Compact summaries are committed; raw checkpoints remain unarchived. |
| `runs/forge/t4-arch-v{1,2,3}-*`, `rhbi-v{1..12}-study`, `empf-v8-*`, `rhbi-*-v1`, `t4-imbalance-nash-stretch-v1`, `particle-interp-stretch-v1`, `snapshots/` | **7.7 G** | **Review / Archive** | Individual study directories; most referenced by `reports/forge/records/readout-*.json` and `attempts/`. |
| `runs/forge/attempt-coverage-refresh-v8` (38 M), `superseded-scaling` (25 M), `dcgan-performance-cloud-2c7d296aac20` (8 K) | **63 M** | **Tier-1 Prune** | 0 report references; aborted/superseded scratch. |

---

### 3.3 Deep Dive: `artifacts/worktrees/` (152 GB)

| Worktree | `du -sh` | Branch | Merged in `develop`? | Dirty? | Internal Heavy Content | Classification |
|---|---:|---|---|---|---|---|
| `artifacts/worktrees/rhbi-family` | **57 G** | `feat/rhbi-v12` | **Yes** | Clean | 51 G `runs/forge/` (`dcgan-performance-cloud-2ebe5d77de1d` 23 G, `scaling/dcgan-performance-cloud-38b89eb060b8` 17 G, `dcgan-performance-cloud-759e40be701e` 12 G — **all 3 committed** in `scaling-receipts/`!) + 5.6 G `reports/` | **Tier-1 Prune** |
| `artifacts/worktrees/empf-family` | **38 G** | `feat/empf-family` | **Yes** | Clean | 35 G `runs/` (`dcgan-performance-cloud-66339a696dff` 18 G, `dcgan-performance-cloud-1acd41cb5f4a` 11 G — **both committed**; `frozen-checkouts` 3.6 G; `runs/tmp` 2.4 G) + 2.3 G `reports/` | **Tier-1 Prune** |
| `artifacts/worktrees/empf-v3` | **29 G** | `feat/empf-v3` | **Yes** | Clean | 25 G `runs/forge/` (`dcgan-performance-cloud-6c2ecde629a6` 23 G — **committed**; `frozen-checkouts` 1.8 G) + 3.5 G `reports/` | **Tier-1 Prune** |
| `artifacts/worktrees/empf-v8` | **25 G** | `feat/empf-v8` | **No (3 commits)** | Clean | 21 G `runs/forge/dcgan-performance-cloud-2c7d296aac20` (receipts committed on `feat/empf-v8`!) + 3.2 G `reports/` | **Tier-1 Prune nested cloud (21 G); Review branch (4 G)** |
| `artifacts/worktrees/rhbi-texture-active-pair` | **2.1 G** | `research/rhbi-texture-feature-transport` | **No (20 commits)** | Clean | 2.0 G `reports/` checkout only | **Review branch before pruning worktree** |
| `artifacts/worktrees/tier1-conv-texture32` | **2.0 G** | `feat/tier1-conv-texture32` | **No (2 commits)** | **1 file (`M reports/forge/import-gaps.json`)** | 1.9 G `reports/` checkout only | **Review dirty file & branch before pruning** |
| `artifacts/worktrees/first-principles-memory` | **1.9 G** | `research/first-principles-memory` | **No (16 commits)** | Clean | 1.9 G `reports/` checkout only (references the 7 non-worktree `artifacts/*-v1` dirs) | **Review branch before pruning worktree** |

> **Important Git note:** Removing a Git worktree (`git worktree remove`) only deletes the checked-out working directory and its untracked `runs/`; **the Git branch ref and all its commits remain safely in `.git/refs/heads/`**. Still, we keep the 5 unmerged worktrees out of Tier 1 so their branches and `tier1-conv-texture32`'s dirty file can be reviewed first.

---

## 4. Copy-Pasteable Cleanup Commands (NOT EXECUTED)

> **WARNING:** None of the commands below have been executed. Verify no background job is actively reading a target directory before running.

### Tier 1 — Immediate Zero-Risk Cleanup (~217 GB Recovered on `/nvme`)

Removes:
- 3 merged, clean worktrees in `artifacts/worktrees/` (**124 GB**)
- 14 merged, clean worktrees in `runs/forge/worktrees/` (**42.9 GB**)
- 3 detached pytest worktrees + `runs/forge/tmp/` (**14 GB**)
- `runs/forge/frozen-checkouts/`, unreferenced `dcgan-performance-cloud-a989784d6f16`, stub/superseded dirs (**15.8 GB**)
- `artifacts/worktrees/empf-v8/runs/forge/dcgan-performance-cloud-2c7d296aac20` (raw `.pt` checkpoints already summarized on `feat/empf-v8`, **21 GB**)

```bash
cd /nvme/dev/hypergan/ParticleGAN

# 1. Remove 3 merged, clean worktrees in artifacts/worktrees/ (~124 GB)
for wt in rhbi-family empf-family empf-v3; do
  git worktree remove --force "artifacts/worktrees/$wt"
done

# 2. Remove 14 merged, clean worktrees in runs/forge/worktrees/ (~42.9 GB)
for wt in \
  calibration-holdout cond-ddgan nash-complete \
  r4-ablation-razor r4-minimal-core r4-solution-complete \
  shepherd-unify theory1-coupling theory2-critic \
  theory3-dynamics theory3-rigor theory4-arch \
  theory4-imbalance unified-v4
do
  git worktree remove --force "runs/forge/worktrees/$wt"
done

# 3. Remove detached pytest worktrees and scratch tmp/frozen-checkouts (~17.7 GB)
git worktree remove --force runs/forge/tmp/pytest-of-martyn/pytest-134/test_republish_registered_inve0/wt 2>/dev/null || true
git worktree remove --force runs/forge/tmp/pytest-of-martyn/pytest-141/test_republish_registered_inve0/wt 2>/dev/null || true
git worktree remove --force runs/forge/tmp/pytest-of-martyn/pytest-163/test_clean_cpu_ci_and_non_a6000/wt-clean-cpu-ci 2>/dev/null || true
rm -rf \
  runs/forge/tmp \
  runs/forge/frozen-checkouts \
  runs/forge/dcgan-performance-cloud-a989784d6f16 \
  runs/forge/dcgan-performance-cloud-2c7d296aac20 \
  runs/forge/superseded-scaling \
  runs/forge/attempt-coverage-refresh-v8

# 4. Remove empf-v8 nested scaling checkpoints (receipts already on feat/empf-v8; ~21 GB)
rm -rf artifacts/worktrees/empf-v8/runs/forge/dcgan-performance-cloud-2c7d296aac20

# 5. Prune stale git worktree metadata
git worktree prune
```

### Tier 2 — Committed DCGAN Scaling Cloud Raw Checkpoints (~76 GB Additional on `/nvme`)

Every scaling cloud below has its compact JSON scaling receipts already committed in `reports/forge/scaling-receipts/dcgan-performance-cloud-<id>/`; >98% of disk usage is raw `state.pt` weights (~950 MB/cell):

```bash
cd /nvme/dev/hypergan/ParticleGAN

# Remove 6 scaling-cloud dirs whose receipts are already committed in reports/forge/scaling-receipts/ (~76 GB)
rm -rf \
  runs/forge/dcgan-performance-cloud-2edfe08a4bf5 \
  runs/forge/dcgan-performance-cloud-ee2635996fe6 \
  runs/forge/dcgan-performance-cloud-184aa7597be3 \
  runs/forge-rhbi-refresh/dcgan-performance-cloud-dc48d630d646 \
  runs/forge-rhbi-v7/v8-scaling-queue/dcgan-performance-cloud-679a26b257b6 \
  runs/forge/queues/release-v0110-scaling/dcgan-performance-cloud-e132cf1f36af
```

*(Optional ultra-conservative variant: instead of deleting the whole scaling-cloud directories, delete only `state.pt` files inside them to recover ~74 GB while preserving all local JSONs, logs, and GIFs:)*
```bash
find \
  runs/forge/dcgan-performance-cloud-2edfe08a4bf5 \
  runs/forge/dcgan-performance-cloud-ee2635996fe6 \
  runs/forge/dcgan-performance-cloud-184aa7597be3 \
  runs/forge-rhbi-refresh/dcgan-performance-cloud-dc48d630d646 \
  runs/forge-rhbi-v7/v8-scaling-queue/dcgan-performance-cloud-679a26b257b6 \
  runs/forge/queues/release-v0110-scaling/dcgan-performance-cloud-e132cf1f36af \
  -name 'state.pt' -type f -delete
```

### Tier 3 — Post-Review Cleanup (~12 GB on `/nvme` + up to 26 GB Emergency Relief on `/home`)

1. **After reviewing/pushing the 5 unmerged worktree branches** (`feat/empf-v8`, `research/first-principles-memory`, `research/rhbi-texture-feature-transport`, `feat/tier1-conv-texture32` [check `M reports/forge/import-gaps.json` first!], `feat/readme-hero-gif-unified-v11`), remove their worktree checkouts (~12 GB):
   ```bash
   # ONLY after confirming branch refs are pushed/merged and tier1-conv-texture32 dirty change is handled:
   git worktree remove artifacts/worktrees/empf-v8
   git worktree remove artifacts/worktrees/first-principles-memory
   git worktree remove artifacts/worktrees/rhbi-texture-active-pair
   git worktree remove artifacts/worktrees/tier1-conv-texture32   # check git diff first!
   git worktree remove runs/forge/worktrees/readme-hero-gif
   ```
2. **Emergency `/home` disk relief (`/home` is at 97% / 17 GB free):** All 20 committed historical tarballs inside `/home/martyn/dev/particlegan/.worktree-cleanup/20261018T045452Z/` are now byte-exact archived on HF (`3cac9836b867e1a1f49050e68d64255c6a0f7e6d`). After confirming nothing else in that Oct 17 backup is needed:
   ```bash
   # Recovers 26 GB on /home (brings /home from 97% down to ~91%):
   rm -rf /home/martyn/dev/particlegan/.worktree-cleanup/20261018T045452Z
   ```
3. **After all Phase 1–4 HF migration PRs finish**, clear temporary HF transfer caches:
   ```bash
   rm -rf runs/hf_cache runs/tmp
   ```

---

## 5. Summary of Expected Disk Recovery

| Cleanup Tier | Target | Expected Recovery | Risk |
|---|---|---:|---|
| **Tier 1** | 17 merged worktrees + `runs/forge/{tmp,frozen-checkouts}` + unreferenced scaling cloud + `empf-v8` nested scaling checkpoints | **~217 GB** (`/nvme`) | Very Low (all commits in `develop` or `feat/empf-v8`) |
| **Tier 2** | 6 committed `dcgan-performance-cloud-*` dirs (or `state.pt` only) | **~74–76 GB** (`/nvme`) | Low (all scaling receipts committed in Git) |
| **Tier 3a** | 5 unmerged worktrees (after branch review) | **~12 GB** (`/nvme`) | Low once branches pushed/merged |
| **Tier 3b** | `/home/martyn/dev/particlegan/.worktree-cleanup/20261018T045452Z` | **~26 GB** (`/home`) | Low (all 20 committed tarballs now verified on HF) |
| **Total** | **`runs/` + `artifacts/` (+ `/home` backup)** | **~305 GB `/nvme` + 26 GB `/home`** | — |
