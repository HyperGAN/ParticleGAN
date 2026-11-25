# Parallel Google Colab Forge Catch-Up (`ops/colab_forge.py`)

`ops/colab_forge.py` orchestrates parallel Google Colab GPU workers (`T4`, `L4`, `A100`, `H100`, or `CPU`) for ParticleGAN Forge, with decentralized branch-merge cache catch-up, automatic GPU right-sizing, crash-proof two-layer VM teardown, and local-only Hugging Face publication.

## Mental model: Forge as a decentralized branch cache

Independent agents can work on separate branches with **zero central coordinator** and **zero Git merge conflicts**:

```text
origin/develop (caught-up cache)
  ├── Branch A: adds new experiment `foo_test`
  │     └── `python ops/colab_forge.py catchup --through-tier 1 --workers 2`
  │         fills ONLY (existing families × foo_test)
  │         writes content-addressed ~4 KB `reports/forge/technique-receipts/<id>.json`
  │
  ├── Branch B: adds new family `bar-v1`
  │     └── `python ops/colab_forge.py catchup --through-tier 1 --candidate bar-v1 --workers 2`
  │         fills ONLY (bar-v1 × existing tasks)
  │         writes content-addressed ~4 KB `reports/forge/technique-receipts/<id>.json`
  │
  └── Branch C merges A + B:
        └── `python ops/colab_forge.py catchup --through-tier 1 --workers 2 --publish-hf`
            sees ONLY (bar-v1 × foo_test) is missing → spins up 1 right-sized GPU,
            fills that single cell, projects its receipt, and publishes locally to HF
```

### Why it stays fast and conflict-free
- **Content-addressed receipts:** Every completed Forge attempt projects via Forge's official `project_receipt()` to `reports/forge/technique-receipts/<attempt_id>.json` (~4 KB, passes `scripts/check_repo_bloat.py`, includes native hardware NVML `J-to-PASS` energy telemetry). Independent branches never touch the same receipt file.
- **Smart cache invalidation:**
  - `--cache-mode branch` *(default)*: skips every `(candidate, task)` cell already satisfied by a committed receipt or local attempt. Automatically checks `git diff --name-only origin/develop...HEAD` so **modifying `configs/forge/ideas/<cand>.json` invalidates only that candidate**, and **modifying `configs/forge/tasks/<task>.json` invalidates only that task** across families, without rerunning unchanged work.
  - `--cache-mode energy`: fills any cell missing hardware NVML `J-to-PASS` telemetry (`cost.energy`).
  - `--cache-mode exact` (or `--strict-source`): requires matching current `source_digest` (rebuilds everything after core `particlegan/` changes).
  - Statically preflight-blocked `(candidate, task)` pairs (e.g., unsupported prior/formulation combinations) are reported under `blocked_count` and never allocate Colab GPUs.
- **Unpushed branches work out of the box:** If `HEAD` is not yet pushed to `origin`, `colab_forge.py` packs a tiny `git bundle` (`merge-base(origin/develop, HEAD)..HEAD`) plus an overlay tarball of any dirty `particlegan/` / `experiments/` / `configs/forge/` files and syncs them to each warm VM.

## Prerequisites

1. Install and authenticate the official Google Colab CLI once locally:
   ```bash
   uv tool install google-colab-cli
   colab sessions   # triggers OAuth consent on first run; verifies auth
   colab usage      # shows compute-unit balance and active burn rate
   ```
2. Ensure you have an SSH public key at `~/.ssh/id_ed25519.pub` (used by `colab ssh`).
3. For optional `--publish-hf`, authenticate `huggingface_hub` locally (`hf auth login`). **`HF_TOKEN` is never sent to Colab** — results are always downloaded locally first, then published atomically from the local machine via `python -m experiments.forge publish --inventory`.

## Common commands

```bash
# 1. Preview the exact cache diff and GPU plan (0 Colab GPUs allocated)
python ops/colab_forge.py catchup --through-tier 1 --cache-mode branch --gpu auto --plan-only

# 2. Catch up missing cells in parallel on Colab (default 2 warm GPUs; Colab caps concurrent GPU allocations at ~2)
python ops/colab_forge.py catchup --through-tier 1 --workers 2 --gpu auto

# 3. Catch up and immediately publish refreshed Parquet/JSON tables to Hugging Face
python ops/colab_forge.py catchup --through-tier 1 --workers 2 --gpu auto --compile --publish-hf

# 4. Fill missing hardware NVML J-to-PASS telemetry on a matched NVIDIA L4 cohort
python ops/colab_forge.py catchup --through-tier 1 --cache-mode energy --gpu L4 --workers 2 --publish-hf

# 5. Full rebuild for current source digest (e.g. after editing particlegan/)
python ops/colab_forge.py catchup --through-tier 1 --cache-mode exact --gpu auto --workers 2

# 6. Run explicit CANDIDATE:STUDY pairs in parallel on matched L4s
python ops/colab_forge.py run \
  --job unified-v11:my-study-a \
  --job rhbi-v12:my-study-b \
  --gpu L4 --workers 2

# 7. Check active sessions / compute-unit balance, or emergency-stop all VMs
python ops/colab_forge.py status
python ops/colab_forge.py reap
```

### Tailing live logs
Every run creates a gitignored directory under `runs/forge/colab/<run-id>/` containing `plan.json`, `summary.json`, and per-worker live logs streaming Forge's `runs/forge/events.jsonl`:
```bash
tail -f runs/forge/colab/<run-id>/forge-*.log
```

## GPU right-sizing & hardware cohorts

| `--gpu` | VRAM | Measured Rate | Recommended Forge workload |
| --- | ---: | ---: | --- |
| `cpu` | — | ~0.07 CU/hr | CPU cohort tasks, probes, validation |
| `T4` | 15.6 GB | ~1.15 CU/hr | Lightweight Tier-1 smoke (`--prefer-t4` under `auto`) |
| `L4` | 23.7 GB | ~1.94 CU/hr | **Default sweet spot** for Tier 1–2 vector, image, VAE/AE, precision, `t3_iso`, `nash`, and `dcgan_performance` |
| `A100` | 40 GB | ~6.80 CU/hr | Tight wall-time / long-horizon tasks (`five_word_joint_*`, `ring_extension`, `*_14k`, `grid100_mlp16`, `grid100_res16_*`, Tier 3–4) |
| `H100` | 80 GB | ~12.00 CU/hr | Heavy scaling or high-VRAM image sweeps |
| `auto` | mixed | adaptive | Routes tight/Tier 3–4 tasks to `A100`, standard Tier 1–2 tasks to `L4`, CPU-only tasks to `cpu` |

**Cohort discipline ([`AGENTS.md`](../AGENTS.md)):** Forge binds `(gpu_model, driver, compute_capability)` in `compute_profile` and `gpu_name` in NVML energy telemetry. Use an explicit single `--gpu` (e.g. `--gpu L4` or `--gpu A100`) whenever running head-to-head A/B comparisons or matched `J-to-PASS` benchmarks.

## Crash-proof teardown & Colab CLI hardening

`ops/colab_forge.py` enforces a two-layer shutdown safety net so GPUs are never stranded burning credits:
1. **Local teardown:** `try/finally`, `SIGINT`/`SIGTERM` handlers, and `atexit` immediately call `colab stop` on every active worker as soon as its queue drains or on any error/interrupt.
2. **VM-side watchdog (survives local machine crash):** Before bootstrapping, each VM launches a detached `/content/watchdog.py` that captures `TBE_RUNTIME_ADDR` and sends `POST http://${TBE_RUNTIME_ADDR}/unassign` if:
   - Total uptime exceeds `--max-minutes` (default `120`),
   - The worker sits idle past `--idle-grace-minutes` (default `10`) after finishing its job without being stopped/downloaded, or
   - Logs and heartbeats stall for `--stall-minutes` (default `25`).
3. **Hard OS-level subprocess timeouts & bounded transport retries:** Every `colab` CLI invocation (`new`, `exec`, `upload`, `ssh`, `download`, `stop`) runs in its own process group with `SIGTERM` → `SIGKILL` enforcement, protecting against `google-colab-cli`'s websocket disconnect hang. Idempotent transport calls (`upload`, `download`, `stop`, and `reap` session queries) perform 1 bounded retry with brief backoff on transient `rc=124` timeouts without ever retrying scientific Forge runs or non-transport exit codes.
4. **Per-worker `--config` isolation + staggered boot:** Each worker gets its own `*.colab.json` session state file and staggered allocation (`--stagger-seconds 10`, `--boot-timeout 150`) to prevent `sessions.json.lock` contention and Colab `503` burst retries.
5. **Isolated-config aware `reap`:** `python ops/colab_forge.py reap` stops active sessions in the default Colab config and scans recent `runs/forge/colab/*/*.colab.json` worker configs (deduplicated, with hard timeouts) to terminate lingering per-worker sessions and print a summary (including any unmatched orphaned server endpoints).
