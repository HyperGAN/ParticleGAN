---
title: ParticleGAN Forge Leaderboard
emoji: ⚛️
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Interactive Forge benchmark leaderboard & training GIFs
---

# ParticleGAN Forge Leaderboard

Public, zero-backend interactive leaderboard and receipt explorer for [HyperGAN/ParticleGAN](https://github.com/HyperGAN/ParticleGAN), backed directly by the [HyperGAN/particlegan-forge-evidence](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence) Hugging Face Dataset.

## What This Space Shows

- **Selected Recipes Leaderboard:** Sortable comparison across Tier 1–4 counted view requirements, the fixed 69-test full-suite benchmark (failures included), and stretch experiments, together with qualification policy notes and tuning-budget disclosures.
- **Task / Receipt Explorer:** Filter recorded family × task outcomes by family, tier, status (`PASS`, `FAIL`, `blocked`, `not_run`, `incompatible`, `error`), role (`required`, `stretch`, `held`, `removed`), GIF availability, and free-text search, with exact recipe/source provenance hashes.
- **Training GIF Inspector:** Side-by-side and multi-family comparison of actual saved training GIFs streamed directly from `HyperGAN/particlegan-forge-evidence`.

## Scientific Caveats

Counts describe screening and selection evidence only. Scientific superiority or public-default adoption requires report-bound accepted calibration and independently frozen confirmation evidence; software repairs grant neither. Rows with different tuning budgets are not equally tuned comparisons. Not measured, blocked, incompatible, and error outcomes remain distinct from numerical `FAIL` and earn no pass credit. See [`reports/forge/technique-inventory.md`](https://github.com/HyperGAN/ParticleGAN/blob/develop/reports/forge/technique-inventory.md) for the authoritative specification.

## Deploying Updates

From the `HyperGAN/ParticleGAN` repository root:

```bash
python ops/hf_space/deploy.py --dry-run
python ops/hf_space/deploy.py
```
