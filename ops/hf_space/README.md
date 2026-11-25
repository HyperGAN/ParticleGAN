---
title: ParticleGAN Forge Leaderboard
emoji: ⚛️
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Interactive Forge leaderboard, laws, Pareto & training GIFs
---

# ParticleGAN Forge Leaderboard

Public, zero-backend interactive researcher dashboard, receipt explorer, Pareto efficiency viewer, and training GIF inspector for [HyperGAN/ParticleGAN](https://github.com/HyperGAN/ParticleGAN), backed directly by the [HyperGAN/particlegan-forge-evidence](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence) Hugging Face Dataset and linked to the [ParticleGAN Research Portal](https://hypergan.github.io/ParticleGAN/).

## What This Space Shows

- **Selected Recipes Leaderboard:** Sortable comparison across Tier 1–4 counted view requirements, the fixed full-suite benchmark (failures included), and stretch experiments, together with qualification policy notes and tuning-budget disclosures.
- **Research Laws & Families Matrix:** Dynamic evolving-laws registry, interactive Family × Law coverage heatmap with plain-English mechanism explanations and known limits, KaTeX-rendered mathematical formulations, Mermaid update-loop diagrams, and copyable PyTorch `get_recipe(...)` quickstarts.
- **Energy & Pareto Efficiency:** Interactive zero-dependency SVG Pareto scatter plot (`grid100 J-to-PASS`, `r256 J/step`, `r256 ms/step` vs. suite pass rate) with strict GPU-cohort and contention separation alongside full hardware NVML telemetry tables.
- **Task & Receipt Explorer + Experiment Deep-Dive:** Filter family × task outcomes by family, law, edge case, tier, status, role, GIF availability, and search. Click any task ID (or share `#experiment/<task_id>`) to open its Experiment Deep-Dive view with target-geometry SVG schematic, protocol Mermaid diagram, exact numerical gates, cross-family outcomes, side-by-side training GIFs, and copyable CLI reproduction command.
- **Training GIF Inspector:** Side-by-side and multi-family comparison of actual saved training GIFs streamed directly from `HyperGAN/particlegan-forge-evidence`, with law-based task filtering.

## Scientific Caveats

Counts describe screening and selection evidence only. Scientific superiority or public-default adoption requires report-bound accepted calibration and independently frozen confirmation evidence; software repairs grant neither. Rows with different tuning budgets are not equally tuned comparisons. Not measured, blocked, incompatible, and error outcomes remain distinct from numerical `FAIL` and earn no pass credit. See [`reports/forge/technique-inventory.md`](https://github.com/HyperGAN/ParticleGAN/blob/develop/reports/forge/technique-inventory.md) for the authoritative specification.

## Deploying Updates

From the `HyperGAN/ParticleGAN` repository root:

```bash
python ops/hf_space/deploy.py --dry-run
python ops/hf_space/deploy.py
```
