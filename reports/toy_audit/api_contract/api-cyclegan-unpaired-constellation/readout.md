# Public-API Contract Readout: `api-cyclegan-unpaired-constellation`

- **Case ID:** `api-cyclegan-unpaired-constellation`
- **Forge Task:** `configs/forge/tasks/cyclegan_unpaired_constellation.json`
- **Provider:** `benchmarks.toy_audit.api_cyclegan` (`CycleGANConstellationFixture`)
- **Recipe:** `bispf` (`MoGParticlePrior(num_particles=256, z_dim=2, sigma=0.08, standardize=False)`)
- **Protocol Seed:** `0` (`particlegan.init.deterministic_orthogonal_`)
- **Device:** `cuda:1` (NVIDIA RTX A6000)
- **Budget:** `3500` updates (`batch_size=256`, `eval_samples=4096`, `20` metric observations, `5` terminal checks)
- **Status / Verdict:** `COMPLETE` / `PASS` (`sustained_metric_passed: true`)

## Research Formulation

Inspired by **Kim et al. (ICML 2017, DiscoGAN, arXiv:1703.05192, Figure 2 toy)** and **Zhu et al. (ICCV 2017, CycleGAN, arXiv:1703.10593)**, this benchmark tests unpaired cross-domain translation $(G_{X\to Y}, G_{Y\to X}, D_Y, D_X)$ with bidirectional cycle-consistency regularization:
$$\mathcal{L}_{\text{cyc}}(G_{X\to Y}, G_{Y\to X}) = \mathbb{E}_{x \sim P_X}\|G_{Y\to X}(G_{X\to Y}(x)) - x\|_2^2 + \mathbb{E}_{y \sim P_Y}\|G_{X\to Y}(G_{Y\to X}(y)) - y\|_2^2$$
between two disjoint 8-mode 2D constellations ($\sigma = 0.08$):
- **Domain $X$:** 8-mode ring of radius $R_X = 2.5$ at $\theta_k = 2\pi k / 8$ ($k=0,\dots,7$).
- **Domain $Y$:** 8-mode square-perimeter constellation on $\{-2.0, 0.0, +2.0\}^2 \setminus \{(0, 0)\}$.

Whereas one-sided adversarial translation suffers from many-to-one mode collapse, bidirectional adversarial training coupled with cycle-consistency regularization enforces a mode-level bijection across all 8 modes without paired supervision.

## Terminal Window Metrics (Updates 2800–3500)

| Metric | Gate | Step 2800 | Step 2975 | Step 3150 | Step 3325 | Step 3500 | Status |
|---|---|---|---|---|---|---|---|
| `modes_y` | `>= 7` | `8` | `8` | `8` | `8` | `8` | PASS |
| `modes_x` | `>= 7` | `8` | `8` | `8` | `8` | `8` | PASS |
| `hq_y` | `>= 0.85` | `0.9294` | `0.9294` | `0.9294` | `0.9292` | `0.9292` | PASS |
| `hq_x` | `>= 0.85` | `0.9563` | `0.9563` | `0.9563` | `0.9563` | `0.9563` | PASS |
| `mass_tv_y` | `<= 0.15` | `0.0172` | `0.0172` | `0.0172` | `0.0171` | `0.0171` | PASS |
| `cycle_rmse` | `<= 0.12` | `0.0306` | `0.0306` | `0.0306` | `0.0306` | `0.0306` | PASS |
| `bijection_ber` | `<= 0.15` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | PASS |

## Artifacts

- [`goal.gif`](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/api_contract/api-cyclegan-unpaired-constellation/goal.gif) (`539,814` bytes, 9 frames across steps `0..3500`)
- [`receipt.json`](receipt.json) (`schema: "particlegan_api_toy_run_v1"`)
