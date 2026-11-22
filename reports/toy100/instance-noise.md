# Instance-noise mechanism probe

The five conventional MLP architecture changes did not pass the grid gate.
This separate probe pairs two geometry-blind mechanisms: fresh isotropic
Gaussian output noise (standard deviation 0.026) addresses collapsed
within-mode spread, and fresh discriminator input noise (standard deviation
0.5 at the start, linearly decreasing to zero by update 3,500) gives the
critic a coarse distribution view before it resolves individual modes.
The input noise enters every real and generated discriminator call in both
the D and G updates; it uses its own seeded random stream. It does not use
target centers or mode labels in training.

The base recipe uses one MLP architecture (Fourier 2, width 128, depth 3),
20,000 learned particles, batch 512, LR 0.0006, discriminator/prior LR
multipliers 1.5/10, `b_cap` coefficient 1, prior regularization 0.05,
β₂ 0.999, cosine anneal from 40% with a 0.05 floor, seed 1234, and 7,000
updates. The paired GPU control has the same output noise and recipe but no
discriminator input noise. Both use the unchanged 20,000-draw, five-terminal-
check live gate.

| Device and arm | Grid | Rotated | Staggered | Final staggered modes / weakest count / mass TV |
| --- | --- | --- | --- | --- |
| GPU1, paired control (input σ=0) | FAIL: 97 modes, HQ 98.74%, TV 0.1206 | unmeasured | unmeasured | — |
| GPU1, input σ=0.5→0 | **PASS**: 100 modes, HQ 99.00%, TV 0.0666 | **PASS**: 100 modes, HQ 98.78%, TV 0.0578 | FAIL | 99 / 99 / 0.1052 |
| CPU, input σ=0.5→0 | **PASS**: 100 modes, HQ 98.12%, TV 0.0835 | **PASS**: 100 modes, HQ 98.92%, TV 0.0546 | FAIL | 99 / 98 / 0.0952 |

Both passing GPU problems first cover all 100 modes at update 5,500, begin
the terminal full-quality suffix at 6,000, and confirm at 7,000. The CPU
grid and rotated runs also pass the final five checks from 6,000. The
batch-512 staggered result is a real failure in both devices: one mode has fewer than
the required 100 in-radius draws, so it never earns a complete mode count or
a passing suffix. The GPU version also misses the 0.10 mass-TV bound.
These shared batch-512 settings alone are not an all-problem recommendation.

The GPU aggregate gate is `FAIL (2/3)` with full curves and a 14-frame
recorded-snapshot animation at `artifacts/toy100/instance-noise/sigma05`.
The CPU evidence is under `artifacts/toy100/instance-noise-cpu/sigma05`.
Rescoring each saved final 20,000-draw cloud on CPU reproduced every CPU
scalar exactly and every GPU gate verdict; the largest GPU scalar difference
was 7.2×10⁻⁶ in a median radial ratio. Every run records its config and a
copy of the exact probe source matching the saved SHA256 provenance.

## Bounded staggered repair screen

Three declared changes were tried on the staggered grid alone, with the
same seed, output noise, model, budget, and evaluator. Each remains a failure.

| Input-noise schedule / prior regularization | Final modes | Weakest in-radius count | HQ | Mass TV | Other failure |
| --- | ---: | ---: | ---: | ---: | --- |
| σ=0.5→0 by 50%, prior reg 0.05 (shared starting arm) | **99** | **99** | 98.07% | 0.1052 | — |
| σ=0.5→0 by 50%, prior reg 1 | 96 | 70 | 97.75% | 0.1020 | — |
| σ=1.0→0 by 50%, prior reg 0.05 | 95 | 21 | 96.92% | 0.1247 | HQ below 97% |
| σ=0.75→0 by 60%, prior reg 0.05 | 99 | 91 | 97.65% | 0.0910 | max covariance ratio 1.821 > 1.70 |

The last three arms and their frozen plan are preserved at
`artifacts/toy100/instance-noise-repair` and
[`search_instance_noise.json`](../../configs/toy100/search_instance_noise.json).
Independent rescoring of their final saved clouds reproduced the recorded
metrics to numerical precision, and the archived launcher/model/plan hashes
match each run's provenance. Increasing noise or prior regularization did not
repair the remaining allocation failure at this fixed budget.

## Successful batch-size follow-up

One further fixed-seed CPU probe changed **only staggered100 batch size from
512 to 1024**, keeping input noise σ=0.5→0 by 50%, output noise σ=0.026,
prior regularization 0.05, and the same 7,000-update budget. It earned the
[strict individual gate PASS](search/trials.json):
all 100 modes were covered by update 5,500; full quality began at update
6,000 and held for each of the five terminal live checks through update
7,000. The final 20,000 draws had HQ 98.26%, weakest in-radius count 113,
mass TV 0.0750, worst covariance eigenvalue ratios 0.499–1.038, and radial
median ratios 0.777–1.118. Its [run receipt](search/trials.json),
[probe source](search/sources/1a9f0daff39025e231fbf66d82222437e76f5391f76769aa9639008dd535f92a.py),
and [schedule/batch search report](search-schedule.md) preserve the full
curve and exact implementation.

The [recommended manifest](../../configs/toy100/recommended.json) declares
batch 512 for grid100 and rotated100, with an explicit staggered100 batch
1024 override. The three individually passing CPU curves support that one
command suite recipe. The [production all-problem gate](recommended/gate.json)
passes 3/3, with five terminal live checks for each problem; its
[leaderboard](recommended/leaderboard.md) and
[animation](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy100/recommended/toy100-progress.gif) retain the
full evidence. Every production evaluation metric row and all final scored
sample arrays match the saved passing CPU probes exactly. The
[toy100 report index](README.md) links the aggregate result. The unchanged
batch-512 staggered failure remains visible above.
