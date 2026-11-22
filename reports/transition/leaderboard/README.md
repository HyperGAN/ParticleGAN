# Transition GAN leaderboard

Primary score: conditional joint SW1 on the fixed held-out geometries, lower is better. Transition residual and class separation are required companion diagnostics. These geometries now serve as a development benchmark because we reuse them for model selection; this leaderboard is not an untouched generalization test.

Every entry uses 1,024 MoG components, bcap, 28,000 updates, batch 256, and training seed 24002. Generator parameter counts stay within 5% of baseline. E and D capacity and wall time are reported; added encoders/critics cost extra compute. No seed-only repeats.

The registry pins summary/source archives and checks identical data/evaluation functions, recipe settings except critic conditioning, normalization, prior initialization metadata, and reference samples. Model/trainer revisions are allowed and recorded; this does not claim identical critic initialization across different architectures.

Encoder rows form a paired-supervision cohort: E adds reconstruction/prediction losses and synthetic composition. They reuse the same real training draws and frozen prior-sampling benchmark, but are not adversarial-only or total-capacity-matched comparisons. E counts must match within that cohort; the historical G budget check is unchanged. Prediction metrics are separate below.

| Rank | Run | Critic conditioning | G class gain | G geometry/time gain | Joint SW1 ↓ | Interp. | Extrap. | Residual ↓ | G / E / D parameters | Train seconds |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | encoder_shared_state | concat | 8 | 1 | 0.08565 | 0.06526 | 0.14681 | 0.01736 | 65,286 / 42,688 / 203,779 | 954.7 |
| 2 | encoder_separate | concat | 8 | 1 | 0.09538 | 0.07449 | 0.15804 | 0.01803 | 65,286 / 42,688 / 238,084 | 826.9 |
| 3 | concat_class8_marginals | concat | 8 | 1 | 0.10027 | 0.08008 | 0.16084 | 0.02393 | 65,286 / 0 / 238,084 | 565.1 |
| 4 | concat_class8 | concat | 8 | 1 | 0.10563 | 0.08454 | 0.16888 | 0.02177 | 65,286 / 0 / 135,169 | 283.1 |
| 5 | concat_class8_context2 | concat | 8 | 2 | 0.10655 | 0.08701 | 0.16518 | 0.01909 | 65,286 / 0 / 135,169 | 252.1 |
| 6 | concat_class8_context4 | concat | 8 | 4 | 0.10692 | 0.09279 | 0.14933 | 0.01550 | 65,286 / 0 / 135,169 | 265.8 |
| 7 | concat_class6_context2 | concat | 6 | 2 | 0.11996 | 0.10223 | 0.17314 | 0.01622 | 65,286 / 0 / 135,169 | 242.8 |
| 8 | concat_class4 | concat | 4 | 1 | 0.14179 | 0.12405 | 0.19503 | 0.01464 | 65,286 / 0 / 135,169 | 318.2 |
| 9 | ucd_marginals | ucd | 1 | 1 | 0.24999 | 0.23839 | 0.28480 | 0.01486 | 65,286 / 0 / 237,448 | 609.6 |
| 10 | concat_joint | concat | 1 | 1 | 0.25182 | 0.23727 | 0.29548 | 0.01532 | 65,286 / 0 / 135,169 | 319.5 |
| 11 | ucd_joint | ucd | 1 | 1 | 0.25501 | 0.23885 | 0.30348 | 0.01529 | 65,286 / 0 / 134,914 | 248.2 |
| 12 | ucd_monolithic | ucd | 1 | 1 | 0.26215 | 0.24316 | 0.31914 | 0.00519 | 65,526 / 0 / 134,914 | 216.1 |

Reference-vs-reference joint SW1 floor: **0.03788**.

## Marginals and preference

Upper-side frequency is measured near the route midpoint, using the observed analytic centerline. Targets are 0.8 for class 0 and 0.3 for class 1. Similar frequencies across classes indicate weak class conditioning; these frequencies alone do not establish support validity.

| Run | State SW1 ↓ | Action SW1 ↓ | Next SW1 ↓ | Upper class 0 | Upper class 1 | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder_shared_state | 0.07631 | 0.10091 | 0.06980 | 0.838 | 0.303 | 0.248 | 0.238 |
| encoder_separate | 0.08171 | 0.10741 | 0.08282 | 0.865 | 0.278 | 0.277 | 0.255 |
| concat_class8_marginals | 0.08385 | 0.11447 | 0.09351 | 0.882 | 0.309 | 0.158 | 0.157 |
| concat_class8 | 0.08982 | 0.11797 | 0.10041 | 0.872 | 0.324 | 0.195 | 0.204 |
| concat_class8_context2 | 0.08906 | 0.12468 | 0.09320 | 0.901 | 0.290 | 0.212 | 0.237 |
| concat_class8_context4 | 0.09718 | 0.12322 | 0.09368 | 0.885 | 0.313 | 0.194 | 0.207 |
| concat_class6_context2 | 0.10391 | 0.13733 | 0.10608 | 0.892 | 0.345 | 0.238 | 0.249 |
| concat_class4 | 0.12387 | 0.15086 | 0.12743 | 0.826 | 0.485 | 0.301 | 0.301 |
| ucd_marginals | 0.24065 | 0.23641 | 0.23913 | 0.577 | 0.571 | 0.262 | 0.240 |
| concat_joint | 0.24514 | 0.23762 | 0.24049 | 0.613 | 0.619 | 0.229 | 0.202 |
| ucd_joint | 0.24402 | 0.24174 | 0.24352 | 0.659 | 0.665 | 0.266 | 0.230 |
| ucd_monolithic | 0.25621 | 0.25135 | 0.25643 | 0.566 | 0.587 | 0.115 | 0.113 |

## Encoder cohort: prediction and composition

Real-input prediction supplies st/at and measures G3 next-state error in physical units. Synthetic SW1 evaluates G1/G2 -> E -> G3. Neither replaces the original prior-sample rank. Here action is displacement: the analytic st + at control has zero prediction error.

| Run | Next-state L2 ↓ | p95 ↓ | Synthetic joint SW1 ↓ | Synthetic residual ↓ | E used / effective components |
|---|---:|---:|---:|---:|---:|
| encoder_shared_state | 0.01665 | 0.04780 | 0.08671 | 0.01471 | 7 / 4.9 |
| encoder_separate | 0.01794 | 0.04958 | 0.09595 | 0.01322 | 5 / 4.1 |

## Experiment notes and artifacts

- **encoder_shared_state:** Paired encoder cohort, matching encoder_separate except one shared state critic for G1/G3 with common physical-state normalization and actual t/t+dt conditioning. Retains three-role marginal weighting. E 42688; D 203779; identical source, data, recipe and budgets across encoder arms. [Config](encoder_shared_state_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/encoder_shared_state_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/encoder_shared_state_transitions.png)
- **encoder_separate:** Paired encoder cohort: E(st,at) -> particle code -> G1/G2/G3; real triple MSE and synthetic st/at reconstruction, adversarial mean of original and composed paths. Adds 42688 E parameters; separate marginal critics; same real draws and original benchmark. [Config](encoder_separate_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/encoder_separate_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/encoder_separate_transitions.png)
- **concat_class8_marginals:** Adds three scalar concat marginal critics at G class scale 8. G and joint D unchanged; marginal_weight=1. Same MoG, data and updates; extra critic compute. GPUs shared with other workloads. [Config](concat_class8_marginals_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_marginals_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_marginals_transitions.png)
- **concat_class8:** G class input scale 8; otherwise matches concat_class4. Same parameters, MoG, data and updates. GPUs shared with other workloads. [Config](concat_class8_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_transitions.png)
- **concat_class8_context2:** Intermediate G geometry/time input scale 2; class scale 8, shared MoG1024 and one concat joint critic unchanged. [Config](concat_class8_context2_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_context2_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_context2_transitions.png)
- **concat_class8_context4:** G geometry/time input scale 4 versus 1; class scale 8, shared MoG1024 and one concat joint critic unchanged. [Config](concat_class8_context4_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_context4_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class8_context4_transitions.png)
- **concat_class6_context2:** G class input scale 6 versus 8 at geometry/time scale 2; same shared MoG1024, one concat joint critic, data and update budget. [Config](concat_class6_context2_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class6_context2_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class6_context2_transitions.png)
- **concat_class4:** Scalar class-conditioned joint D; generator class indicators scaled by 4. Same G parameters, prior, data and updates. [Config](concat_class4_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class4_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_class4_transitions.png)
- **ucd_marginals:** Established MoG baseline; weak preference-class separation. [Config](ucd_marginals_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/ucd_marginals_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/ucd_marginals_transitions.png)
- **concat_joint:** Explicit class input to scalar joint D; no UCD CE. Did not restore preference separation. [Config](concat_joint_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_joint_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/concat_joint_transitions.png)
- **ucd_joint:** Established MoG baseline; weak preference-class separation. [Config](ucd_joint_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/ucd_joint_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/ucd_joint_transitions.png)
- **ucd_monolithic:** Established MoG baseline; weak preference-class separation. [Config](ucd_monolithic_config.yaml) · [Viewer](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/ucd_monolithic_viewer.html) · [Plot](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/transition/leaderboard/ucd_monolithic_transitions.png)
