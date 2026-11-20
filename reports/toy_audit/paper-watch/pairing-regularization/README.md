# Paper watch: pairing regularization (arXiv:2604.20130)

Question: does the paper's contrastive pairing term help ParticleGAN's
learned particle prior on an 8-Gaussian ring (radius 2, sigma 0.15)?
Source test: [`tests/test_paperwatch_pairing_regularization.py`](../../../../tests/test_paperwatch_pairing_regularization.py).

Cohort: CPU, one intra-op thread, ka2 recipe, 2,400 updates, batch 128,
seeds 0/1/2, shared deterministic initializer. Metrics are in
[`summary.json`](summary.json); [`training.gif`](training.gif) shows actual
seed-0 samples from all three arms at 12 evenly spaced updates. Its frame
titles show a 1,024-sample coverage, which is not on the same scale as the
4,096-sample gate metric.

| Arm | Cohort | Naeem coverage (mean) | Dispersion ratio (mean, ideal 1.0) | Tuning rounds |
| --- | --- | --- | --- | --- |
| plain GAN control (frozen fresh-Gaussian prior) | recipe:ka2 | 0.638 | 1.069 | 0 |
| ParticleGAN + pairing, lambda_pair 0.03 | assisted host hook | 0.245 | 1.461 | 6 |
| ParticleGAN control (learned 256-particle prior) | recipe:ka2 | 0.201 | 0.775 | 0 |

## Scope

The pairing arm is an **assisted/diagnostic cohort**. The test host injects
the pairing network through the `PriorManipulator` penalty hook, and its
parameters ride the generator optimizer. No recipe declares a pairing
regularizer capability, delta or evidence identity. This result is not recipe
evidence and cannot fill any qualification or matched-comparison cell.

## Findings

- PARTIAL transfer. Pairing raises the particle control's coverage by 0.044
  (ahead in every seed) and turns intramode concentration (0.78) into
  overdispersion (1.46). It stays far behind the plain control.
- Recorded control finding: with no tuning on either side, the plain
  fresh-Gaussian control outperforms the learned-particle control on coverage
  (0.638 vs 0.201) under this recipe and budget. This gap is much larger than
  any pairing effect. It points to the particle prior/update and not to a
  missing generator regularizer. A declared recipe-level study would have to
  confirm it before it can be attributed to a mechanism.
- The per-seed ordering was selected after 6 tuning rounds on the same seeds,
  so it is not independent confirmation.

Reproduce: `PYTHONPATH=.:tests python tests/test_paperwatch_pairing_regularization.py`
(writes this directory). Assertions: `PARTICLEGAN_RUN_PAPERWATCH=1 python -m pytest tests/test_paperwatch_pairing_regularization.py`.
