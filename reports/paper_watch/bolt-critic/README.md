# Paper watch: BOLT-GAN critic (arXiv:2510.25609)

Question: does replacing ParticleGAN's critic objective with BOLT-GP (a loss
linear in sigmoid of the raw score, with WGAN-GP applied to the raw score)
help a learned particle prior on the paper's 25-Gaussians grid?
Source test: [`tests/test_paperwatch_bolt_critic.py`](../../../tests/test_paperwatch_bolt_critic.py).

Cohort: CPU, one intra-op thread, ka2 recipe, 6,000 updates, batch 256,
lambda_GP 1.0, seeds 0/1/2, shared deterministic initializer. Modes use the
paper's rule: at least one of 10,000 clean samples within 0.15 of a centre.
Metrics are in [`summary.json`](summary.json). [`training.gif`](training.gif)
shows actual seed-0 samples from all three arms at 12 evenly spaced updates.

| Arm | Cohort | Modes (mean, per seed) | Naeem coverage (mean) | In-mode fraction (diagnostic) | Tuning |
| --- | --- | --- | --- | --- | --- |
| plain NS-GAN control (frozen fresh-Gaussian prior) | recipe:ka2 | 25.0 (25/25/25) | 0.461 | 0.311 | none |
| ParticleGAN control (learned 256-particle prior) | recipe:ka2 | 24.33 (25/23/25) | 0.088 | 0.939 | none |
| ParticleGAN + BOLT-GP | assisted host swap | 21.0 (22/24/17) | 0.046 | 0.913 | 6 GPU rounds over lambda_GP, plus one doubling of the update budget |

## Verdict

**FAIL on CPU.** The GPU gauntlet reported that all three arms tied at 25.0
modes. That tie does not reproduce in this pinned CPU cohort. Both "BOLT
matches control" assertions are therefore recorded as strict xfails that
carry the measured values. The thresholds were not loosened.

BOLT-GP also has the lowest coverage of the three arms, matching the GPU
result. The plain control never collapses in this harness, while the paper
reports about 6 of 25 modes. The paper's headline margin therefore has no
headroom here.

The plain control's 25/25 modes and high coverage come from a diffuse cloud
over the whole grid square, not from recovering the modes. Only 31% of its
samples fall within 3 sigma of any centre, compared with 94% for the particle
control (see the GIF). The paper's modes rule and Naeem coverage both reward
that cloud. The in-mode fraction is a report-only diagnostic added after the
run and is not a declared gate. Read "plain beats particle on coverage" in
that light.

## Scope

The BOLT arm is an **assisted/diagnostic cohort**. After the recipe resolves,
the test host replaces `trainer.loss` and `trainer.penalty` by attribute
assignment, because the recipe API has no declared BOLT-loss or raw-score-GP
capability. This result is not recipe evidence and cannot fill any
qualification or matched-comparison cell. To adopt the mechanism, it would
need its own public recipe capability, a declared delta and an evidence
identity.

The original GPU tuning log (RTX A6000, rounds R1 to R6) remains in the test
docstring as the historical record.

Reproduce: `PYTHONPATH=.:tests python tests/test_paperwatch_bolt_critic.py`
(writes this directory). Assertions: `PARTICLEGAN_RUN_PAPERWATCH=1 python -m pytest tests/test_paperwatch_bolt_critic.py`.
