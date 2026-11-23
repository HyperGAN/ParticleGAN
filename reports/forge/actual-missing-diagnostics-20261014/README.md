# Actual attempts of the missing diagnostics

**27 new training attempts: 2 PASS, 21 FAIL, 4 INCOMPLETE.** They performed 53,440 optimizer updates. All 15 native continuation starts were also attempted; each stopped before training because its required passing own-checkpoint receipt was unavailable. The dispatcher reached four concurrent tests and two per GPU. All attempts are terminal.

This follows the request to actually try the unrun and incompatible cases, including diagnostics after failed lower-tier gates. No formulation, numerical gate, training horizon, prior policy or optimizer was changed. These are separate diagnostic measurements; the selected ordinary qualification is preserved. [Exact outcomes, source identities and hashes](results.json).

## Atlas and E22 really do run

The blanket Forge policy guard rejected the original image calls before constructing a model. The public trainer nevertheless ran both unchanged formulations through all four 600-update image tasks. Both passed Stripes. Bars, Blobs and Intensity failed their original sustained numerical criteria. Intensity reached a passing final endpoint but only four consecutive passing observations; five are required.

| Unchanged public formulation | Stripes | Bars | Blobs | Intensity | Total |
| --- | --- | --- | --- | --- | --- |
| Atlas | PASS | FAIL | FAIL | FAIL | 1/4 |
| E22 | PASS | FAIL | FAIL | FAIL | 1/4 |

[Actual image trajectories](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/actual-missing-diagnostics-20261014/image-diagnostics.gif) show every scored output and the target templates. These runs retain the public policy's selected serving state. The ordinary image cards demand live weights, so these measurements are displayed as diagnostics rather than silently replacing that contract. No policy, noise, averaging or row control was disabled to make the tests run.

Atlas's [original 19/19 CUDA result](../../develop-gates-20261001/README.md) remains valid under its original hosts and noisy sampling law. The current ordinary row's incompatibility labels do not mean that Atlas has never run or cannot learn these problems.

## The previously skipped deeper tests

The exact selected recipes ran the original strict clock probe, ring-endurance producer and paired target-shift test. Gate-only prerequisites were not used to prevent these independent diagnostic executions. Explicit checkpoint consumers still require their actual producer state.

| Selected recipe | Strict clock audit | Ring hold | Target shift | Native 14k continuations | Ring extension |
| --- | --- | --- | --- | --- | --- |
| BCAP | PASS, reused prior audit | FAIL | FAIL | 3 not run: checkpoint required | not run: hold failed |
| K3P | FAIL | FAIL | FAIL | 3 not run: checkpoint required | not run: hold failed |
| KA2 | FAIL | FAIL | FAIL | 3 not run: checkpoint required | not run: hold failed |
| R1/R2 | FAIL | FAIL | FAIL | 3 not run: checkpoint required | not run: hold failed |
| GAN v3 release 0.7 | FAIL | FAIL | FAIL | 3 not run: checkpoint required | not run: hold failed |

The four new strict audits completed all 24 updates and failed because changing the step label changed the learning state. They were measured failures, not merely inferred from the source audit. BCAP's existing strict PASS was reused without retraining.

The leaderboard now treats those proven-runnable clock tasks as not run in the original runtime, with their actual FAIL results in the diagnostic rows. A static clock dependency no longer labels them incompatible merely because they cannot pass the clock-free claim.

BCAP acquired the required 200-check ring window at update 3,729, then failed its hold at update 3,760. The other four ring producers exhausted the 4,800 settling checks at update 6,000 without acquiring the required window. Every paired target-shift run completed all 3,600 updates and failed its original recovery criterion. [Actual recorded learning curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/actual-missing-diagnostics-20261014/deeper-diagnostics.gif) use the saved measurements, with no additional training or sampling.

The original native continuation adapter was invoked for each of the 15 candidate/task pairs. Its own-checkpoint check refused every start. Supplying a fictional PASS, restarting a continuation from initialization, or changing the test would not be an unchanged execution; none was done.

## What the Atlas/E22 policy attempts uncovered

The existing selected-cloud two-pole adapter was incorrectly rejected by inherited optional transport metadata, even with transport disabled. A narrow preflight fix admits that existing owner only when both transport weights are zero. Active transport remains blocked. The full Atlas and E22 two-pole runs then completed all 80 updates.

| Actual attempt | Updates | Result and concrete cause |
| --- | ---: | --- |
| Atlas two-pole | 80 | INCOMPLETE receipt: missing recipe-owned prior-policy metadata. Measured mean absolute coordinate 0.002586, below the required 0.3. |
| E22 two-pole | 80 | Same missing metadata; measured mean absolute coordinate 0.002586, below 0.3. |
| E22 Gaussian acquisition | 1,000 | ERROR / INCOMPLETE during checkpoint hashing: a legitimate policy diagnostic NaN is rejected by the JSON state digest. All 24 observations were saved; final CDF KS 0.029021. |
| E22 Ring16 acquisition | 1,600 | Same checkpoint hashing ERROR / INCOMPLETE after all 96 observations. Final component covariance error 1.139007 exceeds the 0.85 bound. |
| E22 policy clock audit | 24 | FAIL on the actual clock/state comparisons. |

[Actual saved policy observations](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/actual-missing-diagnostics-20261014/policy-diagnostics.gif) show the two-pole, Gaussian and Ring16 trajectories without adding updates or draws.

The saved Gaussian checkpoint passes the existing `finite_policy_state` health check. Its export error is a serialization problem, not evidence that the learned tensors are nonfinite. The original error receipts are preserved; no missing attestation was fabricated and no error was converted into a qualified pass.

The ordinary MoG tasks have a separate concrete incompatibility: constructing their exact prior with Atlas/E22 raises `particle_birth_death requires prior_kind='particles'`. Conditional, encoder and joint-word hosts also require ownership/row semantics that the independent formulation does not supply. Those cases remain skipped under the instruction not to change the formulation.

## Evidence and checks

The compact [results](results.json) bind the original local receipts, request files, execution-source digests, candidate revisions and dispatch logs by SHA256. Raw logs, checkpoints and sampled tensors stay in `/ml2/hypergan/fill-tier-values-20261014-run/`. The deeper runs used the exact selected source `a0493ca3c213d559523eadc76e244ec9f9c1f81f`; all new runs used Torch 2.13.0+cu126, separately from the leaderboard's recorded Torch 2.14 cohort.

The only execution-code change is the two-pole preflight metadata fix. All 136 focused checks pass, including 30 policy tests covering rejection of active transport and preservation of the original source-bound policy cohort. The diagnostic rows add no qualification credit and do not modify the selected scientific rows.
