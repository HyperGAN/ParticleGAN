# Removing the fixed output-bandwidth assumption: bounded evidence

The shared 22-task recipe still depends on a fixed output-noise standard deviation of **0.029**. No tested replacement cleared the frozen older-host screen. The existing 0.029 recipe remains a valid reference; the new rows in this report do not establish a common-22 result.

The motivation is concrete: the 100-Gaussian evaluator's component standard deviation is 0.03, so selecting 0.029 by hand leaves a target-informed width in the proposed shared recipe. Gaussian noise can smooth otherwise disjoint GAN supports [Arjovsky and Bottou (2017)](https://arxiv.org/abs/1701.04862), but that observation does not determine its amplitude or make density-estimation bandwidth equal to the generator's required output noise.

## Frozen screens

All rows kept the frozen host seeds, budgets, data, metrics, and strict original gates. The two wave manifests fixed the task order and configurations before training; each attempted episode was independently regraded from saved curves. One CPU thread per worker and the declared AVX2 profile were used. Skipped hosts were not counted as failures.

| Source epoch and manifest SHA-256 | Declared rows | Saved episodes | Result |
| --- | ---: | ---: | --- |
| `6a8914a`; local ignored manifest: `artifacts/toy100-accuracy/bandwidth-wave-v1-6a8914a/predeclared_manifest.json`; SHA-256 `64a71ad2708c5baf47b890c8e005841e848d7ceaefcca82400809d69e2e0a549` | 56 | 42 PASS, 56 FAIL | 0 new 10/10 survivors |
| `f0c1604`; local ignored manifest: `artifacts/toy100-accuracy/bandwidth-wave-v2-f0c1604/predeclared_manifest.json`; SHA-256 `04c4165db5fa10144d89803e4c27faf51dc55d1901dc63cebac580c05301bd81` | 62 | 71 PASS, 61 FAIL | 0 new 10/10 survivors |

The first wave crossed zero output, fixed or learned output widths 0.005–0.2, zero or 0.5 input noise, and no or 20% output warmup, using the isolated output-noise stream for positive widths. The second wave repeated positive-width interactions on the historical global output stream and included the known noiseless public-Fourier-5 core. The only 10/10 row was the **nonpromotable exact 0.029 reference** with input noise 0.5, end fraction 0.1, and warmup 0.2. The best nonreference second-wave row, fixed output 0.005 with no input noise and warmup 0.2, passed six hosts before failing `img_blobs4`. No new row advanced to fresh 19 or native 100-mode training.

## Unlabeled likelihood bandwidth

I then tested a distinct, data-derived rule: flatten the first real discriminator batch, and choose a positive isotropic Gaussian kernel bandwidth maximizing its leave-one-out log likelihood,

$$
\sum_i \log\left[\frac{1}{n-1}\sum_{j\ne i}\frac{\exp(-\lVert x_i-x_j\rVert^2/(2h^2))}{(2\pi h^2)^{d/2}}\right].
$$

This is a recognized kernel-density bandwidth criterion [likelihood cross-validation study](https://www.tandfonline.com/doi/abs/10.1080/10485259108832513). The implementation uses no labels, centers, mode count, true component width, neighbor count, or chosen quantile. Search bounds come from observed nonzero pair distances and batch size; a bound hit raises rather than silently imposing a fallback bandwidth. If **every** observation has an exact duplicate, the criterion tends upward as the width approaches zero, so the estimator returns zero. A mixture of duplicated and singleton observations retains a positive optimum; focused tests cover both cases. The rule assumes Euclidean flattened data space, an isotropic Gaussian density kernel, and a representative calibration batch. Its selected *density-smoothing width* need not equal the optimal output-noise width for an adversarial generator.

The first real batch was intercepted before a training update, then the actual training run restarted from its fixed seed. The batch SHA-256 matched under the derived candidate config. The 100-mode sampler gave width 0.021496 from 2,048 unlabeled observations for each of grid, rotated, and staggered geometry. Across the older hosts, batch size and dimension made the result vary sharply: trajectory/residual had 12 rows of 16 dimensions and width 0.298999; `mode_hold` had 128 rows of two dimensions and width 0.057925; `vector_overlap` had 128 rows of two dimensions and width 0.338771; four image hosts had 32 rows of 64 dimensions and widths 0.00721–0.00884. Three eight-row discrete hosts had every row duplicated and returned zero. These are measured properties of the unlabeled data, not policy values selected by a gate.

Two source/config-bound 400-step trajectory trials changed only the incumbent or the simpler `reg_kappa=1, reg_coeff=1, prior_reg=0` core's output width to the estimated 0.2989987845. Both saved-evidence regrades were valid **FAIL**, with passing suffix zero and identity MSE 0.096066 and 0.096273. The incumbent 0.029 reference's corresponding MSE was 0.007315. The initial first-batch scratch manifest SHA-256 was `6bfbebc05d155fc8401144016cc80319a4aeff2474a435f73ed4129ac7614d7b`; the retained calibration and two trials are in the local ignored path `artifacts/toy100-accuracy/bandwidth-kde-first-v1-83f2f3e/` and are byte-identical to RAM evidence.

Because trajectory's real support repeats, I predeclared one narrow revision: fit the same likelihood rule to the first **two** real batches from a discarded preflight. Two temporal observations are the minimum needed to see a support point recur when one batch has one copy of each. This replaces the width constant with a calibration-count assumption; it is not parameter-free. The first-two-batch tensors and estimated widths were byte-identical under both tested optimizer cores. Trajectory and residual then selected zero; stripes selected 0.006771; mode_hold selected 0.046400. Each candidate was recalibrated before its episode to verify that changing the output width did not change the two real batches.

| Core | Strict staged saved-evidence result |
| --- | --- |
| Incumbent κ=1.176, coefficient 6, prior regularization 0.05 | trajectory PASS, residual PASS, stripes **FAIL** (HQ 0.71875, one mode) |
| Simpler κ=1, coefficient 1, prior regularization 0 | trajectory PASS, residual PASS, stripes PASS, mode_hold **FAIL** (seven modes, HQ 0.99976) |

The valid two-batch screen has source epoch `1ac7184`, manifest SHA-256 `d2043e6ed63063e153a833856d0d67ddba24e393ac9e844939e5be6746549be3`, and retained source, batches, configs, curves, and regrades at the local ignored path `artifacts/toy100-accuracy/bandwidth-kde-two-screen-v2-1ac7184/`. Its first launch at epoch `66ec752` was **harness-invalid**: concurrent in-process monkeypatching collided before any result was accepted. That attempt is retained separately at the local ignored path `artifacts/toy100-accuracy/bandwidth-kde-two-invalid-v1-66ec752/`. The valid epoch serialized only calibration interception; candidate training remained independent subprocesses. All nine attempted KDE episodes regraded to the recorded verdict after relocation (two first-batch failures plus seven two-batch PASS/FAIL episodes).

## Constraint ledger

| Constraint | Outcome |
| --- | --- |
| Fixed 0.029 output width near known 0.03 target width | **Unresolved.** It remains in the shared passing older-host reference; the tested width-free estimator fails. |
| Output-noise warmup 0.2 | Removing it did not yield a new 10/10 survivor in the bounded fixed/learned waves. |
| Input-noise peak 0.5 and end fraction 0.1 | Zero/constant-input controls did not yield a new 10/10 survivor. |
| Learned positive output scale | Tested from materially different initial widths 0.005–0.2 with the common optimizer; none survived. |
| New assumptions in KDE replacement | Isotropic Euclidean density smoothing; first one or two real batches representative; two-batch preflight and restart. It estimates KDE smoothing, not a generator fidelity optimum. |

The evidence supports **no promotion** of KDE or the width ablations. A future width-free rule should optimize generator distributional fidelity or discriminator behavior directly on unlabeled samples, rather than assume a KDE smoothing scale is the needed output Gaussian spread. Such a rule would need its own frozen mechanism receipts and fresh 19+3 replay; no result here inherits the incumbent's full-22 claim.
