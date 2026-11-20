# BISPF investigation: Two intensities: fidelity failure after late two-mode acquisition

The selected BISPF endpoint has both brightness levels, with exactly 16 nearest assignments per level. Only 19/32 images meet the RMSE bound. Its recorded failure is whole-image fidelity, with late qualifying coverage, not endpoint one-mode collapse. This is an investigation of the selected global recipe `bispf-optimized-quintic-v1`, not a new training comparison. The original **FAIL** is retained. The [current technique inventory](../../technique-inventory.md) remains the only goal leaderboard.

## Frozen evidence and protocol

- Candidate revision: `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`; attempt `81b4b1e0cd374e9e80a683e5e192b824`.
- Executed commit: `d9d72d53761c898bce20102e1995fc3d899c1447`; scientific digest `c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213`.
- Canonical original result hash: `70fca26df8ade26a1a98a8a13a487f1bf140592c430526f4fb91f144211d17d5`. [Published compact receipt](../../technique-receipts/81b4b1e0cd374e9e80a683e5e192b824.json) is a display/provenance summary, not a replacement qualification envelope.
- Original full readout: SHA-256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`, 4,926,714 bytes, retained at `/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`. The [published archive pointer](../../bispf-optimized/quality.json) retains its original `/home/martyn/dev/...` identity. The audit verifies the exact bytes at the mounted location.
- Task contract: `dca7a6bbd97d9f50461f27aede455fab14b7852dbbbee5e6e83e54870a542c2f`. [Current task declaration](../../../../configs/forge/tasks/img_intensity2.json) matches the recorded host/measurement contract used here.
- **Seed 0**, public `deterministic_orthogonal_named_parameters_v1` initialization, width-12 transposed-convolution generator (5,173 parameters), critic (2,833 parameters), 8-D latent, 32 learned uniform particle-cloud rows, sigma 0, no standardization. This task's explicit cloud exception applies; the inventory's generic MoG binding is not the image task prior.
- Uniform noisy-template target law: 8×8 grayscale templates with clipped Gaussian pixel noise std 0.01 during training. Evaluation enumerates all 32 learned rows, clean/live weights, no output noise, exactly 24 checks at updates 25,50,…,600.
- 600 updates each for G, D and prior; finite outputs, penalty applied 600 times, zero unintended RNG deviations. Constructor/data/prior/noise/evaluation streams have recorded initial bindings and terminal state hashes in the original receipt. The [audit result](audit-result.json) preserves initial model hashes and named-stream checkpoint hashes.
- Actual recipe: non-saturating GAN loss; BISPF critic penalty; full DualNorm, smoothing 0.001, momentum 0, quintic convolution rule; G lr 0.012, D multiplier 1.5, prior multiplier 2.5; learned prior with SBNP regularizer weight 0.01; no EMA scoring or added output noise. The task's old lr 0.0017, prior weight 0.05 and EMA 0.99 fields are inactive historical provenance, not active training settings. Exact resolved recipe is retained in [audit-result.json](audit-result.json).

## Numerical finding

The unchanged gate requires two modes, each with at least 8/32 clean images; at least 29/32 clean images overall; five consecutive terminal passing observations. Quality uses nearest-template **whole-image RMSE ≤ 0.06**. Count order: central 4×4 patch at intensity 0.35, central 4×4 patch at intensity 0.85.

| Metric | Selected endpoint | Bound | Original result |
| --- | ---: | ---: | --- |
| Qualifying modes | 2 | ≥ 2 | PASS |
| Clean images | 19/32 (0.59375) | ≥ 29/32 | FAIL |
| Terminal passing checks | 0 | ≥ 5 | FAIL |
| Distribution TV (diagnostic) | 0.0 | no task gate | diagnostic |
| Mean RMSE (diagnostic) | 0.063101210 | no mean-RMSE gate | diagnostic |

Nearest assignments are **[16,16]** and clean counts are **[11,8]** at update 600. Five dark-assigned and eight bright-assigned images fail quality. Both populations satisfy the eight-clean threshold, but 13/32 images fail RMSE and at least ten must improve to reach 29/32. Distribution TV is exactly zero because it uses all nearest assignments; balanced assignment mass does not establish image fidelity.

Both qualifying modes first appear at update 600. Earlier recorded states oscillate between zero and one qualifying mode; the dark level first has clean examples at 550, but only three until the last check. No observation meets overall HQ, and maximum HQ is the final 19/32. Even a different terminal HQ could not retroactively supply five sustained passing checks.

For an otherwise exact image whose only error is uniform intensity inside the 4×4 patch, full-image RMSE equals half the intensity error, so 0.06 allows ±0.12 inside the patch. Real misses can instead involve background leakage or geometry. The aggregate readout cannot distinguish those sources, so “brightness is inaccurate” is a task description, not a demonstrated pixel-level cause.

**Historical distinction.** The original BISPF problem narrative says one brightness level and HQ 0.5. That is an earlier cohort. The selected quintic run has two qualifying modes and HQ 0.59375; its FAIL remains unchanged. Do not carry the old endpoint-collapse diagnosis into this source.

## Mechanism examination

Examination is bound to the **executed commit**, not newer `develop` code. File hashes are recorded in [audit-result.json](audit-result.json).

1. [Frozen API image adapter](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/experiments/forge/adapters.py) trains through public `GANTrainer`, draws the noisy target batch once per update, and scores clean enumeration with `fixed_first_n=True`. This is exact finite-prior measurement, not 32 independent Monte Carlo samples. Missing modes here are not evaluation sampling noise.
2. [Frozen recipe loss dispatch](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/recipes.py#L673) only selects `SinkhornRelativisticLoss` when the recipe loss is `relativistic`. The actual recipe says `non_saturating`, so its [adversarial objective](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/gan_loss.py) is ordinary non-saturating loss. Sinkhorn still enters penalty residual/data-scale calculations. The family name does not imply that a balanced coupling supervises output mode populations in this run.
3. [Frozen BISPF penalty](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py#L1136) classifies N×1×8×8 as image tensors: default transport chords are bypassed and thermal scale is fixed to 1. The low-dimensional late harmonic-freeze branch does not apply to this image task. Image-specific endpoint penalty scaling and logit gauge remain active. Therefore a causal story based on image thermal freezing or a chord-bridge force is unsupported.
4. [Frozen SBNP](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py#L173) acts on latent rows and initial moments, with `min_dof=2*k_nn+d`; 32 rows in 8 dimensions exceed its 18-row cutoff for k=5. It does not test image identity or counts. The [shared trainer](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/training.py) adds its recipe-owned 0.01 contribution to generator/prior training. Output many-to-one mappings are not excluded by latent repulsion.
5. [Frozen optimizer](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/optim/dualnorm.py#L548) applies the quintic accelerator only above 32 channels. These host convolutions have at most 24 channels and retain the native small-kernel path. The failure is not evidence against the polynomial's exercised wide-channel path; the complete source/recipe remains its own numerical cohort.

The immediate measured cause of gate failure is too few clean images, especially among bright-assigned rows. A possible mechanism is a generator/critic correction field that preserves the two broad populations while failing to land precisely on either template. Generator sigmoid saturation, background leakage, geometry errors and critic gradients remain unmeasured alternatives. Exact assignment balance and late clean coverage make an endpoint population-collapse explanation inconsistent with this selected evidence.

## Existing attempts and controls

The [older BISPF readout](../../bispf/README.md) has the one-mode/HQ 0.5 endpoint, while the selected [quintic readout](../../bispf-optimized/README.md) preserves the same task FAIL with different numbers. The [disposition](../../test-dispositions-20261015/README.md#img_intensity2) records separate R1/R2, GAN v3 and kinetic passes with regressions on counted controls. The historical wider-network positive is `img_intensity2_residual16`, a different architecture task. BCAP convolution's four-check near pass is separate recipe evidence. None permits relabeling this result or selecting a brightness-specific recipe outside the global comparison.

Actual-training illustration for **this same attempt and saved output tensor**: [GIF](../../bispf-optimized/quality-media/img_intensity2.gif), [source/media receipt](../../bispf-optimized/quality-media/img_intensity2.json). The audit verifies the GIF hash and original `observed-images.pt` hash binding. No footage is invented, no images are regenerated, and no training is added. Metrics above determine the conclusion.

## Bounded next diagnostic and recommendation

Recover the exact archived `observed-images.pt` identified in the audit/media receipt. For the **five terminal checks, 160 images and zero optimizer updates**, compute nearest-target RMSE, mean/intensity variance inside the central patch, squared error outside it, and per-level clean counts. Reproduce [16,16] nearest and [11,8] clean at update 600 first. Brightness-error hypothesis: for at least 10 of the 13 final misses, squared error of the spatially constant patch-mean offset exceeds half the total squared error (using the exact orthogonal decomposition into patch mean, patch residual and background). Falsifier: fewer than 10 misses have that dominance; investigate spatial/background error instead. Preserve the original RMSE gate in either outcome.

The exact trained G/D state is additionally needed to distinguish sigmoid sensitivity from critic direction through the public `GANTrainer`. The saved provenance checkpoint is explicitly `purpose=provenance_only`, `prerequisite_eligible=false`; no restart or continuation is authorized by it. Missing tensors/state are evidence gaps, not reasons to rerun unchanged training. A new causal recipe experiment would require Forge registration, a finite budget, fixed task/protocol, one global recipe and counted controls including `img_stripes2`.

Keep the selected global BISPF recipe and its failures visible. This PR changes no recipe, gates, qualification, held status, view, leaderboard or default. It adds no paid training, continuation or seed experiment. Pixel/gradient causes remain open until the exact required artifacts are recovered.

## Reproduce this investigation

From the repository root, with the archived full readout available:

```sh
python reports/forge/leading-failures-20261022/img_intensity2/audit.py \
  --readout /mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json \
  --output /tmp/img_intensity2-audit-result.json
```

The script is standard-library-only and executes **zero** trainer steps, forwards or random draws. It verifies original archive bytes, selected leader and task identity, all 24 aggregate count/HQ/mode/TV calculations, original sustained result, budget/finite/RNG guards, endpoint projection, frozen source hashes, and actual-training media binding. It exits nonzero for a changed input. It does not regrade qualification or recompute missing image pixels. [Compact result](audit-result.json) retains selected milestones rather than copying bulk logs/per-update streams.

Validation: exact archived-observation audit **PASS**, original scientific **FAIL** unchanged; byte-corruption negative control rejects the altered archive; local report links and `git diff --check` pass. Console audit output can be redirected to any local `.log` and tailed; raw logs stay under `/mnt/ml7tb/ParticleGAN-forge/leading-failures-20261022/image-audits/`.
