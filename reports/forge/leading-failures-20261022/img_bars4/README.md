# BISPF investigation: Four bars: persistent population shortfall

BISPF produces 31/32 clean images, but only two bars receive the four clean particles required per mode. The missing modes were never acquired at any of the 24 recorded checks. This is an investigation of the selected global recipe `bispf-optimized-quintic-v1`, not a new training comparison. The original **FAIL** is retained. The [current technique inventory](../../technique-inventory.md) remains the only goal leaderboard.

## Frozen evidence and protocol

- Candidate revision: `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`; attempt `048820009b3b44108cad4bc5fee0854b`.
- Executed commit: `d9d72d53761c898bce20102e1995fc3d899c1447`; scientific digest `c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213`.
- Canonical original result hash: `9345c8134863f8d830e7ca918e01ecb3d1e5a26b705a32a9683de7d942023b44`. [Published compact receipt](../../technique-receipts/048820009b3b44108cad4bc5fee0854b.json) is a display/provenance summary, not a replacement qualification envelope.
- Original full readout: SHA-256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`, 4,926,714 bytes, retained at `/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`. The [published archive pointer](../../bispf-optimized/quality.json) retains its original `/home/martyn/dev/...` identity. The audit verifies the exact bytes at the mounted location.
- Task contract: `e2a26e57ac94a7473db2b367d00df492ff027e4036ef946f5c5e2dc37fd65296`. [Current task declaration](../../../../configs/forge/tasks/img_bars4.json) matches the recorded host/measurement contract used here.
- **Seed 0**, public `deterministic_orthogonal_named_parameters_v1` initialization, width-12 transposed-convolution generator (5,173 parameters), critic (2,833 parameters), 8-D latent, 32 learned uniform particle-cloud rows, sigma 0, no standardization. This task's explicit cloud exception applies; the inventory's generic MoG binding is not the image task prior.
- Uniform noisy-template target law: 8×8 grayscale templates with clipped Gaussian pixel noise std 0.01 during training. Evaluation enumerates all 32 learned rows, clean/live weights, no output noise, exactly 24 checks at updates 25,50,…,600.
- 600 updates each for G, D and prior; finite outputs, penalty applied 600 times, zero unintended RNG deviations. Constructor/data/prior/noise/evaluation streams have recorded initial bindings and terminal state hashes in the original receipt. The [audit result](audit-result.json) preserves initial model hashes and named-stream checkpoint hashes.
- Actual recipe: non-saturating GAN loss; BISPF critic penalty; full DualNorm, smoothing 0.001, momentum 0, quintic convolution rule; G lr 0.012, D multiplier 1.5, prior multiplier 2.5; learned prior with SBNP regularizer weight 0.01; no EMA scoring or added output noise. The task's old lr 0.0017, prior weight 0.05 and EMA 0.99 fields are inactive historical provenance, not active training settings. Exact resolved recipe is retained in [audit-result.json](audit-result.json).

## Numerical finding

The unchanged gate requires four modes, each with at least 4/32 clean images; at least 29/32 clean images overall; five consecutive terminal passing observations. Quality uses nearest-template **whole-image RMSE ≤ 0.10**. Count order: vertical bar at column 1, vertical bar at column 5, horizontal bar at row 1, horizontal bar at row 5.

| Metric | Selected endpoint | Bound | Original result |
| --- | ---: | ---: | --- |
| Qualifying modes | 2 | ≥ 4 | FAIL |
| Clean images | 31/32 (0.96875) | ≥ 29/32 | PASS |
| Terminal passing checks | 0 | ≥ 5 | FAIL |
| Distribution TV (diagnostic) | 0.40625 | no task gate | diagnostic |
| Mean RMSE (diagnostic) | 0.012657694 | no mean-RMSE gate | diagnostic |

The endpoint nearest-template counts are **[7, 21, 2, 2]**, while clean counts are **[7, 21, 2, 1]**. Both horizontal bars have positive mass; “only two bars exist” would overstate the evidence. They are below the mode population threshold. Sharpening the one dirty image alone cannot produce a pass: even all nearest assignments made clean would still leave two modes below four. At least four particle assignments must change to bring both deficient nearest populations to four.

The second vertical bar first clears its clean population threshold at update 300. The first vertical bar clears it at 375. Both horizontal populations remain below four through update 600. The overall quality bound first passes at 500, and stays above it at the five terminal checks, while coverage never passes. This separates an endpoint population problem from merely missing the sustained suffix after a late full acquisition. The observations are 25 updates apart; transient unrecorded acquisition cannot be excluded.

**Historical distinction.** The earlier BISPF narrative in `problems.md`, the October 15 dispositions and the original BISPF readout says HQ 1.0. This selected quintic source instead records 0.96875. Both retain FAIL; the archived historical values are left unchanged.

## Mechanism examination

Examination is bound to the **executed commit**, not newer `develop` code. File hashes are recorded in [audit-result.json](audit-result.json).

1. [Frozen API image adapter](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/experiments/forge/adapters.py) trains through public `GANTrainer`, draws the noisy target batch once per update, and scores clean enumeration with `fixed_first_n=True`. This is exact finite-prior measurement, not 32 independent Monte Carlo samples. Missing modes here are not evaluation sampling noise.
2. [Frozen recipe loss dispatch](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/recipes.py#L673) only selects `SinkhornRelativisticLoss` when the recipe loss is `relativistic`. The actual recipe says `non_saturating`, so its [adversarial objective](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/gan_loss.py) is ordinary non-saturating loss. Sinkhorn still enters penalty residual/data-scale calculations. The family name does not imply that a balanced coupling supervises output mode populations in this run.
3. [Frozen BISPF penalty](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py#L1136) classifies N×1×8×8 as image tensors: default transport chords are bypassed and thermal scale is fixed to 1. The low-dimensional late harmonic-freeze branch does not apply to this image task. Image-specific endpoint penalty scaling and logit gauge remain active. Therefore a causal story based on image thermal freezing or a chord-bridge force is unsupported.
4. [Frozen SBNP](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py#L173) acts on latent rows and initial moments, with `min_dof=2*k_nn+d`; 32 rows in 8 dimensions exceed its 18-row cutoff for k=5. It does not test image identity or counts. The [shared trainer](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/training.py) adds its recipe-owned 0.01 contribution to generator/prior training. Output many-to-one mappings are not excluded by latent repulsion.
5. [Frozen optimizer](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/optim/dualnorm.py#L548) applies the quintic accelerator only above 32 channels. These host convolutions have at most 24 channels and retain the native small-kernel path. The failure is not evidence against the polynomial's exercised wide-channel path; the complete source/recipe remains its own numerical cohort.

A candidate explanation is that the public generator maps several separated latent rows to the same already-good bar, leaving too little useful generator/critic signal toward the two underpopulated bars. Endpoint counts establish the allocation failure; they do not separate critic direction, generator Jacobian, latent-prior movement or finite-budget acquisition. Latent repulsion is not an output-mode population guarantee.

## Existing attempts and controls

The [original BISPF report](../../bispf/README.md) records a separate older endpoint with the same two-mode failure. The [optimized readout](../../bispf-optimized/README.md) records that the width-64 quintic bars diagnostic passes all four modes and nine terminal checks. Width 64 changes the task architecture; it demonstrates a separate positive, not a width-12 repair or proof that width-12 cannot represent the target. R1/R2 and GAN v3 pass this task in separate recipes but regress counted controls; the [disposition](../../test-dispositions-20261015/README.md#img_bars4) preserves that conflict. Frobenius optimization also fails the separate wide diagnostic, so another normalization substitution is not already justified.

Actual-training illustration for **this same attempt and saved output tensor**: [GIF](../../bispf-optimized/quality-media/img_bars4.gif), [source/media receipt](../../bispf-optimized/quality-media/img_bars4.json). The audit verifies the GIF hash and original `observed-images.pt` hash binding. No footage is invented, no images are regenerated, and no training is added. Metrics above determine the conclusion.

## Bounded next diagnostic and recommendation

First recover the exact saved `observed-images.pt` identified in the audit and media receipt. With **zero training updates**, examine all 32 images at the five terminal checks (160 rows total): report clean/nearest populations, per-row nearest-label transitions and RMSE margins to each missing bar. This is an observation diagnostic, not a new toy training test. Predict a shortage of at least two nearest assignments in each horizontal mode at the endpoint. Falsifier: exact pixels fail to reproduce [7,21,2,2] and clean [7,21,2,1]; stop and repair evidence linkage before any trainer experiment.

To distinguish causes afterward, recover a complete compatible trained public `GANTrainer` state, then measure output-space critic correction and generator/prior Jacobian sensitivity toward the missing modes without optimizer steps. The available provenance checkpoint is explicitly `purpose=provenance_only`, `prerequisite_eligible=false`; it does not authorize a continuation. If complete trained state is unavailable, record that blocker. Do not rebuild it by rerunning the unchanged experiment. Any future causal training change needs a registered study, numerical predictions and budget, one global recipe, plus unchanged `img_stripes2` and counted-task controls.

Keep the selected global BISPF recipe and its failures visible. This PR changes no recipe, gates, qualification, held status, view, leaderboard or default. It adds no paid training, continuation or seed experiment. Pixel/gradient causes remain open until the exact required artifacts are recovered.

## Reproduce this investigation

From the repository root, with the archived full readout available:

```sh
python reports/forge/leading-failures-20261022/img_bars4/audit.py \
  --readout /mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json \
  --output /tmp/img_bars4-audit-result.json
```

The script is standard-library-only and executes **zero** trainer steps, forwards or random draws. It verifies original archive bytes, selected leader and task identity, all 24 aggregate count/HQ/mode/TV calculations, original sustained result, budget/finite/RNG guards, endpoint projection, frozen source hashes, and actual-training media binding. It exits nonzero for a changed input. It does not regrade qualification or recompute missing image pixels. [Compact result](audit-result.json) retains selected milestones rather than copying bulk logs/per-update streams.

Validation: exact archived-observation audit **PASS**, original scientific **FAIL** unchanged; byte-corruption negative control rejects the altered archive; local report links and `git diff --check` pass. Console audit output can be redirected to any local `.log` and tailed; raw logs stay under `/mnt/ml7tb/ParticleGAN-forge/leading-failures-20261022/image-audits/`.
