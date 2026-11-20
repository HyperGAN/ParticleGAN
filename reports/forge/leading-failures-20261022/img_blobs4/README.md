# BISPF investigation: Four corners: underallocation and insufficient clean outputs

BISPF reaches three qualifying corners and 26/32 clean images. The fourth corner has only two nearest assignments and one clean image, so both coverage and quality fail. This is an investigation of the selected global recipe `bispf-optimized-quintic-v1`, not a new training comparison. The original **FAIL** is retained. The [current technique inventory](../../technique-inventory.md) remains the only goal leaderboard.

## Frozen evidence and protocol

- Candidate revision: `8d941696e5e05607c763bccefbcbe34694996370bb6f1b03d70f514be6273def`; attempt `9fd0c3e51af84b8895095f51765d84e4`.
- Executed commit: `d9d72d53761c898bce20102e1995fc3d899c1447`; scientific digest `c82ed2dc712e4c9ebda80678e453635bd8a16dc74fae1450234ed368aa956213`.
- Canonical original result hash: `c9762bd365a4c128e4cdc7406519ac66bd838ad70e7aed38c39d9f2ce4c0ff4a`. [Published compact receipt](../../technique-receipts/9fd0c3e51af84b8895095f51765d84e4.json) is a display/provenance summary, not a replacement qualification envelope.
- Original full readout: SHA-256 `50f249158118686904c8d01bf817fcf802696acbe84f0c56567cbc3cab3dedfe`, 4,926,714 bytes, retained at `/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json`. The [published archive pointer](../../bispf-optimized/quality.json) retains its original `/home/martyn/dev/...` identity. The audit verifies the exact bytes at the mounted location.
- Task contract: `be03bd0b19ad12740cd27187d63d49d16e112632d09b30e0ca8ae317ddc3715b`. [Current task declaration](../../../../configs/forge/tasks/img_blobs4.json) matches the recorded host/measurement contract used here.
- **Seed 0**, public `deterministic_orthogonal_named_parameters_v1` initialization, width-12 transposed-convolution generator (5,173 parameters), critic (2,833 parameters), 8-D latent, 32 learned uniform particle-cloud rows, sigma 0, no standardization. This task's explicit cloud exception applies; the inventory's generic MoG binding is not the image task prior.
- Uniform noisy-template target law: 8×8 grayscale templates with clipped Gaussian pixel noise std 0.01 during training. Evaluation enumerates all 32 learned rows, clean/live weights, no output noise, exactly 24 checks at updates 25,50,…,600.
- 600 updates each for G, D and prior; finite outputs, penalty applied 600 times, zero unintended RNG deviations. Constructor/data/prior/noise/evaluation streams have recorded initial bindings and terminal state hashes in the original receipt. The [audit result](audit-result.json) preserves initial model hashes and named-stream checkpoint hashes.
- Actual recipe: non-saturating GAN loss; BISPF critic penalty; full DualNorm, smoothing 0.001, momentum 0, quintic convolution rule; G lr 0.012, D multiplier 1.5, prior multiplier 2.5; learned prior with SBNP regularizer weight 0.01; no EMA scoring or added output noise. The task's old lr 0.0017, prior weight 0.05 and EMA 0.99 fields are inactive historical provenance, not active training settings. Exact resolved recipe is retained in [audit-result.json](audit-result.json).

## Numerical finding

The unchanged gate requires four modes, each with at least 4/32 clean images; at least 29/32 clean images overall; five consecutive terminal passing observations. Quality uses nearest-template **whole-image RMSE ≤ 0.10**. Count order: top-left (1,1), top-right (1,5), bottom-left (5,1), bottom-right (5,5).

| Metric | Selected endpoint | Bound | Original result |
| --- | ---: | ---: | --- |
| Qualifying modes | 3 | ≥ 4 | FAIL |
| Clean images | 26/32 (0.8125) | ≥ 29/32 | FAIL |
| Terminal passing checks | 0 | ≥ 5 | FAIL |
| Distribution TV (diagnostic) | 0.1875 | no task gate | diagnostic |
| Mean RMSE (diagnostic) | 0.047137164 | no mean-RMSE gate | diagnostic |

Endpoint nearest-template counts are **[8, 9, 2, 13]**, while clean counts are **[8, 6, 1, 11]**. The bottom-left corner is underallocated even before applying quality. Making every current assignment clean would still leave this corner below four; at least two assignments must change, and at least three additional clean images are needed overall.

The bottom-right corner first clears the four-clean population threshold at update 350. At 400 the top two corners join it, giving three qualifying modes. The bottom-left corner reaches just one clean image at 450 and never exceeds one thereafter. No observation reaches four modes, and maximum HQ is 26/32 (0.8125). This is neither a one-image near miss nor a four-check sustained near pass in this selected source.

**Historical distinction.** The older BISPF problem narrative says 28/32 clean (0.875). This selected quintic cohort records 26/32 (0.8125), which is three clean images below the 29/32 bound. Historical transport near passes and older BISPF numbers must not be substituted for this run.

## Mechanism examination

Examination is bound to the **executed commit**, not newer `develop` code. File hashes are recorded in [audit-result.json](audit-result.json).

1. [Frozen API image adapter](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/experiments/forge/adapters.py) trains through public `GANTrainer`, draws the noisy target batch once per update, and scores clean enumeration with `fixed_first_n=True`. This is exact finite-prior measurement, not 32 independent Monte Carlo samples. Missing modes here are not evaluation sampling noise.
2. [Frozen recipe loss dispatch](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/recipes.py#L673) only selects `SinkhornRelativisticLoss` when the recipe loss is `relativistic`. The actual recipe says `non_saturating`, so its [adversarial objective](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/gan_loss.py) is ordinary non-saturating loss. Sinkhorn still enters penalty residual/data-scale calculations. The family name does not imply that a balanced coupling supervises output mode populations in this run.
3. [Frozen BISPF penalty](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py#L1136) classifies N×1×8×8 as image tensors: default transport chords are bypassed and thermal scale is fixed to 1. The low-dimensional late harmonic-freeze branch does not apply to this image task. Image-specific endpoint penalty scaling and logit gauge remain active. Therefore a causal story based on image thermal freezing or a chord-bridge force is unsupported.
4. [Frozen SBNP](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/bispf.py#L173) acts on latent rows and initial moments, with `min_dof=2*k_nn+d`; 32 rows in 8 dimensions exceed its 18-row cutoff for k=5. It does not test image identity or counts. The [shared trainer](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/training.py) adds its recipe-owned 0.01 contribution to generator/prior training. Output many-to-one mappings are not excluded by latent repulsion.
5. [Frozen optimizer](https://github.com/255BITS/ParticleGAN/blob/d9d72d53761c898bce20102e1995fc3d899c1447/particlegan/optim/dualnorm.py#L548) applies the quintic accelerator only above 32 channels. These host convolutions have at most 24 channels and retain the native small-kernel path. The failure is not evidence against the polynomial's exercised wide-channel path; the complete source/recipe remains its own numerical cohort.

A 2×2 corner occupies only 4/64 pixels. The global image objective may give weak location-specific correction, or separated latent rows may still map to the same corner. These are hypotheses, not measured gradient causes. Two nearest assignments at the missing corner rule out a pure “all four populations are present, only sharpening is needed” endpoint account. They do not identify why the allocation became uneven.

## Existing attempts and controls

The [original BISPF readout](../../bispf/README.md) retains an earlier three-mode/HQ 0.875 result. The [October 15 dispositions](../../test-dispositions-20261015/README.md#img_blobs4) document separate R1/R2 and kinetic transport passes, and several kinetic histories with four final passing checks but not the required five. Those outcomes show that numerical feasibility and sustained acquisition are separate questions; their recipe and prior identities do not establish a BISPF repair. The convolution-support report belongs to a different BCAP recipe and is not a BISPF trial.

Actual-training illustration for **this same attempt and saved output tensor**: [GIF](../../bispf-optimized/quality-media/img_blobs4.gif), [source/media receipt](../../bispf-optimized/quality-media/img_blobs4.json). The audit verifies the GIF hash and original `observed-images.pt` hash binding. No footage is invented, no images are regenerated, and no training is added. Metrics above determine the conclusion.

## Bounded next diagnostic and recommendation

Recover the exact saved `observed-images.pt` in the audit/media receipt, then use **zero optimizer updates** on the five terminal checks (160 images) to decompose squared error into the target 2×2 patch, its complementary background, and the other three corners. Report per-corner assignments and clean populations, row transitions and each error component. Exact reproduction of [8,9,2,13] nearest and [8,6,1,11] clean is a mandatory precondition. The location/background hypothesis predicts larger error outside the target patch for missed bottom-left rows; falsify it if their errors are confined to correct-patch intensity with negligible off-patch error. A result may show mixed causes; report them rather than choosing one.

A later critic/Jacobian discrimination requires a complete compatible trained public `GANTrainer` state. The recorded checkpoint is `purpose=provenance_only`, `prerequisite_eligible=false`, and supplies no continuation authority. Missing arrays or model state block the corresponding analysis, not authorize unchanged retraining. Any training repair must retain width 12, the target law, initializer/streams, 600 updates, clean enumeration, all gates and the global recipe across tasks; test counted `img_stripes2` as a regression control.

Keep the selected global BISPF recipe and its failures visible. This PR changes no recipe, gates, qualification, held status, view, leaderboard or default. It adds no paid training, continuation or seed experiment. Pixel/gradient causes remain open until the exact required artifacts are recovered.

## Reproduce this investigation

From the repository root, with the archived full readout available:

```sh
python reports/forge/leading-failures-20261022/img_blobs4/audit.py \
  --readout /mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/ordinary-quality-full-readout.json \
  --output /tmp/img_blobs4-audit-result.json
```

The script is standard-library-only and executes **zero** trainer steps, forwards or random draws. It verifies original archive bytes, selected leader and task identity, all 24 aggregate count/HQ/mode/TV calculations, original sustained result, budget/finite/RNG guards, endpoint projection, frozen source hashes, and actual-training media binding. It exits nonzero for a changed input. It does not regrade qualification or recompute missing image pixels. [Compact result](audit-result.json) retains selected milestones rather than copying bulk logs/per-update streams.

Validation: exact archived-observation audit **PASS**, original scientific **FAIL** unchanged; byte-corruption negative control rejects the altered archive; local report links and `git diff --check` pass. Console audit output can be redirected to any local `.log` and tailed; raw logs stay under `/mnt/ml7tb/ParticleGAN-forge/leading-failures-20261022/image-audits/`.
