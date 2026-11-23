# BCAP optimizer, loss and retention search

All **96 BCAP-on configurations** have concluded, at **2026-10-09 04:51 MDT**. Four passed all **6/6 Tier 1** requirements. Three tied at **7/21 Tier 2**, below the frozen **10/21** target. The extra 24 produced no Tier 1 survivors. The whole-configuration PASS-count/content-hash objective selected **DualNorm with non-saturating loss**; the hash resolves the three-way tie. No candidate qualifies through Tier 2.

After this readout was published, the owner requested using the winner as the named public `bcap` preset. The separate [default-selection decision](DEFAULT_SELECTION.md) records that update; this search's original scientific outcomes and immutable archive are preserved.

The [current technique inventory](../technique-inventory.md) is the single goal leaderboard. [Summary](summary.json), [combined selection](combined-readout.json), [all original 72 results](readout.json), and [all added 24 results](overnight/readout.json) retain every configuration, setting, final metric, gate and attempt identity. Missing higher-tier results remain unmeasured, rather than numerical failures.

The [winner Tier 2 failure analysis](FAILURE_ANALYSIS.md) decomposes saved samples and restored checkpoint gradients, explains the conditional and image failures, and recommends focused next comparisons. It adds no training or random sampling and preserves the original gates and selection.

The subsequent [five-theory repair comparison](../bcap-physics/README.md) completes five separate candidate/control studies and links their PRs, metrics and actual-training GIFs. Output-motion control improves retention and native precision, and transport repairs allocation; no candidate adds a complete sustained pass.

The selected recipe uses G/E rate **0.012**, D rate **0.018**, and prior rate **0.030**, all constant; DualNorm smoothing **0.001**, zero momentum, convolution **per_offset**, BCAP coefficient **1**, cap **1**, and regularization **every update**. Its [complete resolved recipe](../../../configs/forge/configurations/bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36.json) is the authoritative global configuration across tasks. Forge v1's historical `bcap` preset resolves through Adam, so reproducing this configuration requires its explicit optimizer and loss overrides. Public package defaults are unchanged.

All four survivors used the same constant role rates, zero momentum and every-update regularization. The relativistic control with smoothing 1e-5 scored 6/21; smoothing 0.001 scored 7/21. Reducing the cap to 0.5 also scored 7/21, trading mode retention for intensity-image success. The selected non-saturating configuration trades mode retention for complete word retention, without improving the aggregate over the other two tied configurations. In the 36 otherwise matched cadence pairs, every-update regularization won the Tier 1 PASS count in 24 pairs, tied in nine, and lost in three. These are outcomes in this finite domain, rather than general optimizer/loss rankings.

| Optimizer | Configurations | Best Tier 1 | 6/6 survivors | Best Tier 2 among survivors |
| --- | ---: | ---: | ---: | ---: |
| ada_nsgda | 2 | 2/6 | 0 | not run |
| adam | 20 | 4/6 | 0 | not run |
| dualnorm | 64 | 6/6 | 4 | 7/21 |
| dualnorm_D_only | 2 | 3/6 | 0 | not run |
| nsgda_global | 2 | 3/6 | 0 | not run |
| nsgda_layer | 2 | 2/6 | 0 | not run |
| particle_rownorm_only | 2 | 3/6 | 0 | not run |
| sgda | 2 | 3/6 | 0 | not run |

The largest Tier 1 bottleneck was ring acquisition: only five configurations passed it, with 89 numerical failures and two incomplete attempts. Gaussian smoke passed for 56/96 and words for 48/96. Adam and the six other optimizer families produced no 6/6 survivor in the sampled settings; this does not exhaust their hyperparameter spaces. Hinge, least-squares and Wasserstein likewise produced no survivor in their eight configurations each.

For the selected recipe, word hold passes all **25 primary and independent confirmation pairs**, restoring its own earliest confirmed checkpoint at update **834** and completing **4,000 additional updates** through 4,834. Gaussian stability passes only **2/72 stationary checks** and **0/24 shifted hold checks**; its endpoint standard-deviation ratio is **0.662**, with CDF KS **0.321**. Stripes achieve a passing 12-check terminal suffix. Bars cover only 2/4 modes, blobs 2/4, and intensity quality ends at 0.688. Mode hold has a good endpoint but only a three-check passing suffix, below its five-check requirement, so its FAIL is preserved. Native grid, rotated and staggered holdout precision is **0.241, 0.256 and 0.302**, far below the required coverage.

| Selected recipe task | Recorded gate | Actual saved-training GIF |
| --- | --- | --- |
| gaussian1d_smoke | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/gaussian1d_smoke.gif) |
| two_pole | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/two_pole.gif) |
| unused_token_hold | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/unused_token_hold.gif) |
| ae_gan_hold | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/ae_gan_hold.gif) |
| ring16_acquisition | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/ring16_acquisition.gif) |
| five_word_joint_smoke | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/five_word_joint_smoke.gif) |
| clockfree_audit_measurement_v1 | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/clockfree_audit_measurement_v1.gif) |
| gaussian1d_stability | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/gaussian1d_stability.gif) |
| five_word_joint_hold | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/five_word_joint_hold.gif) |
| trajectory | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/trajectory.gif) |
| residual_student | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/residual_student.gif) |
| unipolar | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/unipolar.gif) |
| cover_leftover | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/cover_leftover.gif) |
| mid_scale_identity | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/mid_scale_identity.gif) |
| mode_hold | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/mode_hold.gif) |
| vector_two_broad | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/vector_two_broad.gif) |
| vector_unequal_mass | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/vector_unequal_mass.gif) |
| vector_unequal_width | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/vector_unequal_width.gif) |
| vector_anisotropic | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/vector_anisotropic.gif) |
| vector_overlap | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/vector_overlap.gif) |
| vector_spiral | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/vector_spiral.gif) |
| img_stripes2 | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/img_stripes2.gif) |
| img_bars4 | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/img_bars4.gif) |
| img_blobs4 | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/img_blobs4.gif) |
| img_intensity2 | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/img_intensity2.gif) |
| grid100 | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/grid100.gif) |
| rotated100 | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/rotated100.gif) |
| staggered100 | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-tier2-search/media/staggered100.gif) |

The [media receipt](media-receipt.json) and [index](media/index.json) bind these 28 GIFs to this one complete configuration. Rendering constructs no models, adds no training updates or sampling draws, and never substitutes a better-looking candidate. Image metric audits recompute the original retained samples; spiral media show its actual saved training measurements.

Forge completed **756 attempts** for **17.61 paid worker hours** across about **12.24 elapsed hours**, with no retries. **749 completed normally, three timed out and four errored**. The four errors came from the two SGDA configurations: three nonfinite-loss guards and one NaN JSON logging failure. Three word attempts hit the unchanged 900-second wall budget. Their gates remain INCOMPLETE; they are not rewritten as numerical FAIL. [Exception identities and original certificates](execution-exceptions.json) preserve the failures.

Both GPU supervisors used one device slot each, with no overlap. Their supervised intervals occupied **71.0% and 72.9%** of the full window. Median gaps were **31.1 and 29.7 seconds**; the final queue state files were **315.5 MB and 100.4 MB**. These intervals include startup and independent grading; gaps include source verification, launch, scheduling, campaign transition and bookkeeping. They do not isolate coordinator cost or measure CUDA kernel utilization. The [framework readout](framework-readout.json) retains the measurement scope and terminal-receipt hashes.

This test drive verifies frozen admission, both GPUs, complete-current-tier execution despite failed peers, higher-tier stopping, budget enforcement, immutable source/runtime/protocol bindings, exact receipt certification, and all-at-once evaluation. It exposed a NaN error-reporting weakness, large state/launch gaps, and report-only defects in word checkpoint aggregation and image/procedural rendering. The reporting adapter verifies the actual selected parent checkpoint, preserving the frozen word grader and task declarations; [ten corruption/missing-proof checks](publication-adapter-checks.json) reject invalid bindings. Original receipts and outcomes are unchanged.

Recommended next work: retain the selected configuration as the measured research reference; inspect its saved Gaussian covariance/critic-gradient trajectory and native per-mode moments before proposing a smaller, focused BCAP change. A broader repeat of these optimizer/loss settings has weak support. Separately profile queue serialization/source verification and normalize repeated request/source metadata before enlarging the next search. Leaderboard publication also took several minutes of receipt replay in both the frozen-source and current-declaration passes; index validated receipts and avoid repeated scans when improving reporting. Harden nonfinite failure receipts and calibrate slow word execution budgets in a separately declared revision; do not rerun or silently extend these unchanged experiments.

The [original study](STUDY.md) and [separate overnight study](overnight/STUDY.md) preserve their frozen declarations. Both used protocol seed 0, common scientific source/runtime, fixed task architectures/targets/priors/batch and sampling laws, budgets and evaluation cadence, one global trainer configuration per candidate, and complete-current-tier gates. The extension was frozen from archived motivation and runtime estimates, without interim scientific ranking; it changes none of the original 72 declarations. [Matched-condition audit](matched-conditions.json), [original compact receipt certificates](receipts.json), and [extension certificates](overnight/receipts.json) bind the results. Archived evidence retains its original source and initialization cohort. Tier 2 was used for tuning; independent confirmation and scientific calibration/default adoption remain unperformed/provisional. No Tier 3 run follows.

Reproduction and easy-to-tail logs:

```sh
python -u reports/forge/bcap-tier2-search/run.py \
  --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-tier2-search-v1 --gpus 0,1 \
  > runs/software/bcap-tier2-search/run.log 2>&1
python -u reports/forge/bcap-tier2-search/run_overnight.py \
  --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-overnight-search-v1 \
  --finish-policy complete_batch > runs/software/bcap-overnight-search/run.log 2>&1
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-overnight-search-v1/events.jsonl
```

These are the original launch commands, not permission to rerun unchanged completed trials. Bulk stdout, JSONL, checkpoints, tensors and queue state stay outside Git. [Archive receipt](archive.json) records the immutable data-drive bundle, byte verification and original restore roots. [archive.py](archive.py), [publish.py](publish.py) and [audit_framework.py](audit_framework.py) reproduce packaging and display audits after both completion receipts exist. Preparation passed 87 search-space/configuration tests and 34 convolution integration tests. Reporting validation and memory checks are recorded with the final PR.
