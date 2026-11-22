# Follow-up findings on open toy tests

This report documents test defects and evidence gaps. The separate merged-config
repair workstream owns production repairs; none are included here. Sources are
pinned in [pull_requests.json](pull_requests.json), and exact current results are
in [catalog.json](catalog.json).

## Implemented follow-up

The [latest sorted ledger](IMPROVEMENTS.md) tracks all 109 entries and the
definition/control improvements for all 64 original low-rated entries.
[Per-failure diagnoses](FAILURE_DIAGNOSIS.md) distinguish failed metric bars,
proved structural defects, sampling-law mismatch and unresolved optimizer
causes. [New image](IMAGE_QUALITY_V2.md) and [non-image](NON_IMAGE_QUALITY.md)
gates have their own identities and preserve original verdicts.

The [retention review](CANONICAL_SELECTION.md) removes 12 duplicate standalone
problem entries from the default canonical view while retaining their named
controls and scientific reasons. Every retained case states its purpose.
Distinct conditional exposure, widths, geometry, temporal laws and negative
controls stay separate. No checked-in problem definition was found wholly
useless; unmerged proposal sources cannot be described as deleted from develop.

One actual repeated frozen-reference execution is removed: resolved
`img_residual_bars4` shares the already-observed residual `img_bars4` result.
The alias keeps the original observed source/spec/artifact paths and declares
no independent training or qualification evidence. Source/runtime/recipe,
ordered-target, budget, gate or cadence changes restore separate execution.
The ordinary shipped transpose/residual architecture tests remain intact.

The PR227 repair is now committed on `codex/toy-pr227-signed-use` at
`0b196047`: all three paths require a positive code-removal loss, and each of
four harmful judges rejects. Actual retained endpoint assertions still pass.
The recovery verifier explicitly preserves original training-source identity
while admitting the signed evaluation change. No unchanged campaign was
retrained for this assertion repair. The [merge readout](merge_readiness/README.md)
binds the patch, eligible local diagnostics, held proposals and prospective
integration. Publication/remote merges remain blocked by GitHub access.

Fresh [bounded scalar controls](source_families/README.md),
[routed/ring fixtures](SOURCE_FAMILY_TRAINING.md) and [source demos](SOURCE_DEMOS.md)
now include real intermediate GIFs. Native accuracy, replay fidelity, ring
acquisition and Gaussian fidelity failures remain visible; original Gaussian
and routed-support budget caps remain incomplete. None triggers a production
configuration repair in this workstream.

The remaining [conditional](conditional_sources/README.md),
[route](route_sources/README.md) and [paired](paired_sources/README.md) readouts
complete the 17-entry source-attempt denominator. Affine/swirl validation
passes at 6,000 updates; both sparse and transition laws hit their caps and
remain incomplete. Denoising and trajectory sources stop at their original
CUDA prerequisite, with no training GIF or inferred convergence.

## PR226/227 supplemental findings

Both proposals execute and have useful bounded scientific claims; the
[supplemental review](PR226_PR227.md) supplies their missing training GIFs.
PR227's long assertion, intervention runner and recovered scorer all use
`abs(zero_code_minus_live) > 1e-6`. A synthetic control with ablated scores
0.01 **better** than live scores still passes the actual tensor-based regression.
For a useful-particle claim, all three paths should require a positive signed
effect under every mandatory judge and reject a negative-direction control.
The measured saved run has positive effects, so its result remains supported.

The original PR227 scorer failed after complete training when loading the
unresolved step-zero state into a resolved policy. Its fresh-policy repair is
already present at the reviewed head. Original errors/artifacts remain preserved;
independent re-scoring validates the repaired evidence without retraining.
Neither PR's relative improvement supplies an absolute output-accuracy gate
or validates its large image application. No production config repair follows.

## Execution blockers

| Proposal | Reproduced failure on develop | Consequence |
|---|---|---|
| [PR22 circle](https://github.com/255BITS/ParticleGAN/pull/22) | `ImportError: cannot import name 'edit_cap' from 'lib.gym_particle_finetune'` | No current training starts. Historical radius-hold scores/PNG do not fill the missing training GIF. |
| [PR153 sprite](https://github.com/255BITS/ParticleGAN/pull/153) | `AttributeError: 'Recipe' object has no attribute 'make_gradient_penalty'` | Data/model setup succeeds; training does not start. Its published dream GIF is endpoint playback, not convergence over updates. |
| [PR45](https://github.com/255BITS/ParticleGAN/pull/45), [47](https://github.com/255BITS/ParticleGAN/pull/47), [48](https://github.com/255BITS/ParticleGAN/pull/48), [49](https://github.com/255BITS/ParticleGAN/pull/49), [50](https://github.com/255BITS/ParticleGAN/pull/50), [51](https://github.com/255BITS/ParticleGAN/pull/51), [52](https://github.com/255BITS/ParticleGAN/pull/52), [53](https://github.com/255BITS/ParticleGAN/pull/53), [54](https://github.com/255BITS/ParticleGAN/pull/54), [55](https://github.com/255BITS/ParticleGAN/pull/55), [56](https://github.com/255BITS/ParticleGAN/pull/56), [57](https://github.com/255BITS/ParticleGAN/pull/57), [152](https://github.com/255BITS/ParticleGAN/pull/152) | `AttributeError: 'Recipe' object has no attribute 'loss_type'` | Historical GAN-v3 fields no longer exist on the public recipe. The audit's explicit archived-factory adapter reproduces all 13 FAIL/PASS contrasts, with full parameter identities. It provides no current KA2/Atlas pass. |

For circle and sprite, the next implementation should bind the current optimizer,
critic penalty, EMA/served law and complete budget explicitly. Preserve their
independent analytic/held-out evaluators. Replacing a removed API by a similarly
named method without binding its formulation would not reproduce the old arm.

Historical context, **not current training evidence**:
[circle final validation rollout](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/pr22-historical-rollout.png),
[sprite final dream rollout](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/pr153-historical-rollout.gif).

## Historical counterexample contrasts that no longer reproduce

| Proposal | Historical claim | Current full-budget live contrast | Interpretation |
|---|---|---|---|
| [PR76 b/d](https://github.com/255BITS/ParticleGAN/pull/76) | transpose12 FAIL / residual16 PASS | **PASS / PASS** | Preserve as a solvable glyph-template regression, but remove its current “break” status. |
| [PR70 corner ramp](https://github.com/255BITS/ParticleGAN/pull/70) | FAIL / PASS | **PASS / PASS** | Same: useful bounded template test, stale failing-baseline claim. |
| [PR154 chirp](https://github.com/255BITS/ParticleGAN/pull/154) | FAIL / PASS | **FAIL / FAIL** | Proposed positive control is no longer demonstrated at the declared settings. Do not call the new host calibrated/solvable from this replay. |
| [PR63 mask-inpaint](https://github.com/255BITS/ParticleGAN/pull/63) | FAIL / PASS | **FAIL / FAIL** | Positive-control gap plus the conditional-task mismatch described below. |

All four original-arm/control GIFs show the actual failure or success. The other
39 of 43 proposals retain FAIL/PASS. This review does not tune any arm to restore
its desired label or rewrite its historical receipt. A control failing is a
solvability-evidence gap, not proof that the underlying target law is ill-defined.

## Claims the evaluators cannot verify

The image proposals train an **unconditional** generator of 8×8 grayscale
templates from a finite particle table. For
[PR63 inpainting](https://github.com/255BITS/ParticleGAN/pull/63),
[PR65 colorization](https://github.com/255BITS/ParticleGAN/pull/65) and
[PR61 sparse observations](https://github.com/255BITS/ParticleGAN/pull/61),
the model receives no observed image/mask/source-domain query. These are
well-defined unconditional samplers, but they cannot test conditional completion,
color assignment or an input-output translation relationship. Rename the test's
claim to template fidelity/coverage, or introduce an explicit conditional task,
held-out queries and an appropriate conditional scorer as a new test revision.

Other application names—Braille, barcode, sonar, DSP, OCR, counting, chirality,
occlusion, focus—identify how the fixed templates were drawn. They do not add
independent decoding, physics, topology/counting or generalization metrics.
Each table row states the narrower property that is actually measured.

All 30 two-template image scorers accept exact **25/75** template mass: HQ=1,
two covered modes, distribution TV=0.25. The TV is reported but not gated.
This is consistent with their declared minimum-coverage threshold; a PASS
therefore establishes coarse coverage, not exact uniform density recovery.
Fresh scorer controls reject a collapsed template and the template mean.
These are numerical evaluator checks, not trained positive arms.

The polygon/wide-gap proposals usually change **both** critic architecture
and particle initialization in their positive control. The contrast proves
that the published combined setup is insufficient for that target; it cannot
attribute the failure solely to RBF lengths. Related layouts belong to one
scale/geometry stress family. PR58 duplicates the shipped intensity2 target
and must not receive an extra independent-family vote.

## Other PR evidence boundaries

- [PR222 PacGAN-8](https://github.com/255BITS/ParticleGAN/pull/222) proposes a
  method on existing grid100. Its PR records a **four-update smoke**, not a full
  7,000-update converged run. The fresh Atlas grid GIF demonstrates the problem,
  not this PacGAN/no-regularization arm.
- [PR216 runner migration](https://github.com/255BITS/ParticleGAN/pull/216)
  reorganizes 28 existing units. Runner consolidation does not create new
  problem definitions or independently establish their positive references.
  This review does not adopt its stale API/configuration cohort or implement
  its migration fixes.
- [PR224](https://github.com/255BITS/ParticleGAN/pull/224) reproduces a native
  SettleTest **unit** instability. Its critic and optimizer snapshot are
  constructed. It does not show that full Atlas training reaches that state
  or that Atlas's additional reopen guards fail.
- Atlas's three stationary noisy passes must remain separate from its three
  clean spread failures. Moving-target coverage recovery must remain separate
  from full density accuracy. MisGAN's projected 100-mode coverage must remain
  separate from eight-dimensional and conditional-posterior quality.

These findings justify a follow-up review PR and narrower test explanations;
they do not authorize replacing measured outcomes with desired success labels.
