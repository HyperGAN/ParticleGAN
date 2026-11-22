Fresh default-source runs add five real training GIFs for catalog IDs
`source-family-14` and `source-family-10`. Paired and moving routed fixtures pass
the separate correspondence and endpoint useful-code checks. Support is capped,
replay proves its eight-update software protocol while missing trained quality,
and the ring fails acquisition. These results do not change the original
[failure diagnosis](FAILURE_DIAGNOSIS.md) or any frozen catalog verdict.

| Fixture and exact source default | Original scientific status | Added audit result | What the evidence verifies | Actual training GIF |
| --- | --- | --- | --- | --- |
| [Paired](../../examples/e22_routed_paired.py): 160 updates, 16 rows, batch 32 | `NO_FROZEN_GATE`: source reports scalar RMSE | **PASS**: RMSE 0.002792; MSE/neutral MSE 0.000152; terminal passing suffix 7 | A frozen BF16 host with FP32 adapter learns context-specific paired edits; removing mixed generation codes worsens paired error and the fixed learned-game judge | [9 captured frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/source-14-paired.gif) |
| [Moving](../../examples/e22_routed_moving.py): pre-R1, 500 updates per period, two 30° turns, 1,500 total | `NO_FROZEN_GATE` | **PASS**: terminal RMSE 0.001387; relative MSE 0.0000374; final-period suffix 10 | The same continuing routed state adapts after each target rotation and uses its learned mixed code at the endpoint | [33 captured frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/source-14-moving.gif) |
| [Support](../../examples/e22_routed_support.py): full mode, 128 tokens/128 rows, zero feature-guard harm allowance, 1,200 updates | `NO_FROZEN_GATE` | **INCOMPLETE**: 180 s cap interrupted update 255 after 254 completed; latest complete capture 200 has RMSE 0.037645/relative MSE 0.042413 | Early paired transition improvement is observed; neither full-budget convergence nor endpoint useful code is established | [3 captured frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/source-14-support.gif) |
| [Replay](../../examples/e22_routed_replay.py): API initialization, token-local penalty, eight checkpointed updates | `NO_FROZEN_GATE`; software protocol completes | **FAIL** trained quality: relative MSE 5.470; no passing checks; full convergence untested | Activation recomputation replays private DV12 without advancing live streams; a short software smoke is insufficient to verify a trained paired model | [9 captured frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/source-14-replay.gif) |
| [Ring acquisition](../../benchmarks/toy100/continuous_probe.py): exact default config, constant rates, 1,200 updates | **FAIL**: terminal 1/8 modes, HQ 0.0720 | **FAIL** full Gaussian-law gate: mass TV 0.4912, max radial KS 1, no passing checks | Failure to acquire and sustain all eight ring modes under the declared source protocol; downstream hold/shift claims remain unknown | [24 captured frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/source-10-ring-acquisition.gif) |

The routed source oracles have no absolute trained pass threshold. The added
correspondence definition requires clean held-out paired MSE at most 10% of the
neutral frozen-host MSE. It uses all 180 held-out contexts, and every token for
the token fixtures. Five passing terminal observations are required. Endpoint
useful code additionally requires that code removal increase paired MSE and the
fixed frozen learned-game generator loss. These are new audit gates from
`definition_quality.py`, independent of historical grading and unrelated to
default adoption. Passing correspondence alone is not the full routed gate.

The code ablation retains the trained bank, encoder, router parameters and
host. It zeros mixed codes at the public generation boundary, rather than
zeroing the bank. In two-site models, downstream hidden states and routes can
respond to the earlier code removal. Held-out MSE rises by 0.023727 for paired
and 0.00016227 for moving; mean fixed-game loss rises by 0.17034 and 0.003819.
For replay, zero code **improves** paired MSE by 0.082282, while the learned game
judge favors live code: this short-run judge/quality disagreement is visible,
not treated as proof of an optimizer cause. Four matched private paired-noise
panels use the frozen served critic and clean generation without DV12; they
do not recreate the full stochastic training-game expectation. All three
saved endpoint predictions exactly match their captured arrays, and policy,
model, optimizer and RNG hashes remain unchanged during ablation.

Moving changes its target after completed updates 500 and 1000, without resetting
the model, optimizers, policy or streams. The GIF includes the instantaneous
target jumps and subsequent true observations. The first passing observation
after each turn is 50 updates later; this is a measured cadence bound, not a
claim about an unobserved update. This arm uses the documented pre-R1 defaults;
no matched R1 result is imported or inferred.

Support's finite cap leaves its policy inside an unfinished update. The source
correctly refuses an endpoint checkpoint with `policy checkpoints require a
completed update boundary`. Raw counters can include work from update 255 while
`completed_steps` remains254. The GIF ends at the last retained complete
capture, update 200. The missing artifact is a complete policy/weights/Adam/RNG
checkpoint at a completed boundary aligned with the held-out capture; a faithful
endpoint code intervention and continuation cannot be reconstructed from these
arrays alone. The two early passing correspondence observations do not satisfy
the five-check rule or the 1200-update budget. The cap was not extended.

The ring host actually uses a **12×4 uniform discrete ParticlePrior**, batch 128,
width96 three-layer generator and width96 Fourier-3 critic. Its source bridge
retains these frozen resources even though the input native configuration says
20,000 particles and batch 2048, and the intermediate recipe retains other
resource defaults. The clean law consists of 12 deterministic generator outputs;
the observed live law adds terminal isotropic output noise σ 0.029. The target has
eight equal-weight radius3 Gaussian modes with σ 0.07. This is a LegacyRecipe/GAN-v3
source execution, distinct from current native independent-row Atlas.

There is a proven clean-density/resource mismatch: assigning12 equally weighted
clean rows across all eight target components has minimum mass TV 1/6, attained
by four components with two rows and four with one. A single output-noise kernel
has covariance ratio (0.029/0.07)²=0.17163 relative to the target, below the added
0.5 covariance floor. These arithmetic/kernel witnesses explain why full-density
claims need a representable contract. They do not prove that every approximate
output-noisy gate is impossible, or that this mismatch caused the observed
optimizer path. Actual terminal samples independently fail the original
acquisition gates and the stronger law gates: covariance ratios range from 0 to
111.7, nearest-component mass TV is0.4912 and max radial KS is1.

The best ring quality observation at 1000 has HQ 0.9641 but only six of eight
modes; it is not a passing warm state. Later observations collapse to one
quality mode. The1200-update acquisition fails, so the uninterrupted 2400 hold
and 3600 adaptation run with a shift at 2400 plus matched frozen control were not
executed. The separate `warm_equilibrium_probe.run_warm_variants` source defaults
to a **scheduled** prefix and a Linux fork at 1000, with identity/cold final-state
parity and 200 per-update continuation checks. Its helper does not itself enforce
an all-eight passing-state prerequisite. No scheduled warm/cold variant has
been run or qualified by this constant-acquisition audit. The cause of the
observed optimizer collapse remains unresolved; the full terminal G/D/Adam
checkpoint is available for a bounded follow-up diagnostic.

Observer controls use the unchanged source updates. Two-update routed observer
on/off checks preserve exact model/Adam/average state, gradients, modes and
training streams. Plain and checkpointed replay updates also match exactly in
the separate short protocol test. Ring capture preserves its training-state
hash at all 24 boundaries; its complete diagnostic curve, final live/EMA,
optimizer accounting, noise receipt and rates match an unobserved direct source
default execution exactly. Held-out metrics never enter policy observations.
Every GIF frame is a real captured state; none is interpolated.

The [compact machine-readable join](source-family-training.json) uses key
`fixtures`, keeps each routed subfixture distinct, binds source/runtime/config
and sampling identity, and preserves the original diagnosis digest. Its
`source-family-training-compact-v2` publication schema keeps only labeled
final/best observations, observation counts, terminal-window and budget status,
compact phase summaries and actual GIF step indices. Full metric curves remain
in the SHA-bound `curve_archive`; captured tensors remain in
`observation_archive`. Shared source/runtime bindings use top-level identity
tables referenced by each fixture. Original external receipts are unchanged,
and the generator rejects inline temporal metric streams. All raw
updates, stdout, failures, observation tensors and checkpoints remain outside
Git under `/ml2/hypergan/toy-audit-artifacts-20261001/source-family-training-v1`.
CPU uses one thread. Routed fixture work has180 s caps and ring work a120 s cap;
support's184.74s reported total includes artifact/error cleanup after the 180 s
interrupt. These shared-machine wall times are not isolated throughput claims.
No production defaults, configs or libraries were changed, and no seed study
or failed-acquisition extension was performed.

Reproduction uses the new audit runner name, which avoids a filename collision
with a separate agent's sign/landing audit. The compact receipt maps the old
executed name to the byte-identical published runner; original observer and
ablation sources are archived beside the raw evidence.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_CBWR=AVX2 \
  python -u -m benchmarks.toy_audit.source_routed_ring_training paired \
  --artifacts /tmp/new-source-audit/paired --output /tmp/new-source-audit/paired-receipt.json
python -m benchmarks.toy_audit.source_family_ablation \
  --receipt /tmp/new-source-audit/paired-receipt.json \
  --quality-module benchmarks/toy_audit/definition_quality.py --output /tmp/new-source-audit/paired-ablation.json
MPLCONFIGDIR=/tmp/toy-audit-mpl python -m benchmarks.toy_audit.source_family_media \
  --receipt /tmp/new-source-audit/paired-receipt.json --output /tmp/new-source-audit/paired.gif \
  --media-receipt /tmp/new-source-audit/paired-media.json
```

Use each declared fixture selector separately. The runner rejects an existing
artifact directory to prevent stale captures from masquerading as a new run.
Additional execution requires its own purpose and retained receipt; this report
does not authorize tuning or spending after a failed scientific prerequisite.
