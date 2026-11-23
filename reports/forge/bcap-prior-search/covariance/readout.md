# Moment tether: one Tier 2 repair, one regression

The global `prior_reg=0.01` moment tether converts **`mode_hold` FAIL → PASS**
against its own complete BCAP leader control. It also converts **word hold
PASS → FAIL**. Both finish at 9/21 Tier 2 passes. Keep the feature opt-in; this
round provides a task repair, but no configuration that preserves every
recorded control pass. The finite search is complete, with no refinement,
seed variant, scientific retry, Tier 3 execution or default adoption.

The [complete report](README.md) lists every task's measurements, scores and
frozen targets, including blocked cells and missing diagnostics. The
[current leaderboard](../../technique-inventory.md) is maintained by the shared
goal coordinator. [All 70 completed-task training GIFs](media/README.md) were
exported from saved artifacts, with original grades and hashes intact.

| Global strength | Tier 1 | Tier 2 | Matched outcome |
| --- | --- | --- | --- |
| Leader control, weight 0 | 6/6 | 9/21 | Same-source reference |
| Moment tether 0.001 | 5/6 | Unmeasured: ring prerequisite failed | Nine control Tier 2 passes unresolved |
| Moment tether 0.01 | 6/6 | 9/21 | Mode hold repaired; word hold regressed |
| Moment tether 0.1 | 5/6 | Unmeasured: ring prerequisite failed | Nine control Tier 2 passes unresolved |

The common recipe is the complete `recipe-prior-refactor-bcap-pure-v1` leader:
non-saturating BCAP, direction blend, zero-momentum DualNorm, smoothing 0.001,
G/E learning rate 0.012, discriminator multiplier 1.5, prior multiplier 2.5,
constant rates, no EMA, transport or additional output noise. The only global
recipe delta is `prior_regularizer="moment_tether"` and positive `prior_reg`.
The initializer, task architectures, target/data laws, initial prior laws and
capacities, sampling, named RNGs, evaluation cadence, update budgets and gates
remain frozen. Protocol seed is 0; ordinary view revision is 9.

## The repair is sustained quality

Mode hold executes all 1,200 updates and 24 scheduled observations. Its final
passing suffix increases from **3 / ≥5** to **7 / ≥5**. At step 1,050, both
arms cover 8 / ≥8 modes, but high-quality fraction is **0.869384765625 / ≥0.9**
for the control and **0.998046875 / ≥0.9** for the tether. The candidate's
stable suffix begins at step 900 and is confirmed at 1,100. Endpoint quality
alone would conceal the control failure: both endpoints pass their numerical
mode and quality bounds.

[Original repair witnesses](repair-witness.json), reproduced by
[repair_witness.py](repair_witness.py), pin all four repair/regression
certificates, original result/adapter-receipt/request/log hashes, complete
observation-list hashes, suffix observation hashes and the discriminating
step's score/target. The source, runtime, compute profile, resolved recipe,
initializer tensor receipt and actual prior policy are checked. The
[candidate's actual training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-prior-search/covariance/media/configuration-ec49f810ceed/mode_hold.gif)
and [control GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-prior-search/covariance/media/control/mode_hold.gif) illustrate the measured task.

## Word hold still loses mandatory checks

Word hold requires every primary and independent confirmation check to pass.
The control passes **25/25** in both cohorts; the tether passes **23/25**.
At global step 5,167 it loses a mode and mass balance; at 6,167 its inverse
reconstruction probability falls below the bound.

| Mandatory measurement | Worst tether score / target | Original global step |
| --- | --- | --- |
| Mode count, primary and confirmation | 4 / ==5 | 5,167 |
| Mass TV, primary | 0.203515625 / ≤0.1 | 5,167 |
| Mass TV, independent confirmation | 0.2046875 / ≤0.1 | 5,167 |
| Minimum reconstruction token probability, both cohorts | 0.7967675924301147 / ≥0.9 | 6,167 |

Each arm restores its **own** certified acquisition prefix: control 834,
candidate 4,167, followed by the same 4,000-update hold window. This is the
declared candidate pipeline comparison; equal trained starting weights or
equal total consumed prefixes are not claimed. Restores preserve history,
optimizer/RNG state and the original fixed moment buffers. Both word-hold
endpoints pass; endpoint recovery does not erase mandatory failures. The
complete report and repair witness retain original observation hashes for
every worst bound, separately for primary and confirmation cohorts.

## Initial global moments are an incomplete proxy

[Saved moment diagnostics](moment-diagnostics.md), with
[hash-bound numerical receipts](moment-diagnostics.json), measure final full
live location mean and covariance against checkpointed original references
for all 31 measured positive tasks. They require no model execution, sampling
or optimizer updates and confer no gate credit. The matched control borrows
the same reference only after equal public-initializer receipts and initial
prior law/capacity are checked. Own continuation prefixes remain explicit.

For word hold, the combined discrepancy falls from 0.4674966931 to
0.0001051523 despite the mandatory output failure. For the repaired mode hold,
it falls modestly from 0.1346762329 to 0.1190278828. These observations establish
that the objective can restrain latent drift; they do not establish a causal
explanation for the output changes.

All three 100-Gaussian tasks remain FAIL. The following diagnostics use the
same saved independent 100,000-draw live holdout; they add no draws or gate
credit. Full score/target packets and original terminal gates are in the
complete report.

| Task | Control → tether precision / ≥0.97 | Control → tether modes / ==100 | Control → tether mass TV / ≤0.1 |
| --- | --- | --- | --- |
| grid100 | 0.24072 → 0.29934 | 8 → 17 | 0.14718 → 0.15976 |
| rotated100 | 0.25552 → 0.22569 | 14 → 10 | 0.14325 → 0.13697 |
| staggered100 | 0.30168 → 0.34370 | 19 → 17 | 0.12277 → 0.12345 |

Full-location moment discrepancy is 0.0150197633 versus control 0.0184639022
on grid, 0.0059202560 versus 0.0087848268 on rotated, and 0.0179421008 versus
0.0117850220 on staggered. Thus the tether does not consistently improve
even the full-table diagnostic under the native selected-row regime, and
improved global moments do not guarantee coverage or density accuracy.

Native tables above 1,024 rows use the trainer's declared selected-row
objective; full-table diagnostics differ from that noisy objective. DualNorm
scales each unique sampled row's combined gradient `g` by
`1/sqrt(||g||² + 0.001²)`, with prior base learning rate 0.03. The coefficient
acts before nonlinear smoothed saturation, not as a displacement bound.
Unsampled rows stay fixed despite dense auxiliary gradients. The non-monotonic
ring results at 0.001/0.01/0.1 reinforce the need to evaluate global strengths
empirically rather than infer proportional behavioral changes.

## Provenance and execution limitations

The frozen executed commit is `18e2aa1018f25de8c447a59287025df6f6aded3e`;
scientific source digest is
`c63305181b75bbfae337143b6836397aa998cadbf172581859031a01c90892c2`.
[Saved-state audit](audit.json), reproduced by [audit.py](audit.py), passes
all 70 measured saved states and 39 matched pairs, validates originals
symmetrically, and checks actual recipes, initial tensor receipts, prior
policies, budgets, grades, consumed named non-evaluation streams, fixed
references and own Gaussian/word restores. Two-pole has an explicitly declared
source-bound fixture without a tensor-hash receipt; that absence remains
explicit. Optional clock audits use their smaller diagnostic fixture and do
not prove feature invariance in the research models.

The [shared infrastructure receipt](../infrastructure.json) preserves the
original cancelled non-MPS word-control attempt and charged cost. Confirmed
CUDA-context contention prompted an infrastructure-only retry under unchanged
recipe/source/seed/gates, 20,001 updates and 900 seconds. Research then used
isolated full-SM MPS and shared admission of at most one new job per physical
GPU, with either identical RTX A6000 available to each queue. The
[fixed-tensor MPS probe](../mps_probe.json) matches all eight numerical digests;
the [five-control word audit](../control_word_audit.json) finds identical
20,001-update models, optimizers, named streams and all 24 observation packets.
Caller-global RNG snapshots differ and are explicitly excluded; no consumed
named-stream draw count is invented. This rules out the suspected disabled
feature divergence for the observed word trajectory. Archived and fresh source
cohorts remain distinct; archived passes never fill current result cells.

Charged seconds are 2,702.115146 for the control, including its retained
infrastructure cancellation, and 3,903.762470 for the three-strength search.
Reservations are released. Shared infrastructure timings are not a speed
ranking. Raw logs, JSONL traces, checkpoints and tensors remain under the
registered artifact archive, outside Git. Only compact reports, provenance,
reproducers and saved training GIFs are committed.

Software acceptance remains the 112 focused plus 316 integration tests,
27-host preflight, Forge validation, and the separately scoped public-API
restoring-force toy/GIF. The saved-only publication adds no unchanged
scientific rerun. This is provisional, one-seed screening evidence. Retain
0.01 as the measured task-specific lead; retain the existing global BCAP
leader and opt-in defaults because the word regression and native failures
prevent a general replacement.

Saved-only reproduction commands require the [covariance feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/18e2aa1018f25de8c447a59287025df6f6aded3e) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.
