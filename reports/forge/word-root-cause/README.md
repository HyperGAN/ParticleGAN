# Five-word joint task: root cause and bounded repair

**K3P, KA2 and R1/R2 now each pass the unchanged full word task.** The joint
BiGAN formulation is sound; the inherited recipes did not reliably learn this
host. We repaired two implementation defects, then used only existing settings
to obtain full passes. No reconstruction/cycle/contrastive objective, new GAN
stabilization method, architecture change, spread regularization, or seed search
was added.

These are **task-only diagnostic results**, not ordinary Tier 1 qualifications
or new default selections. The existing [current technique leaderboard](../technique-inventory.md)
selects reusable ordinary family candidates through the full task ladder. This
task-only study and its exact recipe export remain historical motivation and
reproduction evidence; they supply no ordinary qualification or current
task-specific solution selection.
K3P/KA2 clean/full arms disable existing training
noise and the short network horizon cap, crossing Forge's strict mechanism
activation boundaries relative to their original parents. They are authorized
existing-control ablations, **not ordinary same-signature configuration searches**.
Round 2 prior-rate and round 3 positive coefficient/cap contrasts retain the
signatures of their actual measured clean/full comparators. Frozen legacy
numeric-knob labels and round-1 inherited configuration/search annotations are
ancestor metadata, not search qualification. The additive [summary](summary.json)
and [exact task-only recipe exports](../../../configs/forge/selections/word-joint-task-v1.json)
make this distinction explicit without rewriting frozen receipts.

## Selected task recipes and numerical readout

All selected recipes use zero training input/output noise, the full 20,000-step
schedule, five learned 2D rows, batch 256, and **prior_reg=0**. No task geometry,
confidence threshold, observation cadence, initialization, or budget was relaxed.
Every run completed **20,001 updates per role**, used the same named seed-0
streams and initialization, and scored 24 clean/live observations. PASS requires
at least five terminal passing checks, confident canonical fraction ≥0.95, all
five modes, word/rejection TV ≤0.10, correctly paired reconstruction of every
word, and minimum correct reconstruction-token probability ≥0.90.

| Family / exact receipt | Base LR | D multiplier | Prior multiplier | Critic coefficient | Terminal passes | Final modes / quality | TV | Minimum inverse probability |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| [K3P coeff170/cap1](receipts/k3p-coeff170-cap1.json) | 0.0006 | 1.5 | 2 | 170 | **24/24** | 5 / 1.000 | 0.024219 | 0.997103 |
| [KA2 prior0.1](receipts/ka2-slow-prior-0p1.json) | 0.0006 | 1.5 | 0.1 | 1 | **21/24** | 5 / 1.000 | 0.024219 | 0.999942 |
| [R1/R2 mid-rate](receipts/r1r2-mid-rate-fast-roles.json) | 0.0012 | 1.5 | 2 | 1→0.1 | **22/24** | 5 / 1.000 | 0.024219 | 0.996137 |

The exports contain every `Recipe` constructor field, exact normalized recipe
hashes, parent/receipt/source identities, task fingerprint, initialization and
runtime. R1/R2 retains its declared native Adam, beta2 and penalty schedules;
K3P/KA2 retain their formulation optimizers, anchor, guard and penalty kernels.
A2 is requested but **eligible/applied zero times** in all K3P/KA2 runs: dense
sampling of only five rows does not activate its sparse-row branch. Direct
particle gain is also inactive. Synthetic public-component probes check fidelity
and do not demonstrate activation in these training runs.

Actual saved training observations illustrate the same acquisition/inverse goal:
[K3P GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-root-cause/media/k3p.gif), [KA2 GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-root-cause/media/ka2.gif), [R1/R2 GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-root-cause/media/r1r2.gif).
The [media receipts](media/receipts.json) bind each nine-frame GIF to its stored
observation hash and frame updates. Rendering used Python 3.12 separately from
the Python 3.14 scientific runtime, with zero updates or new sampling draws.
Numerical grades use every check, not just the displayed frames.

## What was wrong

**The joint pairs and routing are correct.** The critic compares `(x,E(x))`
against `(G(z),z)`; generator/encoder reverse the relativistic pair ordering, and
the intended roles receive their gradients. Five finite latent rows can represent
a bijective five-word solution. The implementation uses relativistic RpGAN,
so the original BiGAN theory is not a practical convergence guarantee for this
optimizer. The historical successful source/API examples retain their separate
recipe, sampling and source identities in the [original task readout](../five-word-joint/README.md)
and [provenance](provenance.json).

**The inherited schedule and noise change the training question.** The failing
generic K3P/KA2 selections used LR 0.006375, D/prior multipliers 1, input noise
0.5, persistent output noise 0.029 and a 1,600-step network horizon cap. G/E/D
reach nominal base LR **0.00006375 by update 1,600**, while the prior remains at
**0.006375 until 12,000**: a 100:1 prior/network base-rate ratio for much of the
run. Network decay finishes before input noise ends at 2,000 and before output
noise reaches full amplitude at 4,000. The old successful word recipe instead
used LR 0.0006, D1.5/prior2, zero effective noise and a full schedule. Its spread
coefficient was 1; that historical/source-shaped recipe is distinct from our
generic spread-0 arms. `ParticlePrior.standardize` is discarded by the factory
and cannot explain a difference here.

The word generator returns token probabilities, but policy output Gaussian
noise is added **after softmax, only to the fake joint input**. Real canonical
tokens sum to one; noisy fake tokens can have negative coordinates and varying
sums. This is a concrete categorical support mismatch, demonstrated by
deterministic controls in [endpoint anatomy](endpoints.json). It does not by
itself explain collapse: the clean/full low-rate K3P and KA2 bundles also failed.
Round-1 bundled comparisons do not isolate a single causal knob. KA2's active
critic controller can modify the declared nominal D rate; the explicit recipe
and recorded policy state, rather than base LR alone, define its training law.

**The failures were real learning failures, not only grader artifacts.** Original
K3P generated only three distinct words and mapped apple/berry to wrong words.
Original KA2 generated five confident atoms, but its grape encoder code sat off
the matching prior atom and crossed a confidence boundary in late observations.
Original R1/R2 had no measured word cell: its earlier ring failure stopped
ordinary progression. That UNKNOWN did not establish a word-task failure.
New clean low-rate failures had missing/duplicate generator basins. The
[bounded local grid](basin-grid.json) found no missing-word basin near the
learned codes; this is finite local evidence, not global impossibility. High-rate
clean/full failures instead saturated confidently malformed tokens. Lowering
only the prior rate restored KA2 at 0.1 and 0.25, but did not restore K3P at any
of the three tested rates.

**K3P's existing penalty needed a stronger coefficient at this joint scale.**
The joint critic has 170 input coordinates. Pure-A real-gradient L2 coefficient
1/(2×170) is 0.00294118, while the fake pure-A L2 cap is √170×kappa. Coefficient
170 matches **only the initial real-gradient R1 coefficient**. The fake term
remains capped, not R2, and later blended caps/proximity are scaled too. This
is not a swap to the R1/R2 technique. [Matched phase evidence](matched-phase-comparison.json)
also records different blend starts (K3P call16,028; KA2 call800); endpoint
gradient differences across families are correlations, not isolated causal
penalty proof.

Holding the measured clean/full low-rate K3P recipe fixed, coefficient170 passes
all24 checks under either cap1 or cap0.1, while coefficient1 fails under both
caps. [Final endpoint anatomy](k3p-critic-scale.json) shows five confident atoms
and separated codes. The two coefficient170 arms have **identical recorded
observations and final model tensors**, bound by [cap comparison](cap-equivalence.json)
and its [reproducer](cap_equivalence.py). They are not independent robustness
replications, and the lower cap has no observed incremental benefit. The chosen
cap1 export is the coefficient-only contrast. Saved observations/endpoints do
not establish equality of the unobserved trajectory or global cap inactivity.

**Two software defects were repaired for future evidence.** `==` thresholds
previously fell through to `<=`, falsely marking modes3==5 and reconstruction0==1
as individual passes. Shared sustained and endpoint scoring now implement
actual equality and reject unknown operators. `WordFixture` also ignored
explicit `prior_betas`; its prior group now honors that public Recipe field.
Both original whole word verdicts remain FAIL because other bounds failed, and
their prior_betas was null. Neither defect alone caused the original failures.
Current task source pins/evaluator revisions advance; archived receipts stay
unchanged. Numeric ≥/≤ behavior, gates, budgets and main-view revision3 remain.

## All bounded results, in protocol order

| Round | Arm | Full gate | Passing terminal suffix |
| --- | --- | --- | ---: |
| 1 | k3p-clean-full-low-rate | FAIL | 0 |
| 1 | k3p-clean-full-original-rate | FAIL | 0 |
| 1 | k3p-low-rate-original-noise | FAIL | 0 |
| 1 | ka2-clean-full-low-rate | FAIL | 0 |
| 1 | ka2-clean-full-original-rate | FAIL | 0 |
| 1 | ka2-low-rate-original-noise | FAIL | 0 |
| 1 | r1r2-low-rate-fast-roles | PASS | 6 |
| 1 | r1r2-mid-rate-fast-roles | PASS | 22 |
| 1 | r1r2-low-rate-original-roles | PASS | 9 |
| 2 | k3p-slow-prior-0p1 | FAIL | 0 |
| 2 | k3p-slow-prior-0p25 | FAIL | 0 |
| 2 | k3p-slow-prior-0p5 | FAIL | 0 |
| 2 | ka2-slow-prior-0p1 | PASS | 21 |
| 2 | ka2-slow-prior-0p25 | PASS | 11 |
| 2 | ka2-slow-prior-0p5 | FAIL | 0 |
| 3 | k3p-coeff1-cap0p1 | FAIL | 0 |
| 3 | k3p-coeff170-cap1 | PASS | 24 |
| 3 | k3p-coeff170-cap0p1 | PASS | 24 |

[Round1 protocol](round1-protocol.json) / [readout](round1-readout.json): nine
distinct bundles, 8,100 seconds reserved. Clean/low K3P/KA2 prediction falsified;
R1/R2 prediction supported. [Round2 protocol](round2-protocol.json) /
[readout](round2-readout.json): six prior-rate arms, 5,400 reserved. K3P
slow-prior prediction falsified; KA2 supported. [Round3 protocol](round3-protocol.json)
/ [readout](round3-readout.json): three K3P coefficient/cap arms, 2,700 reserved;
prediction supported. No unchanged recipe/seed reruns or subsequent paid stages.
All 18 completed with finite state, exact role updates and zero unintended RNG
deviations. Seven PASS, eleven FAIL, zero execution errors.

Measured adapter execution/evaluation/state-writing cost was **2,839.205 seconds**
(round1 1,400.775; round2 986.222; round3 452.208), against 16,200 reserved.
This excludes imports, source inspection, rendering, archive/export and reporting;
it is not a full worker cost or a speed ranking. Runtime was Python3.14.7,
Torch2.14.0+cu130, one RTX A6000 on GPU0, one worker, one CPU thread, deterministic
algorithms and no TF32. The same prior/model initialization hashes and RNG
manifest bind all arms; no results are pooled with historical CPU/API cohorts.

## Reproduction, preservation and recommendations

The [archive card](archive.json) provides a stable main-repository artifact
locator, byte identities and exact source snapshots for all three frozen
commits. Raw logs, states and observation streams remain ignored; compact
receipts, sources, final metrics and media are committed. Restore the relevant
source snapshot/commit before replaying a scientific arm, because each request
is bound to its executed runner and source manifest. An ordinary replay incurs
training; it is not required to regenerate the publication.

```sh
# Source ecc2dc8f: round1; 1a2c8d06: round2; 575d485e: round3.
# From the matching restored checkout; one arm only, a new ignored output.
python reports/forge/word-root-cause/reproduce.py --protocol reports/forge/word-root-cause/round3-protocol.json --run k3p-coeff170-cap1 --output runs/forge/word-replay > runs/forge/word-replay.log 2>&1
tail -F runs/forge/word-replay.log
# Read-only receipt/source/history audit; no training or requalification.
python reports/forge/word-root-cause/publication_audit.py --worktree .
# Durable archive audit also works if original absolute raw directories move.
python reports/forge/word-root-cause/publication_audit.py --worktree . --archive /home/martyn/dev/ParticleGAN/artifacts/forge/word-root-cause-18-arms-v1.tar.gz --archive-sha256 f67ffff63dc71c1490b77618660c8d8154975742f0b889c6e1511e7b0ee3f796
python reports/forge/word-root-cause/cap_equivalence.py runs/forge/word-root-cause-round3
# Stored-observation rendering only, using the separately documented environment.
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/word-root-cause/reproduce.py --protocol reports/forge/word-root-cause/round3-protocol.json --render k3p-coeff170-cap1 --output runs/forge/word-root-cause-round3
```

The preservation audit covers **598 preexisting files**: 292 technique receipts,
32 configuration cards, 11 search receipts, one selection, five goal leaderboards
and 257 immutable memory/history records. Generated experiment memory and its
compilation metadata are intentionally refreshed; they are outside that byte
preservation claim. New diagnostic records enter recall with qualification reuse
disabled. The [independent final audit](publication-audit.json) passes all 18
receipt bindings, all 3,599 inventory entries and all three executed source
snapshots. The [historical archive audit](historical-archive-audit.json) separately
verifies the old archive, all 333 original receipts and both original word states.
The software regression passed 4,080 tests (63 skipped, one xfail,
18 subtests); the bounded renderer passed separately in its required fresh
process. After regeneration, all **83 focused word/grader/publication/freshness
tests passed**. These software checks launch no additional scientific word runs.

The three historical recipe exports reproduce this declared word study with its
explicit host binding and sampling law. Reuse proceeds through ordinary global
family candidates and the existing complete view under a bounded campaign;
these task-only passes do not bypass prerequisites, select a current standard,
predict transfer or authorize default adoption. The provisional calibration
policy remains in force. Future host registration should verify effective role rates,
schedule horizons, categorical output noise and joint-input penalty units before
attributing a failure to a formulation. No automatic recipe adaptation is added.
