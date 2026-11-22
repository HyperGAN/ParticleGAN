# ParticleGAN Forge: the experiment engine

**ParticleGAN Forge** turns new ideas into comparable evidence through cheap gates,
shared leaderboards, and a memory of what the project has already tried.

Date: 2026-09-28. Status: framework complete as of 2026-09-30; scientific
profiles remain provisional. Initial inventory: `92dc0319`;
required additional source: PR #155 at `0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b`.

Implementation uses the existing **`codex/tiered-experiment-qualification`**
feature branch, based on **`develop`**, and
[PR #221](https://github.com/HyperGAN/ParticleGAN/pull/221). Keep the plan and all
Forge implementation in this single branch/PR, using incremental commits.

**Delivery scope, clarified 2026-09-30:** build the framework for proposing,
queuing, running and comparing ideas. The upstream research is still developing;
finding an accepted screening profile or a production formulation is future
work. The [final framework audit](../reports/forge/FRAMEWORK_READINESS_AUDIT.md)
verifies this delivery. Calibration, endurance and public-default promotion
requirements below remain scientific gates for future candidates, not unfinished
engine implementation.

## Goal and decisions

Make it easy for an agent to propose an idea, implement its changed mechanism,
run the cheapest useful tests, and compare its evidence with everything already
tried. Bad candidates should stop before consuming expensive training budgets.
Candidates that clear the gates should advance automatically within the declared
campaign budget. Every outcome should teach the next agent something.

The first deliverable in implementation is a classification of the existing
experiments and their evidence. Build the automation around that inventory,
reusing the current evaluators and their thresholds.

Decisions from this discussion:

- Three qualification tiers: **smoke, quality, endurance**. Higher tiers require
  lower-tier passes for the same candidate and goal.
- Tier placement is a small configuration edit, not a refactor of an experiment.
  Keep task/evaluator definitions independent of each view's tier assignments.
- New experiments assume **learned MoG particle priors**. Tests that require
  particle clouds explicitly declare `sigma=0`; they are a named exception.
- All ideas and experiments use one shared, extensible API. Centralize common
  code and defaults; declare formulation-specific API additions in one place.
- Dogfood ParticleGAN's public API wherever possible. Forge coordinates
  experiments; reusable formulation capabilities belong in the package.
- Publish a root-level **`EXPERIMENTATION.md`** as the operational starting point
  for engineers and AI agents. Put a prominent read-first link in `AGENTS.md`
  and `README.md` so the experiment workflow is discovered before work begins.
- Multiple leaderboard views over shared evidence. Start with **discriminator
  stability**; monotonically decreasing metrics are a **future** view, with no
  monotonicity gate in the first release.
- A modeled experiment lifecycle, including useful negative results and
  incomplete work, rather than a collection of launch scripts.
- A durable queue that agents can submit to and workers can drain across multiple
  GPUs, with automatic result recording and centralized, tail-friendly logs.
- Repo files are the source of truth. No database, hosted service, or private
  agent memory is required to propose, execute, compare, or learn.
- One compiled, searchable experiment-memory file explains what worked, what
  failed, what remains unknown, and what to try next.
- No seed-only idea experiments: screening uses one fixed protocol seed and
  isolated named RNG streams. A finished public-default candidate gets one
  preregistered robustness stage across a fixed seed set; this is a promotion
  stage of that candidate, not another search. Prefer metrics over images.
- Clock-free continuous learning is an explicit eligibility contract, separate
  from scheduled training. Each view explicitly declares live or EMA scoring.
- Refactoring and sweeping changes are allowed. Preserve existing evidence and
  reproducible protocols while moving new research onto the common interface.

## 1. What already exists

At the initial revision there are 181 tracked paths under `benchmarks/`, 100 under
`experiments/`, another 467 Python/shell files under `reports/`, 544 report
Markdown files, and 673 config files. These are **file counts, not independent
experiment counts**: many are shared helpers, frozen source copies, reports,
configs, or repeated descriptions of the same run.

These counts do not cover the later LR-free work. Phase A must also import
`reports/toy100/lrfree-search/**` from the pinned
[PR #155 source tree](https://github.com/HyperGAN/ParticleGAN/tree/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search):
358 tracked paths at this snapshot, including 42 Markdown and 168 JSON files.
These are another source inventory, not 358 new experiments or counts to add
without deduplication. Record each source revision independently and refresh the
inventory when later evidence is imported; `92dc0319` is not a completeness cutoff.

Important foundations:

| Existing component | Reuse in the new system |
| --- | --- |
| [Grid runner](../experiments/run_grid.py), [runner contract](experiment-runner.md) | Config validation, immutable resolved requests, output locks, source/config fingerprints, completion manifests, preserved failed attempts |
| [Transfer protocol](../benchmarks/transfer_suite/protocol.py) | Independently reconstructed live-weight verdicts, complete observation schedules, sustained success, explicit missing evidence |
| [Transfer suite guide](../benchmarks/transfer_suite/README.md) | Required/ranking/diagnostic importance, solvability controls, development/reserved families, domain-balanced comparisons |
| [Toy100 coverage gate](../benchmarks/toy100/gate.py), [accuracy gate](../benchmarks/toy100/accuracy_gate.py) | Full-budget coverage and fidelity qualification, terminal sustained checks, saved-sample audits |
| [Continuous probe](../benchmarks/toy100/continuous_probe.py) | Uninterrupted hold, extended training, target-shift recovery, matched frozen controls |
| [Selected K3P declaration](../reports/toy100/current-research-base.json) | Exact formulation, complete mechanism/response/latent drivers, known passes and remaining failures |
| [Gap-fill evidence](../reports/toy100/gap-fill-20260925/README.md) | Candidate lineage, task-level evidence, source hashes, historical cross-candidate comparison |
| [LR-free harness and GPU pool](https://github.com/HyperGAN/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/harness/README.md) | Existing submit/pool/wait tools, task fixtures, applied-rate receipts, stream-parity audits, package hashes, and result ledger; extend these instead of assuming queue infrastructure is absent |

The current transfer protocol already calls importance levels `tier`. The new
cost/progression field must be named `qualification_tier`; retain importance as
a separate dimension. An expensive diagnostic does not become a requirement
merely because it is assigned to Tier 3.

### Initial classification

Classify **techniques**, **experiment declarations**, **benchmark tasks**,
**execution attempts**, **evaluators**, and **reports** separately. K3P is a
candidate formulation; `b_cap` is a technique/configuration axis; `two_pole` is a
task. None of these is interchangeable with an individual execution attempt.

| Family | Proposed role and tier suitability |
| --- | --- |
| Mechanism/configuration checks and deterministic initialization | Preflight and cheap Tier 1 checks; shared helpers are support code, not scored experiments |
| Short behavioral hosts: `two_pole`, `unused_token_hold`, `ae_gan_hold` | Initial Tier 1 screen using their existing complete short protocols |
| Remaining behavioral, vector, and procedural-image hosts | Tier 2 quality and stability coverage; intentionally weak or ambiguous diagnostic cases remain diagnostic |
| Native `grid100`, `rotated100`, `staggered100` | Tier 2 full coverage **and** accuracy at the established 7,000-update budget |
| Own-state hold, ring extension, longer training | Tier 3 endurance; detect late collapse after ordinary quality gates pass |
| Target-shift recovery and reserved transfer families | Tier 3 adaptation/transfer tasks; attach to views that require those capabilities |
| Learned LR, smart descent, architecture/regularizer/noise searches | Candidate-generation families; qualify their outputs through shared tasks instead of treating a whole sweep as one result |
| Paired transport, MoG, sparse/denoising studies | Additional objective families with their own quality metrics and calibrated tier profiles |
| CIFAR/image training, trajectory/transition, Gym/control studies | Domain-specific qualification tracks; production evidence where relevant, not universal blockers for every GAN idea |
| Plotting, rendering, analysis, launchers, archived drivers | Support/evidence roles; retain lineage and dependencies, do not count them as separate scientific trials |

Inventory all tracked experiment/config/report families, including executable
drivers under `reports/`. Record untracked/local result roots as discovery gaps,
without silently importing them or treating them as reproducible repo evidence.
Every scoped file must map to a family/role or an explicit `unclassified` entry.
The coverage audit must fail on newly introduced, unclassified scoped files.

## 2. The agent-facing workflow

Keep the authoring burden small. An agent supplies a hypothesis, parent/reference,
changed factor(s), implementation/config reference, and desired goal. The system
fills in identifiers, protocol defaults, source hashes, tasks, budgets, logs,
receipts, and leaderboard wiring. Do not require agents to copy a launch matrix
or implement a new evaluator for each idea.

Proposed commands below describe the intended interface; they do not exist yet:

```sh
# Read relevant prior successes, failures, and unresolved evidence first.
python -m experiments.forge recall --goal discriminator_stability --query "critic anchoring"

# Scaffold a declaration and standard candidate adapter from a known parent.
python -m experiments.forge new --id critic-anchor-v2 --parent k3p \
  --goal discriminator_stability

# Implement the idea, then show exact gates, reuse, estimated cost, and blockers.
python -m experiments.forge plan critic-anchor-v2

# Stop at the first blocking failure. Default to Tier 1 unless a campaign says otherwise.
python -m experiments.forge run critic-anchor-v2 --through-tier 1

# Submit without waiting; GPU workers execute only currently eligible work.
python -m experiments.forge enqueue critic-anchor-v2 --through-tier 3 --campaign campaign.json
python -m experiments.forge drain --gpus 0,1 --workers-per-gpu 1
python -m experiments.forge queue
python -m experiments.forge logs --follow

# Compare successes, failures, and partial candidates; rebuild shared knowledge.
python -m experiments.forge board --goal discriminator_stability
python -m experiments.forge compile
```

Before launch, show exact duplicates and closely related prior attempts. Reuse
compatible completed evidence automatically. A changed hypothesis string alone
must not cause identical training to run again. A substantive code/config change
creates a new candidate revision with explicit lineage. A controlled rerun for a
broken execution remains an attempt of that revision, with its reason recorded.

Agents may propose new views or tasks, but cannot weaken an active protocol to
turn their own failure into a pass. Policy edits create a new view revision;
changes to task execution or scoring create a new task/protocol revision. Record
the rationale and relevant calibration without requiring code changes for retiering.

### One shared API, with explicit extensions

Use one candidate/task API across ideas and experiments, backed by the package's
recipe, prior, optimizer, and trainer/component APIs. The common runtime owns
construction, initialization, RNG streams, training/update orchestration,
checkpoints, result recording, and logging. Task adapters supply task data,
objectives, resource declarations, and evaluators; formulation code supplies the
changed mechanism. A new idea should not need its own launcher or training loop.

**Dogfood the public API:** prefer `Recipe` factories and `GANTrainer` where
supported, and the public component API for capabilities the trainer does not
yet expose. Keep Forge's API a thin experiment contract around those interfaces,
not a parallel implementation of losses, prior sampling, optimizers, or updates.
Put reusable missing capabilities into the public package API with docs and
tests; all experiment adapters then consume that same implementation. In
particular, shared MoG support should exercise the public prior and training
path, not introduce a private Forge-only MoG trainer.

Declare `execution_path` (`public_trainer`, `public_components`, or
`archived_replay`) and API exceptions in each receipt. Custom host loops must
bind shared public components and verify parity where a reference path exists.
Archived private hooks can reproduce historical evidence, but cannot alone
establish current public-API or public-default readiness. Qualification for those
claims must include execution through the shipped public interface.

Put the versioned `FormulationContext` contract and extension registry in
`experiments/forge/api.py`. This is the normal place to expose an additional
training variable to experiments. An extension declares its name, type/shape,
producer, required/optional status, explicit default if any, gradient/parameter
ownership, optimizer binding, initialization/RNG behavior, and checkpoint state
as applicable. The shared runtime resolves that declaration once for every
compatible task. Expose training inputs, not hidden evaluator labels or scores.
Where the public API already defines a capability's schema/defaults/ownership,
reference that authoritative definition; do not redeclare it in Forge. Add
reusable capability definitions in the package once, with Forge registering the
binding. Reserve Forge-only definitions for experiment orchestration fields.

When a formulation needs new behavior, implement it once in the shared package
or runtime and add its binding at this extension point. If a task family needs
an extra provider, add that provider in its shared adapter, not in every
experiment. A new field does not guarantee every host supports it: missing,
ignored, or unsupported required variables produce a reasoned `BLOCKED` before
training. Avoid permissive argument dictionaries that silently drop additions.

Record `api_version`, `requires_capabilities`, and `api_changes` in the candidate
card and readout. An API change states which variables were added/changed, why,
which shared implementation/provider supplies them, affected hosts, and any
compatibility or checkpoint migration. The scaffold fills unchanged fields from
the parent; agents document the delta, not repeat a whole framework definition.

Keep experiments DRY: one prior factory/default resolver, capability binder,
training implementation per supported task family, evaluator per metric protocol,
and common runner/queue/logging/checkpoint/result code. Configs refer to shared
definitions and declare only overrides; receipts still store the full resolved
values. Historical source snapshots remain immutable evidence, not templates for
new copied runners. Bind enqueued candidates to immutable source snapshots,
including their worktree changes; later edits cannot alter queued/running work.

### Learned MoG priors are the default

New Forge declarations resolve to learned MoG particle priors through the shared
factory: learned component locations and an explicit nonzero prior-width policy.
Use versioned shared defaults so each experiment need not restate the prior.
Declare which locations, widths, and mixture weights are learned or fixed; a
learned prior does not imply that every parameter is trainable. The current
[`MoGParticlePrior`](api.md#mogparticleprior) learns locations, has uniform
component weights, and stores sigma as a fixed scalar buffer. Learned width or
nonuniform weights are explicit formulation extensions, not assumed capabilities.

Particle-cloud tasks must explicitly declare `prior_kind=particle_cloud`,
`sigma=0`, and a reason in their task specification; this is proposed Forge
metadata mapped to the appropriate package API. Record the read-standardization
policy too: zero-sigma MoG matches plain-cloud outputs and RNG consumption only
with `standardize=False` and matching centers/indices. Keep latent prior sigma
distinct from generator output noise and observation noise. Train and evaluate
the declared sampling law; `.eval()` must not silently replace a MoG with centers.

This is a required Forge default and API integration task, not a claim that the
current convenience trainer already supports it. At the initial revision,
[`GANTrainer`](../particlegan/training.py) rejects MoG and
[`Recipe.make_optimizers`](../particlegan/recipes.py) enables A2 only for the exact
plain `ParticlePrior` type. Implement shared MoG support and explicit mechanism
capabilities, then verify update, sampling, RNG, and checkpoint parity before
adoption. Never silently disable a candidate hook when changing prior type.

Import old cloud runs with their actual prior, not Forge's new default. Converting
one to MoG creates a new candidate/task execution identity and requires new
evidence. Where a frozen historical test intentionally needs a cloud, classify
that exception explicitly. Calibration of the initial three smoke hosts must
state their prior regimes; cloud passes alone do not qualify MoG behavior.

## 3. Repository records and lifecycle

Proposed layout, using the repo's existing code/config/report conventions:

```text
EXPERIMENTATION.md                     # read first: engineer/agent workflow and quickstart
AGENTS.md                             # prominent pointer to the guide and experiment memory
experiments/forge/                    # CLI, scheduler, adapters, map/reduce
  api.py                             # shared context, extension schema and capability bindings
configs/forge/
  defaults.json                           # versioned common defaults, including learned MoG prior
  catalog.json                            # task/family/source inventory
  tasks/<task-id>.json                     # versioned host/evaluator references and prior requirements
  protocols/<protocol-id>.json             # immutable versioned gates and budgets
  views/<view-id>.json                     # goal, task tier placement, importance, ranking, eligibility
  campaigns/<campaign-id>.json             # resource limits and candidate selection
  ideas/<idea-id>.json                     # concise agent-authored declaration
reports/forge/
  records/<record-id>.json                 # normalized historical or new experiment card
  attempts/<attempt-id>/
    request.json                          # fully resolved immutable request
    events.jsonl                          # lifecycle and execution events
    result.json                           # terminal result, cost, evidence references
    evidence.json                         # hashes, runtime, evaluator receipt
  leaderboards/<view-id>.json              # generated data
  leaderboards/<view-id>.md                # generated reviewable view
  EXPERIMENT_MEMORY.md                     # single compiled agent reading file
  compilation.json                        # input hashes, coverage, conflicts, reducer version
runs/forge/                               # durable local operational state; ignored by Git
  queue/                                  # pending requests, claims, terminal receipts
  workers/                                # worker identity, GPU allocation, heartbeat
  claims/<compatibility-key>/              # prevent duplicate work across campaigns
  events.jsonl                            # centralized structured log
  status.json                             # current queue, workers, resource use
  <campaign-id>/
    progress.jsonl                        # campaign-filtered event stream
    <attempt-id>/run.log                   # complete child stdout/stderr
```

Declarations, normalized records, compact evidence, and generated readouts live
in Git. Bulk checkpoints, verbose logs, and large sample arrays stay in repo-local
ignored run directories by default; keep their paths and hashes in receipts.
Copy/adopt evidence needed for portable regrading into the durable report, or
explicitly mark the record as dependent on unavailable local artifacts. Do not
claim that a Git checkout can replay evidence it does not contain. No required
state should live in `/tmp`, an external agent session, or an undocumented path.

An experiment card records:

- Stable idea ID, exact candidate revision, parent, technique tags, hypothesis,
  changed factors, and links to related/duplicate ideas.
- `mechanism_class`: `structural`, `floor_constant`, or `sampling_only_patch`,
  with a rationale and secondary classes for mixed mechanisms. Record whether
  training dynamics, public sampling, or both change. A structural-first view
  can declare that preference; the class itself never changes a numeric verdict.
  The LR-free research recommendation should prefer structural fixes and retain
  sampling-only row EM as a diagnostic patch rather than a training fix.
- View/protocol versions, task identity, fixed seed/fixture policy, full effective
  config, implementation/harness hashes, and training/evaluation budgets.
- Resolved prior kind, width/standardization policy, parameter trainability,
  and any explicit particle-cloud exception; `api_version`, `execution_path`,
  required capabilities, and the formulation's API delta with affected hosts and
  migration requirements.
- `claim_contract`: scheduled versus clock-free, shared-setting requirement,
  scoring weights (`live`/`ema`), and training/public-sampling/evaluation laws.
  Record EMA definition/decay and RNG/initializer manifests, not only a seed.
- `applicability`: supported, unsupported, or unknown, with required API/host
  capabilities, parity receipt, blocker reason, and the adapter work needed.
- Raw metrics and their meanings/directions, live versus EMA, complete versus
  partial windows, controls, artifact references, and provenance quality.
- Outcome per task and view; failure reason/first failing step where available;
  measured compute, elapsed time, and resources.
- Conclusion, limitations, and next action: advance, investigate, revise,
  deprioritize, or preserve as a useful negative result.

Separate administrative lifecycle from scientific qualification:

| Lifecycle state | Meaning |
| --- | --- |
| `proposed` | Idea exists; hypothesis and prior-art references can still change |
| `ready` | Candidate revision and protocol are resolved, validated, and frozen |
| `running` | One or more eligible tasks have active attempts |
| `awaiting_readout` | Execution stopped or exhausted its permitted work; conclusions need recording |
| `concluded` | Metrics, comparison, explanation, and recommendation are published; a failed idea can be fully concluded |
| `superseded` / `abandoned` | Administrative disposition with a reason and successor link where applicable; evidence remains visible |

An execution attempt separately reports `queued`, `running`, `completed`,
`error`, `timeout`, or `cancelled`. A gate reports `PASS`, `FAIL`, `INCOMPLETE`,
`INVALID`, `NOT_RUN`, or `BLOCKED`. Infrastructure failure is not scientific
evidence against the idea. Missing evidence never earns a pass.

A public-API extension rejected by a custom host is `BLOCKED` at qualification,
with the exact capability mismatch. Preserve the original attempt's `ERROR`
and exception rather than rewriting raw history. It is neither a candidate
`FAIL` nor a passing/skipped task: it stays in the required denominator and
prevents full qualification until the host is adapted and parity is verified.
Do not repeatedly retry a known incompatibility as though it were a transient
worker error. Changing the supported-use scope requires a new explicit view.

Qualification is a derived field **per candidate revision, protocol, and view**:
highest fully passed tier, next eligible task, blocker, and evidence completeness.
A Tier 3 failure leaves its Tier 2 result visible; it does not rewrite history.
Passing a stability view need not pass an adaptation or future monotonicity view.

Append attempt events and write final artifacts atomically. Concurrent agents own
different idea/attempt files. Reserve work by candidate revision, task, and evidence
compatibility key, not just attempt ID: two requests for the same work should
attach to one execution. Use per-attempt locks and a single reducer for shared
outputs. Recovery after a crash inspects the saved request, process/lock state,
and completion receipt before resuming or creating a new attempt. Never silently
overwrite failed evidence or count two attempts as two independent ideas.

### Paired initialization and random streams

Each new comparison protocol fixes one screening seed and a versioned derivation
of named `init`, `data`, `prior`, `noise`, and `eval` streams. Record the resolved
seed, stream bindings/derivation, RNG implementation/runtime, initializer version,
and fixture hashes in the request and evidence compatibility key. Stream identity
must not depend on candidate name, worker, GPU assignment, or queue order.

Isolate streams further by component and draw purpose. One extra draw by a new
mechanism, an evaluation call, or a changed model shape must not shift unrelated
data, prior, noise, or initialization draws. New stochastic mechanisms get a
declared private stream. Audit draw counts/positions and digests for the shared
consumers, aiming for **zero unintended stream deviations**; record intentional
sampling differences separately. Undeclared drift invalidates a paired comparison.

"Same init" means the same initializer and initialization streams, with identical
draws for shared compatible parameter components. It does not require identical
weights for different architectures or shapes. Changed components get stable
bindings without consuming other components' draws. Save all stream states in
checkpoints and verify that resume and evaluation preserve the training streams.

Historical #155 fixtures have their own stream layout. Preserve and audit that
layout for historical replay; introducing Forge's split-stream scheme is a new
protocol revision, not permission to relabel older results as equivalent.
Different seeds, initializers, stream bindings, or sampling laws are different
evidence. Never substitute or average them away under one receipt identity.

### Screening and public-default robustness

One deterministic screening run is one draw. A small difference in centre error
is not a demonstrated mechanism improvement just because it crosses a threshold.
Import the review's reported sub-0.03σ sensitivity findings with their original
scope; do not turn that observation into a universal significance threshold.

After a finished candidate is frozen and passes its applicable qualification
gates, a **public-default promotion claim requires one preregistered robustness
stage**. Freeze the seed list, tasks, controls, budgets, scoring weights,
aggregation, and acceptance rule before execution. Run the same candidate on that
fixed set, preserve every outcome and denominator, and forbid seed selection or
parameter tuning inside this stage. This is a stage of the finished experiment,
not a set of new idea variants or an automatic sweep for every screen survivor.
Declare live/EMA scoring in that frozen contract; do not choose the scoring
weights retrospectively after observing the robustness results.

The promotion stage is the explicit exception to seed-only submission rejection.
Its requests must reference the frozen candidate and registered seed set. Each
seed has distinct evidence identity; reuse is possible only for an exact match
including that seed and the full protocol. A failed/partial robustness stage
cannot confer a public-default claim, even when the original screening run passed.

## 4. Tier gates: cheap rejection, then stronger evidence

### Retiering is a configuration change

A task has a stable execution/evaluation definition; `qualification_tier` lives
in a view's assignment of that task. To move a test between tiers, edit that
assignment in `configs/forge/views/<view-id>.json` and create a new view revision.
No script moves, trainer edits, duplicated configs, or rewritten evaluators are
needed. An optional `forge retier` helper can make the edit, validate it, and show
the resulting queue/leaderboard changes. For example, these are view entries:

```json
{"task": "img_bars4@v1", "qualification_tier": 2, "importance": "required"}
```

Changing `qualification_tier` to `1` moves the unchanged test into smoke.
Tier, importance, order, and ranking are policy; task inputs, prior, budgets,
schedule semantics, and scoring are execution/evaluation. Keep separate
fingerprints for these layers. A tier-only move reuses compatible result receipts
and recomputes qualification without new training or a new candidate identity.
Changing a task's budget or prior is a new task definition, not a tier-only move.

Validate prerequisite/checkpoint dependencies, missing task references, cycles,
and accidental empty required tiers before accepting an edit. Retiering itself
recomputes boards and shows missing evidence; it never enqueues training. A later
explicit submission against the new view queues only newly needed evidence.
Existing campaigns retain their pinned view revision; their
history is not regraded silently. The new view can compare the same evidence
under its new progression policy and show that policy change explicitly.
Candidate attainment remains computed from evidence, never a manual tier label.

### Initial profile

Preflight is a zero-training validation step, not a fourth qualification tier.
It checks adapter availability, resolved config, resource limits, provenance,
duplicate work, and applicability to the chosen view.

| Tier | Question | Initial discriminator-stability profile | Progression |
| --- | --- | --- | --- |
| **1 — smoke** | Does the mechanism execute correctly and preserve basic bounded adversarial learning? | Existing `two_pole` (80 updates), `unused_token_hold` (200), `ae_gan_hold` (250), plus finite-state and intended-update checks | All required checks pass before any Tier 2 launch |
| **2 — quality** | Does useful live output quality survive across the declared problem families? | Complete the 19 transfer hosts and three 7,000-update native coverage/accuracy gates, reusing the three identical Tier 1 host results | All required checks pass before Tier 3 |
| **3 — endurance** | Does a qualified solution remain good after acquisition and extended training? | First-convergence ring hold for 1,200 updates and the following 300-update extension; add other long-budget hosts as their protocols are calibrated | Qualifies the measured endurance scope; production promotion is a separate decision |

This is an initial profile to calibrate, not a declaration that these smoke tests
already predict production success. Tier 1 contains only **530 host updates**,
using already short complete tests rather than shortening a full native run.
No wall-time or FLOP target is claimed until measured on the reference runtime.
If any proposed smoke criterion rejects an established useful reference, examine
whether it is detecting a relevant failure or an accidental implementation bias
before freezing the protocol.

**Phase D adoption blocker:** replay the #155 lineages through this proposed
screen before making it the default. Those lineages expose failures in
`img_bars4`, `mode_hold`, `img_intensity2`, `vector_unequal_mass`, and native100
that the three cheap hosts may not predict. First regrade compatible saved
evidence; run only missing comparisons against frozen fixtures and packages.
If the proposed hosts are unsupported, report that calibration gap explicitly.

Publish a per-lineage matrix and counts of false accepts (smoke passes, later
goal-relevant reference fails) and false rejects (smoke fails, independent later
reference succeeds), plus rejection cost and completeness. Define that reference
without including the proposed smoke predicate itself, which would make false
rejections impossible by definition. Missing, incompatible, or blocked evidence
stays unknown rather than entering either count. Agree on cost/error acceptance
criteria before the replay; incomplete or unacceptable calibration blocks Phase E
adoption. Revise the smoke set/order in a new protocol if the saved FLOPs do not
justify its missed failures or useful candidates lost.

The ring endurance task has a **maximum total budget of 7,500 updates**: 1,200
initial updates, up to 4,800 settling updates seeking the first 200 consecutive
qualifying live checks, then 1,200 hold updates and 300 extension updates. The
confirmation is inside the settling allowance. Reserve the full possible cost,
not just the 1,500 updates after convergence. Preserve the existing
[convergence declaration](../reports/toy100/gap-fill-20260925/sources/k3p/convergence_gate.py).

Keep target-shift recovery as a separately visible Tier 3 capability. An
adaptation view can require it, including the matched frozen negative control.
The first stability profile should not silently equate recovery after a changed
target with remaining stable on a stationary target.

Protocol rules:

- Freeze required tasks, thresholds, observation cadence, evaluation sample
  counts, budgets, schedules, and permitted early stops before a campaign.
- Recompute verdicts from complete recorded evidence using existing evaluators.
  A process exiting successfully or a trainer writing `PASS` is insufficient.
- Preserve live scoring for existing live-qualified protocols, including #155.
  Record paired EMA results separately with exact decay and sampling law; the
  review reports EMA .995 passing 3/3 on D-tracking runs. A view can explicitly
  qualify EMA, but cannot silently replace a live failure, pool live/EMA results,
  or select the better weights after observing scores. Averaging the evaluated
  generator is distinct from an EMA critic used inside a training mechanism.
  Best-checkpoint results cannot hide later collapse under either policy.
- Preserve complete observation/sustained-pass requirements. For transfer hosts,
  the existing gate uses 24 observations and a terminal passing suffix; for native
  tasks, retain both coverage and accuracy evidence.
- Endurance must continue the candidate's own trained state, optimizer moments,
  and random streams. For a **scheduled** candidate, preserve its original schedule
  horizon: do not restart or stretch annealing because training lasts longer. This
  earns scheduled-endurance evidence, not clock-free eligibility. The current
  [`--steps` override changes schedules](toy100.md#run-every-problem-or-inspect-one),
  so it cannot serve as an exact-prefix screen.
- A frozen or nonlearning system must not qualify merely because its numbers are
  steady. Check actual intended optimizer updates and nondegenerate output quality.
- Verify candidate-specific hooks are installed and exercised by suitable smoke
  checks. A delayed guard or anchoring phase that never activates provides no
  evidence about that mechanism; declare expected inactivity on individual hosts
  and exercise it in a targeted cheap mechanism check when needed.
- Archived K3P drivers have module-global mechanism/EMA/LR/latent/response state
  beyond ordinary model and optimizer checkpoints. Run endurance uninterrupted
  until complete fresh-process state restoration is independently verified.
- Timeout, invalid evidence, or a failed **required** task blocks downstream
  qualification. Diagnostic errors/failures remain visible without vetoing required
  passes. Any process that exceeds its execution budget must still be stopped.
- Changing training/scoring code, effective config, or the measurement protocol
  changes evidence compatibility. Editing tier placement or ranking changes the
  view revision only; reuse unchanged task evidence and recompute attainment.
  Do not rewrite the verdicts or qualification recorded for older view revisions.

Tests should reject as early as the protocol permits. Within a tier, order tasks
by observed rejection value per measured cost, while preserving a frozen order
for each campaign. Do not early-stop on an unvalidated noisy signal. Cap tasks,
candidates, and campaigns independently so a faulty mechanism cannot spend an
unbounded budget before producing its first useful result.

### Additional eligibility for clock-free / "LR-free" claims

The #155 goal is no horizon-driven LR or noise schedule, one setting across tasks,
and continuous learning. Register that as a required claim contract. A scheduled
K3P pass or a constant-LR run does not establish this contract.

- Require effective rates and adaptation decisions to depend only on declared
  learning state/statistics, not step/epoch labels, elapsed time, planned horizon,
  fixed release steps, or evaluator feedback. One shared setting must apply across
  tasks; fixed host resource/shape differences are declared by the task, not tuned
  per-candidate settings.
- Audit the source and applied-rate/decision receipts for hidden clocks, including
  table-stationarity ladders, warmups, freeze/release milestones, and counter-driven
  optimizer corrections. Saving a counter in a checkpoint does not make it a
  state-based rule. Document allowed state and every counter's effect; an
  unexplained clock dependency blocks the clock-free claim.
- With identical permitted learning state and RNG state, changing the external
  step label or requested stopping horizon must not change the next update.
  Verify common-prefix parity for different evaluation budgets and full-state
  checkpoint continuation. Evaluation deadlines may bound compute, but must not
  control the learner's rates, noise, or decisions.
- Require longer native continuation for this claim: preserve the 7k prefix and
  continue the same state to at least 14k, with dense observations around observed
  late releases and the frozen terminal quality/holdout checks. Predeclare any
  longer continuation needed from the clock audit. The review's dt075 grid and
  staggered failures after the stationarity ladder released are mandatory
  calibration cases once their exact package/fixture receipts are imported.
- Require continued capacity to learn, not just low motion. Pair quality/hold
  checks with an appropriate sustained adaptation test and its negative control.
  Report finite tested duration; a 7k or 14k pass cannot prove indefinite learning.

Failure of this eligibility contract disqualifies the clock-free claim, not the
candidate's already measured scheduled or ordinary quality results. Unsupported
instrumentation is a blocker, not an inferred scientific failure.

## 5. Queue, GPU workers, and centralized logs

Submission and execution are separate. Agents should enqueue a resolved candidate
once and return to other work. A repo-local queue coordinator decides which tasks
are eligible and dispatches them to GPU workers. `run` is a convenient submit-and-
wait wrapper around the same path, not a second execution implementation.

The first implementation supports one machine with several GPUs and multiple
submitting agents/worktrees. Keep coordinator state in one configured repository
root; worktrees submit there and retain their own immutable source references.
Default to one worker per GPU. Permit explicit sharing only when memory/resource
estimates and campaign policy allow it. Multi-machine execution can later use the
same request/result contract; do not require a distributed service initially.
Enforce one active coordinator per queue; additional drain invocations attach to
it or fail clearly. Reserve GPU capacity and campaign budget atomically across
workers, counting outstanding reservations as well as completed spending.

Queue behavior:

1. **Enqueue:** validate the idea, freeze the request, assign an idempotency key,
   and persist it atomically. Repeated submissions return the existing request or
   reusable result. Record requested maximum tier, GPU/memory requirements,
   campaign, priority, and total resource allowance.
2. **Claim:** atomically reserve an eligible task and its candidate/task
   compatibility key across campaigns. Allocate GPU(s), memory headroom, CPU
   threads, and a worst-case task budget before starting a child. Pass device
   assignments explicitly, including `CUDA_VISIBLE_DEVICES` where appropriate.
3. **Execute:** spawn in the declared source environment, record PID/process
   group and worker heartbeat, stream logs, and persist incremental progress.
   A child receives only its own output directory and bounded execution allowance.
4. **Record:** atomically save raw result/evidence references, exit status, elapsed
   resource use, and independently graded verdict. Every exit path emits a terminal
   receipt, including errors, timeouts, cancellations, and unavailable evidence.
5. **Advance or stop:** update the reducer; enqueue the next eligible required
   task only if all prerequisites pass, the requested tier cap permits it, and
   the remaining campaign budget can cover it. A Tier 1 failure never reserves
   downstream Tier 2/3 GPU jobs.
6. **Drain:** exit when submitted work and its permitted automatic promotions are
   finished or explicitly blocked. An optional `--watch` mode waits for new ideas.
   Surface unschedulable GPU/memory requests as blockers instead of hanging silently.

Prioritize cheap screens while aging queued quality/endurance work so survivors
are not starved. Allow filtering/draining by campaign, goal, or priority. Expose
queue inspection, cancellation, bounded retry of execution failures, and campaign
pause/resume. Retry neither a scientific failure nor a seed-only variant as an
automatic recovery strategy.

When several requests share one execution, record its subscribers and cost owner.
Charge actual compute once under a declared accounting policy. Cancelling one
subscriber detaches its request; it must not kill work another subscriber still
needs. If the paying campaign withdraws, transfer the remaining reservation only
after another campaign accepts it, or stop the work. Shared jobs must not evade
resource limits through double counting or uncharged ownership changes.

Budget checks apply before **every** task, including after successful promotion.
`--through-tier 1` must stop after Tier 1 even when it passes. Timeouts and forced
cancellation terminate the entire child process group. A graceful drain shutdown
stops claiming new jobs and records the disposition of active work.

Heartbeats alone do not prove a training process died. Before reclaiming expired
work, check the process/lock state and terminate or fence the old worker; stale
workers cannot publish a second accepted completion. A restarted coordinator
reconstructs queue state from durable requests and receipts, reconciling active
workers before relaunch. Resume from a checkpoint only when the task adapter can
restore all required state; otherwise preserve the attempt and restart explicitly.

All workers emit structured records with timestamp, campaign, idea/revision,
attempt, worker, GPU, tier, task, event type, latest metrics, and cost. Workers own
separate streams; one collector writes `runs/forge/events.jsonl` and materializes
`status.json`, preventing interleaved JSON writes. Keep full child stdout/stderr
in stable per-attempt logs and make the central stream link to them. Provide
`logs --follow` filters for candidate, task, worker, and GPU, with a bounded replay
of recent events so a new agent can quickly see what is happening.

On each completion, update durable records and materialized leaderboard views.
Refresh the compiled memory after the experiment readout, or include an explicit
pending-readout entry in the interim. A user should be able to tail one file while
several agents feed several GPUs without assembling logs from each worktree.

## 6. Goal-specific leaderboards and useful compute

Store one candidate-by-task evidence matrix. Views select requirements and
ranking rules from it; creating another view must not trigger another complete
training sweep. Shared evidence is reusable only when task, candidate revision,
measurement protocol, budget, seed, RNG/initializer/fixture manifest, sampling
law, scoring weights, and runtime compatibility match.

Every board should show candidate/parent, exact revision, highest passed tier,
per-tier status and pass denominator, missing/blocked tasks, principal raw
metrics, first blocker, cost, and next action. Show failed and partial candidates
by default, with filters for tier, goal, family, outcome, and evidence quality.
Keep unmeasured candidates visible as unranked rows.

Show cost **beside pass counts**: required passes/total, FAIL, BLOCKED, scoring
weights, wall time, relative cost against a named reference, and compute/memory
where available. Break out training, sampling/calibration, and evaluation costs
when measurable. For example, #155's fresh-reservoir sampler has native100 **3/3**
alongside **4.4–4.8× reported wall time**; its full22 result is **10 PASS / 4 FAIL /
8 BLOCKED** after mapping the recorded pretraining parity errors. Neither the
three native passes nor the implementation blockers should hide the other facts.
Treat that cost multiplier as a historical measurement under the reported runtime,
not a portable speed guarantee.

Compare only compatible cohorts. Full-quality candidates should not lose to a
short screen because the latter omitted difficult tests. Filter by qualified
tier first, then apply predeclared goal-specific metrics within the cohort.
Expose quality and cost separately; preserve existing domain-balanced ranking
where appropriate. Avoid a universal average that lets easy tasks cancel a
required failure or lets a large family outvote the others.

Initial views:

| View | State and interpretation |
| --- | --- |
| Discriminator stability | First implemented qualification profile: bounded learning, useful quality, and stationary endurance |
| Clock-free continuous learning | Required additional eligibility whenever "LR-free" is claimed: shared settings, no hidden schedule, native continuation and continued adaptation |
| Quality / coverage | Alternate readout of the same evidence; separates ordinary quality from endurance failures |
| Live / EMA | Separate declared scoring policies; retain live historical gates and paired EMA evidence without substituting or pooling them |
| Adaptation | Later profile requiring target-shift recovery and its controls |
| Monotonic progress | Future design only; later choose appropriate error/quality signals, tolerances, windows, and improvement requirements. Do not assume GAN losses should decrease monotonically |
| Production/domain readiness | Later application-specific views combining target-workload tests with the preregistered fixed-seed robustness stage for a finished public-default candidate |

Measure the automation itself: cost to reject, cost to qualify, rejection rate
by tier, tasks avoided after rejection, evidence reuse rate, and incomplete/error
rate. Measure actual updates, G/D forward/backward work where instrumented, peak
memory, device/runtime, elapsed time, and concurrency. Label FLOPs as measured,
estimated, or unavailable; **equal steps are not equal compute**, particularly for
extra-gradient, penalty, or EMA-critic work. Do not rank speed from runs with
incompatible hardware or contention.

## 7. Compiled experiment memory: map, then reduce

The main agent reading artifact is
`reports/forge/EXPERIMENT_MEMORY.md`. It should answer: what has been
tried, in what exact form, against which goals, what happened, and what a new idea
needs to change to be informative. Include successes, failures, mixed results,
and unfinished questions. A blacklist of technique names would discard too much.

### Map: normalize independently

Partition the historical inventory by report/experiment family. Subagents or
importers produce one normalized card per exact experiment revision with explicit
links to source evidence. They must not edit the shared compiled file directly.
Map workers can run in parallel because each owns different records.

The initial import includes the pinned #155 LR-free lineages, not just the earlier
K3P gap-fill report. Bind their frozen host fixtures, initializer and stream
receipts, package/source manifests, overrides, applied-rate traces, raw result
files, parity refusals, and controls. Copy compact supporting evidence into repo
records with original paths/hashes; mark unavailable local checkpoints or samples.
Snapshot later follow-ups separately when obtained. A documentation-only claim
with no bound result stays narrative evidence, not a newly verified gate pass.

Prefer structured configs, result snapshots, event streams, and evaluator outputs.
Use Markdown to recover hypotheses, explanations, and recommendations, citing the
specific source section. Mark narrative-only claims as such. Do not infer a pass
from positive prose, a failure from a missing file, or a scientific outcome from
the number of scripts in a directory.

Track corrections and superseded reports explicitly. In #155, clean versus noisy
public sampling changed the interpretation of earlier scores. Pin the actual
sampling law and scorer revision from the run; do not combine a historical helper
README's policy with a newer result or treat a sampling-only fix as a training
mechanism improvement.

Record source hashes and mapper/schema versions so unchanged sources need not be
mapped again. Preserve distinctions between negative controls and candidate
failures, and between a failed configuration and its broader technique family.
Expose conflicts, ambiguous identities, absent raw evidence, and unmapped sources
in the coverage report; do not resolve contradictions by majority vote.

### Reduce: compile and compare deterministically

Validate cards, resolve exact duplicates, retain attempt/lineage relationships,
and materialize per-view qualification and comparison tables. Evidence matching
uses hashes and configuration identity, not a short label such as `K3P`.
Conflicting evidence stays visible and blocks unsupported promotion.

Generate the single Markdown memory file with:

1. A compact index and goal/family summaries for quick agent orientation.
2. One concise entry per experiment revision: hypothesis/change, outcome and
   scope, strongest metrics, failure mode, limitations, next action, evidence links.
3. A section of unresolved ideas, contradictory reports, and missing evidence.
4. Compilation provenance and inventory coverage, so omissions are detectable.

Keep entries concise and link raw detail rather than copying whole reports.
The `recall` command selects relevant entries from the same records for a small
agent context; the complete single-file history remains available. Compilation
must be deterministic, incremental, and independent of training or image access.
Exclude generated outputs from mapper inputs to prevent recursive self-ingestion.

After each attempted idea, including early failures, the lifecycle requires a
short readout and recompilation. Agents should find closely related prior work
before execution and cite it in their declaration. A retry of a failed idea is
valid when it names a substantive changed factor and the failure it addresses.

## 8. Historical calibration examples

The existing evidence already demonstrates why separate tiers and views matter.
These are **historical results**, not newly earned passes under the proposed
protocol. Import them with their original scope before considering reuse.

| Exact historical formulation | Declared quality suite | Standard ring hold | Following extension | Target-shift recovery |
| --- | --- | --- | --- | --- |
| K3P, selected base | 22/22 | 1,200/1,200 pass | 300/300 pass | Fail: 28/81 deadline checks |
| K3G | 22/22 | 1,200/1,200 pass | Fail: 63/300 passing | Fail: 22/81 |
| P1, step-keyed K3 formulation | 22/22 | 1,200/1,200 pass | Fail: 63/300 passing | Fail: 22/81 |
| RG5 + `b_cap`, no A2, base floors | 18/22 | Not converged | Not reached | Fail: 0/81 |
| RG5 + A2 + `a_r1r2`, .1/.1 floors | Not qualified as a full 22-task configuration | Different shift-protocol hold | Not run | Paired pass: 81/81 live versus 0/81 frozen |

Sources: [gap-fill comparison](../reports/toy100/gap-fill-20260925/README.md),
[gate-by-gate evidence](../reports/toy100/gap-fill-20260925/qualification-summary.json),
[paired recovery audit](../reports/toy100/gap-fill-20260925/rg5-recovery-pair.json).
The quality total uses the declared runs; historical additional seed results stay
separate. No new seed experiment is needed to build this history.

The memory should explain these outcomes: K3G and P1 are examples of ordinary
quality passing while endurance fails; K3P separates stationary stability from
adaptation; RG5's recovery improvement belongs to its particular configuration.
Neither `b_cap` nor critic anchoring should receive a universal works/doesn't-work
label from these rows. The frozen control's failure is expected supporting
evidence for recovery, not another failed technique.

### Required LR-free history and calibration cards

Import at least these additional lineages and issues from #155:

| Evidence family | What the card must retain | Why Forge needs it |
| --- | --- | --- |
| LR-free quick/native and custom-host lineages | Each exact package's quick, custom, native and continuation outcomes; initialization/scoring policy | Calibrate the proposed smoke set against bars4, mode_hold, intensity2, unequal_mass and native failures, rather than assume short hosts predict them |
| `structural100` 14k extension | Verified 7k prefix parity, full 14k metrics/holdout, original package and fixture | Demonstrates how to extend the evaluator budget without changing the learner and how ordinary short results can remain insufficient |
| Fresh-reservoir `row-em-renew` | Native 3/3; 4.4–4.8× wall time; all22 10 PASS / 4 FAIL / 8 raw ERROR; `_sigma_intrinsic_scale` parity refusal | Preserve a useful sampling diagnostic, classify it as `sampling_only_patch`, expose cost, and normalize unsupported custom hosts to BLOCKED |
| D-tracking / floor changes | Distinct floor constants, package hashes, live and EMA results at matching checkpoints | Keep structural changes separate from tuned floors, preserve failed terminal streaks, and record the view's scoring policy |

Pinned sources: [LR-free overview](https://github.com/HyperGAN/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/README.md),
[structural continuation](https://github.com/HyperGAN/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/structural100/README.md),
[sampler results and cost](https://github.com/HyperGAN/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/README.md),
[all22 receipt](https://github.com/HyperGAN/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/all22-summary.json),
[D-tracking follow-up](https://github.com/HyperGAN/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/row-em-renew/bd-pair-sigma-release-plan.md).

The review additionally reports dt075's grid/staggered 14k failures after ladder
release, EMA .995 D-tracking 3/3, and sub-0.03σ centre-error sensitivity. Their
exact later receipts must be bound during import: these particular claims were
not located in the pinned #155 snapshot. Preserve them as review-supplied evidence
and unresolved import items, not as results of a nearby similarly named package.
They remain explicit requirements for a future calibration or scoring-policy
review that uses those claims. The user confirms that this research is still in
development; the framework must support later imports without inventing receipts.

Use these and other archived positive/negative references to calibrate gate
selection. Estimate how often an early rejection would hide a later useful
candidate. Where historical evidence is insufficient, run a small, budgeted
calibration set with fixed declared fixtures, changing tasks or mechanisms rather
than seeds. A separate, explicitly budgeted **calibration lane** may run deeper
diagnostics on selected rejected candidates to detect an overaggressive screen.
This is the sole declared exception to normal prerequisite scheduling: those
results cannot bypass failed gates or retroactively qualify that candidate under
the same protocol. They can motivate a future protocol revision. Do not silently
turn every rejection into a full-suite run. A gate is a compute-saving heuristic,
not proof of production readiness or a reason to stop improving the test suite.

## 9. Framework delivery and future adoption

| Phase | Deliverable | Acceptance criteria |
| --- | --- | --- |
| **A. Inventory and history** | Machine-readable catalog, mapped experiment cards, initial compiled memory | Cover initial sources and the pinned #155 LR-free tree, including frozen fixtures/package hashes; later unbound claims remain explicit import gaps; compilation needs no GPU |
| **B. Shared API, lifecycle and views** | Thin public-API integration, central extension contract, learned-MoG defaults, schemas, scaffolding, reducer and read-only commands | One API across ideas/tasks; explicit cloud exceptions and capability blockers; config-only retiering reuses evidence; future monotonicity remains inactive |
| **C. Queue and gated execution** | Shared MoG-capable public training path; adapt existing runner and #155 pool/submit/ledger contracts; provisional smoke profile, queue, multi-GPU drain, logs | MoG sampling/update/checkpoint parity; no copied experiment loops; required failures/blockers prevent later tiers; deduplicated execution and complete receipts |
| **D. Calibration machinery and operational pilot** | Historical replay, bounded diagnostic registrations, report-bound calibration and a physical multi-GPU pilot | Preserve smoke/reference matrices, false accepts/rejects, unknown/blocked denominators and costs; verify cancellation, recovery, deduplication and retiering. Finding a scientifically accepted profile remains a future adoption prerequisite; no screening seed sweeps |
| **E. Agent workflow and migration support** | Root experimentation guide, agent entrypoints, consumer dispositions, domain views and public-default robustness contract | A new engineer/agent can follow the guide without chat history; one small declaration joins comparisons; outcomes update memory. Future ownership transfers preserve requests/rollback; public-default claims require a finished candidate and registered robustness stage |

Implementation should include meaningful tests for:

- Lower-tier failures, missing evidence, and infrastructure errors blocking higher
  tiers; diagnostic failures not doing so; partially qualified rows staying visible.
- Changed source/config/protocol invalidating reuse, exact duplicate detection,
  attempt locking, atomic writes, interrupted-run recovery, and budget enforcement.
- Moving an unchanged task between tiers by editing one view file, with zero
  training launches; old campaigns stay pinned and dependencies remain valid.
- Shared learned-MoG defaults, explicit sigma-zero cloud exceptions, and distinct
  resolved manifests; no silent loss of prior learning, noise, or candidate hooks.
- A new formulation variable registered once reaches compatible task families,
  optimizers/checkpoints/RNG handling and receipts; unsupported hosts block before
  training rather than dropping it or requiring copied per-experiment loops.
- Public trainer/component execution and parity for supported paths; archived
  private hooks cannot stand in for shipped-API qualification.
- Concurrent submissions from separate worktrees sharing one queue, correct GPU
  assignments, memory-aware dispatch, lease recovery without duplicate execution,
  and complete centralized logs without interleaved/corrupted event records.
- Requested tier caps and campaign-budget exhaustion preventing new launches;
  timeout/cancel killing all child processes; seed-only proposals rejected before
  training except for the registered finished-candidate robustness stage; expected
  calibration exceptions remaining outside qualification.
- Component-scoped RNG isolation under extra draws, evaluation, architecture
  changes, worker placement, and restart; changing seed/initializer/bindings blocks
  reuse; shared components keep matching draws despite changed unrelated shapes.
- Clock-free next-update and common-prefix parity under changed external clocks
  and evaluation horizons; hidden ladder releases detected by longer continuation.
- Applicability refusals retaining raw errors, becoming reasoned BLOCKED entries,
  and preventing full qualification without counting as scientific failures.
- Live/EMA and noisy/clean or calibrated sampling evidence remaining distinct;
  mechanism class and cost shown next to results; no retrospective scoring switch.
- A frozen promotion candidate running only its preregistered seed set, with all
  per-seed evidence and outcomes retained and no cross-seed reuse or in-stage tuning.
- Regrading failures hidden by a best checkpoint, EMA, missing terminal checks,
  or a passing standard hold followed by a failing extension.
- Deterministic map/reduce, complete inventory coverage, conflict reporting,
  negative-control handling, and no accidental merging across configurations.
- Adding a new view over existing evidence without rerunning compatible tasks;
  a view with unknown requirements cannot invent passes or rank missing metrics.
- A fresh-checkout walkthrough of `EXPERIMENTATION.md`: find prior work, scaffold
  an idea, plan and submit a bounded smoke request, follow logs, inspect its board,
  and publish a readout using only documented instructions. Validate no-GPU steps
  with fake workers and the real submission path during the bounded pilot.

Expose one stable campaign log and one per-attempt log, all unbuffered, with
timestamps, candidate, tier, task, metric/verdict, elapsed cost, and next action:

```sh
tail -F runs/forge/events.jsonl
tail -F runs/forge/<campaign-id>/progress.jsonl
tail -F runs/forge/<campaign-id>/<attempt-id>/run.log
```

Verify scheduling first with serial fake workers, then require bounded multi-GPU
draining in Phase C. Never dispatch a candidate's expensive downstream tiers
speculatively. Parallelize historical mapping and candidate implementation with
disjoint files, then reduce centrally.

The first usable milestone is **inventory + memory + read-only goal views**.
The next is **one command that tries an idea through cheap gates and publishes a
useful readout even when it fails**. Broader production tracks can then use the
same lifecycle and evidence without forcing every new idea through every study
the repository has ever accumulated.

## 10. Migration checklist and subagent workflow

The framework checklist is complete, with implementation evidence in
[the migration status](../reports/forge/MIGRATION.md) and
[final readiness audit](../reports/forge/FRAMEWORK_READINESS_AUDIT.md). The
separate future-research checklist retains scientific calibration and adoption
work; none is claimed complete by delivering the engine. Keep existing running
experiments and historical evidence intact throughout any later transition.

Continue in `codex/tiered-experiment-qualification`, branched from `develop`,
with PR #221 as the sole implementation PR. Refresh against `develop` as needed
while preserving the historical source pins. Subagents contribute disjoint files
in this same implementation worktree; the coordinator owns commits, integration
and pushes. The experiment engine may support independent idea worktrees, but
building Forge itself stays in this one feature branch/PR.

### Framework migration checklist — complete

- [x] **Freeze the migration inputs.** Confirm the existing feature branch/PR
  targets `develop`; pin the package baseline, #155 source
  snapshot, report roots, and later follow-up receipts; record missing sources.
  Inventory existing launchers/queues and any active jobs before changing ownership.
- [x] **Agree on the shared contracts.** Define TaskSpec, idea/candidate records,
  FormulationContext/capabilities, attempt receipts, and view placement. Separate
  execution/evaluation fingerprints from view-policy revisions. Assign file owners
  and freeze these interfaces before parallel implementation.
- [x] **Map the history.** Classify all scoped files; import existing experiment
  lineages, including successes, failures, raw errors, explicit blockers and
  incomplete work. Bind prior regime, fixture, package, RNG and sampling law.
  Keep dt075/EMA/sensitivity import gaps visible until exact evidence is available.
- [x] **Build shared public-API support.** Add the required MoG training and
  formulation hooks to the public package/runtime path. Use existing public
  components where appropriate; centralize Forge's bindings and document API
  additions, changed variables, supported hosts and checkpoint migration.
- [x] **Establish prior defaults and exceptions.** Resolve new tasks to learned
  MoG priors in one defaults/factory path. Explicitly classify sigma-zero cloud
  tasks. Verify gradients, sampler/RNG behavior, state restoration, and mechanism
  activation, including the A2 type-sensitive behavior. Preserve old identities.
- [x] **Consolidate experiment adapters.** Port the initial toy/native/hold
  families onto the shared API. Extract duplicated setup, training, sampling,
  checkpoint, logging and result logic. Retain thin compatibility entrypoints
  where useful; keep frozen historical source copies outside active implementation.
- [x] **Declare tasks and tier views.** Reference shared task/evaluator definitions
  and place tasks in versioned view configs. Demonstrate a tier move without code
  changes or training. Validate dependencies and freeze the provisional campaign.
- [x] **Compile memory and comparisons.** Implement deterministic map/reduce,
  conflict/coverage checks, the single history file, boards and recall. Distinguish
  MoG/cloud cohorts, API support, live/EMA, cost and mechanism class in the output.
- [x] **Unify execution and queue state.** Adapt the existing grid runner and #155
  submit/pool/ledger behavior to the shared request/receipt contract. Add atomic
  ownership/budget claims, worker recovery, central logs and durable results.
  Verify with fake workers before using GPUs.
- [x] **Build calibration and promotion machinery.** Replay bound historical
  evidence, register bounded diagnostic lanes and bind acceptance to frozen
  criteria and compatible receipts. Keep unknown/blocked outcomes and costs
  visible. Implement the fixed finished-candidate robustness contract; candidate
  success and robustness execution belong to future research.
- [x] **Verify physical multi-GPU operation.** Run a bounded campaign through
  multi-GPU draining; test cancellation, restart, retiering and memory updates.
  The [physical v3 pilot](../reports/forge/PHYSICAL_GPU_PILOT_READOUT.md) completed
  these operational checks with explicit resource ownership and preserved receipts.
- [x] **Publish the root guide and entrypoints.** Write `EXPERIMENTATION.md` with
  a read-first agent brief, working quickstart, command examples, extension paths,
  and result/readout requirements. Add prominent pointers in root `AGENTS.md` and
  `README.md`; other agent-specific files should link to the same guide. Validate
  the fresh-checkout walkthrough and make implementation readiness explicit.
- [x] **Verify the agent workflow and consumer dispositions.** Have new
  engineers/agents follow the root
  guide to read memory, add an idea, run it within budget, and record its outcome.
  The [fresh-checkout walkthrough](../reports/forge/FRESH_CHECKOUT.md) passed;
  the [consumer dispositions](../reports/forge/legacy-consumers.json) retain
  uncovered domains and historical replay. No covered pending local request needs
  transfer. Default scientific adoption remains a separate decision.

### Future research and adoption TODOs — outside framework delivery

- [ ] **Calibrate a useful screening profile.** Complete the required fixed-fixture
  references under a justified protocol, separating historical clouds from current
  MoG qualification. Publish false accepts/rejects, blockers, costs and continuation
  findings. The [current control readout](../reports/forge/CONTROL_MODE_READOUT.md)
  shows all three lineages failing smoke; this exact profile cannot meet both its
  positive-reference minimum and zero false rejections. Declare a new profile
  rather than filling this one solely for adoption.
- [ ] **Import ongoing research when available.** Bind exact upstream receipts and
  public API changes before using them in calibration or scoring-policy conclusions.
  Keep unavailable claims explicit; historical passes retain their original law.
- [ ] **Adopt a calibrated default workflow and transfer actual consumers.**
  Reserve resources under one owner; reconcile pending requests before transferring
  them from an old scheduler. Preserve rollback receipts. Retire duplicated active
  launchers only after their covered behavior and consumers are migrated.
- [ ] **Qualify a production candidate.** Pass its declared quality/endurance gates,
  freeze scoring and preregister robustness before a public-default claim. Run only
  the finished candidate's declared seed set; framework completion creates no
  promotion claim or need for speculative seed runs.

### Parallel work, with explicit handoffs

Use one coordinator and up to three implementation subagents in the same
feature-branch worktree, with disjoint file ownership. The coordinator owns shared
contracts, Git operations, integration, the migration
checklist and final readout. Each assignment specifies input revisions,
dependencies, owned paths, expected artifacts, checks, and a compute allowance
(zero training by default). Agents do not concurrently edit generated shared
boards/history or each other's source files.

| Wave | Parallel work packages | Join condition |
| --- | --- | --- |
| 0 — coordinator | Pin inputs, define contracts/path ownership and plan the bounded calibration budget | All agents have the same schema/API assumptions and immutable input refs |
| 1 — foundations | **History:** catalog/importers and normalized cards. **API:** public MoG/formulation support, central context/bindings and parity checks. **Views:** task declarations, tier config and reducer contracts | Schemas align; MoG/cloud support and capability blockers are explicit; history gaps are recorded |
| 2 — integration | **Adapters:** port initial task families using the shared public API. **Execution:** queue/drain/resources/recovery/logging. **Knowledge:** compiler, boards, recall, scaffolding and root guide draft | Representative requests produce compatible receipts end to end; fake-worker tests pass; no divergent training implementations |
| 3 — verification | Independent review of parity/RNG/checkpoints, import fidelity and retiering; coordinator runs the bounded physical pilot and verifies calibration/promotion machinery | Framework checks and pilot pass; metrics, costs, failures/blockers and recommendations are published; scientific profiles remain provisional |
| 4 — onboarding | Coordinator records consumer dispositions; agents finish root guide/entrypoint links and scoped migrations | New engineer/agent completes the documented workflow; historical evidence preserved; future handoff/rollback requirements documented |

Commit small dependency-ordered changes to the same feature branch and PR.
Task adapters consume the agreed API; if a formulation needs a new variable,
coordinate one public API/central binding change, then update dependent adapters.
If an interface must change between waves, publish the delta before dependent
work resumes. Maintain a single migration status/readout with completed TODOs,
remaining blockers, validation results and links to each work package.

## 11. Engineer and agent entrypoint: EXPERIMENTATION.md

Use **`EXPERIMENTATION.md` in the repository root** as the single operational
guide. Its title should be **"Experimentation with ParticleGAN Forge — read
before running ideas"**, followed by a prominent brief for both engineers and
agents. A new contributor should be able to act from this guide without reading
this full design plan, prior chats, or an unrelated experiment's launcher.

The opening brief should immediately establish the workflow: read the compiled
memory and relevant leaderboard; state a hypothesis and changed factors; use the
shared public API and learned MoG defaults; declare cloud/API exceptions; run the
cheapest eligible gate within budget; record failures as well as passes; publish
metrics, a comparison and a recommendation. Explain the fixed-seed screening
policy and the distinct preregistered promotion stage. Point to the current view
and protocol definitions for exact requirements rather than restating thresholds.

Make it discoverable through the files agents and people actually read:

- Put an **"Experiment work: read EXPERIMENTATION.md first"** pointer near the
  top of root `AGENTS.md`, with links to the guide and compiled experiment memory.
  Preserve the existing repo instructions; make the full workflow easy to find.
- Add a visible **"Running experiments / proposing ideas"** link in `README.md`.
  If other agent-specific entrypoints are introduced, give them the same pointer.
  Do not maintain competing copies of the workflow in each agent file.
- Have new-idea scaffolding and CLI help/status output point to the guide and
  print the declaration, log and report paths relevant to the command.

The guide must include:

| Reader's question | Required operational content |
| --- | --- |
| What can I run in this checkout? | Current implementation/readiness status, environment setup, repo-root working directory, dependencies, and available CPU/GPU paths |
| Has this been tried? | How to use recall, compiled memory, lineage and leaderboard filters; when an exact result is reusable and what makes a retry informative |
| How do I add an idea? | A minimal declaration and small public-API example; choose a parent, state the hypothesis/delta, use shared defaults, and declare API/prior exceptions |
| How do I try it cheaply? | A copy-paste path from new → plan → enqueue Tier 1 → inspect status/logs → board → readout/compile, with explicit budget and expected output files |
| How do I operate the queue? | Submit versus drain, selecting available GPUs, coordinator ownership, central log filters, cancellation/recovery, and how to avoid duplicate jobs |
| What do I change for a new test or variable? | The shared task adapter and authoritative public API/capability definition, central Forge binding, required `api_changes` declaration, and parity checks |
| How do I move a test between tiers? | Edit one view assignment, validate/recompute the board, preserve pinned campaigns, and submit separately if new evidence is needed |
| What does the result mean? | PASS/FAIL versus ERROR/BLOCKED/INCOMPLETE, live/EMA and MoG/cloud scope, raw evidence locations, costs, and next eligible work |
| When is the experiment finished? | A readout template covering hypothesis, exact revision, tested scope, metrics/cost, comparison, failure/blocker explanation, recommendation and memory update |

Keep the first runnable example small and self-contained; put advanced
continuation, clock-free claims, promotion, and migration details behind links.
Use the same examples in CLI checks and docs where practical so they do not
drift. API/CLI changes must update affected examples and extension guidance in
the same change. Defaults, schemas and gate values remain authoritative in code
and config; the guide teaches how to use them.

Draft the guide as the commands become available, clearly labeling unavailable
steps until implemented. Turn on the read-first agent pointers with a usable
guide, not an instruction to execute planned commands that do not exist. Root
documentation and its fresh-checkout walkthrough are adoption deliverables,
not an optional follow-up after the engine ships.
