# Experimentation with ParticleGAN Forge — read before running ideas

Read this before proposing or running an experiment. Forge records ideas,
checks cheap prerequisites before spending on larger tests, and compares exact
candidate revisions through goal-specific leaderboards. Terms such as *view*,
*tier* and *held test*, and the family acronyms (BISPF, BCAP, K3P, KA2, E22, …), are
defined in the [research glossary](RESEARCH_GLOSSARY.md).

## Quickstart

Run every command from the repository root in the project's Python environment
(`python -m pip install -e '.[experiments,dev]'`). Every step except `run`
launches no training. `discriminator_stability` is the default view (goal).

```bash
# 1. Recall: what has been tried, what failed, what is in flight (no training)
python -m experiments.forge recall --query "<topic>" --goal discriminator_stability

# 2. New: scaffold one idea from a parent idea, plus its companion study.
#    Writes configs/forge/ideas/<id>.json and configs/forge/studies/<id>-study.json;
#    edit changed_factors, mechanism_rationale and recipe_overrides before planning.
python -m experiments.forge new --id my-idea --parent <parent-idea-id> \
  --goal discriminator_stability --hypothesis "<one falsifiable sentence>"

# 3. Plan: exact tasks, evidence reuse and worst-case cost; writes nothing
python -m experiments.forge plan my-idea --study my-idea-study --show-boundaries

# 4. Run: freeze, enqueue and drain through the declared tier cap
#    (--gpus 0,1 for physical GPUs, --gpus cpu for the CPU cohort)
python -m experiments.forge run my-idea --study my-idea-study --gpus 0
python -m experiments.forge queue            # active/blocked requests, costs, log paths
python -m experiments.forge logs --follow    # tail the timestamped event stream

# 5. Board: regrade the goal over compatible evidence (no launches)
python -m experiments.forge board --goal discriminator_stability --candidate my-idea

# 6. Readout: conclude the study; required before reusing or modifying the candidate
python -m experiments.forge readout my-idea --study my-idea-study \
  --conclusion "..." --comparison "..." --next-action "..."
```

Parent ids are the file names under `configs/forge/ideas/` (for example
`bispf-optimized-v1`, `ka2` or `k3p`). Each command's `--help` lists its
options. `python -m experiments.forge --help` lists the other commands
(`probe`, `study`, `enqueue`, `drain`, `techniques`, `compile`, …), which the
sections below describe.

## Read first

- The [current technique leaderboard](reports/forge/technique-inventory.md) is the
  single current status page. Stretch and held experiments appear in its dedicated
  `## Stretch experiments` table by tier as well as in the `## Full-suite results — failures included`
  table; the required qualification subset is separate.
- The [problem list](reports/forge/problems.md) explains each outstanding question:
  what it tests, what fails and which configurations have passed it.
- The [compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md) is a
  reference index. Query it with `recall` instead of reading it in full.
- For a new research question or host, follow the
  [experiment creation guide](docs/forge-new-experiment.md). It has worked ring and
  joint BiGAN examples, scorer controls, registration and artifact publication.
- To drain missing Forge cache cells in parallel on Google Colab GPUs (with automatic
  GPU right-sizing, decentralized branch-merge catch-up, crash-proof VM teardown, and
  local Hugging Face publication), see the
  [Colab Forge catch-up guide](docs/forge-colab-catchup.md).
- To refresh leaderboards across GitHub, Hugging Face Dataset/Space, and GitHub Pages
  (including Pareto energy benchmarks and clean-commit family promotions), see the
  [Leaderboard Refresh Runbook](docs/leaderboard-refresh.md) and
  [`ops/refresh_leaderboard.py`](ops/refresh_leaderboard.py).

Register ordinary new questions as required Tier 1 tasks in the existing
`discriminator_stability` view, or assign frontier/extra-credit/optional questions
with `importance: "stretch"` in their appropriate tier (`1`–`4`). Revise that view and check its full reservation
budget. Create a separate view only for an explicitly different claim or
diagnostic scope. Preserving an old denominator alone is not a reason for a new
view. Historical receipts and pinned view revisions keep their original results.
Keep bulk execution logs out of the repository, as [`AGENTS.md`](AGENTS.md) requires.

**`particlegan_forge`** is a separate, namespaced package that ships in the wheel.
Its purpose is to expose Forge's task, tier and contract vocabulary (`TaskSpec`,
`TierView`, named RNG streams, training-GIF rendering) to code outside this
repository. The experiment workflow above runs through `python -m experiments.forge`.
Treat `particlegan_forge` as the external tier API, and do not mix its registries
with the repository's `configs/forge/` declarations.

## Readiness and scope

**The framework is ready to use.** The
[final readiness audit](reports/forge/FRAMEWORK_READINESS_AUDIT.md) verifies the
requested engineering scope. [Full CI](https://github.com/HyperGAN/ParticleGAN/actions/runs/36677816874)
passed 1,807 tests and 18 subtests; the [physical two-GPU pilot](reports/forge/PHYSICAL_GPU_PILOT_READOUT.md)
and [fresh-checkout walkthrough](reports/forge/FRESH_CHECKOUT.md) verified the
queue, recovery, cancellation, reuse, logs and readout workflow. Use the public
API and learned MoG defaults; particle-cloud exceptions must be explicit in the
task. Do not run seed-only experiments.

**Screening profiles remain provisional; scientific default adoption requires
calibration.** Framework delivery does not require finding a winning formulation.
The [current control readout](reports/forge/CONTROL_MODE_READOUT.md) records 7/57
measured cells, 50 unknown, one true rejection and no complete positive reference.
All three declared lineages fail smoke, so this exact profile cannot meet both
the frozen positive-reference minimum and zero false rejections. Stop filling it
solely for adoption. A new justified screen/profile and its bounded calibration
are required; the controls' independent references and false-reject rate remain
unknown. [Migration status](reports/forge/MIGRATION.md) retains the earlier
studies, metrics, costs, source cohorts and future research/adoption work.

The [2026-09-30 bounded reference study](reports/forge/SCIENTIFIC_CALIBRATION_20260930_READOUT.md)
adds one substantive training-output-noise-removal lineage while preserving all
three host-profile smoke gates, 16 independent references and frozen criteria.
Its sole full 7k native diagnostic failed for 93.747 new paid seconds. Shape
improved but terminal quality and sustained coverage failed. The new profile has
8/76 measured cells and is also infeasible: every lineage is smoke-negative or
reference-negative. Stop filling both failed profiles; no robustness or promotion
run is eligible. Original evidence and the concluded failed revision remain in
compiled memory.

Task declarations describe intended coverage. Preflight and execution check the
public API, formulation, host and frozen source; unsupported capabilities remain
`BLOCKED` in the required denominator. New ordinary requests finish all runnable
jobs in the current tier before applying required failures to higher tiers.
Checkpoint/data and explicit gate dependencies still require passing parent
evidence; unsupported tasks spend nothing and independent peers can run.
Selected deeper comparisons require the separately registered calibration
lane described below and confer no ordinary qualification.

Historical cards preserve successes, failures, raw errors, negative controls and
missing evidence under their original protocols. [Initial](reports/forge/calibration/initial.md)
and [alternate](reports/forge/calibration/quick-discriminator.md) historical
screens also remain blocked under the [frozen criteria](configs/forge/calibration/criteria-v1.json).
Historical cloud, noisy served-model or EMA passes cannot fill a current MoG/clean-live
cell. The [newer #155 archive](reports/forge/supplemental/pr155-current-archive-v1/README.md)
adds 31 native arms and six context families, preserving noisy/clean scoring,
failed ideas, host refusals and OOM attempts in the compiled memory. The
[newer #155 audit](reports/forge/calibration/pr155-current-followup-source-audit.md)
still leaves the exact dt075 14k, EMA .995 D-tracking and centre-sensitivity
experimental receipts [unbound](reports/forge/import-gaps.json); a later written
sensitivity assertion is separately preserved. That research is still developing;
import its receipts when available using the existing framework.

E22/KA2 is available through the [public API](docs/api.md), including the E22
preset and `E22Policy`. The inspected positives in the
[historical integration checklist](reports/forge/UPSTREAM_E22_COMPATIBILITY.md)
use particle clouds and noisy state-selected serving. Preserve their frozen
source, formulation, prior and sampling identities; they do not supply evidence
for a different MoG or clean-live task. MoG support, named streams, external
budgets and complete policy checkpoints remain required integration contracts.

Run commands from the repository root in the project's Python environment.
The [registered formulation comparison](reports/forge/studies/FORMULATION_COMPARISON_V1.md)
compares fixed R1/R2, BCap and full released v0.7 GAN v3 with K3P on declared
native hosts. `Recipe.reg_arm` exposes the two fixed L2/autograd penalties through
the shared public trainer. Paired noisy artifacts retain the required clean/live
grade and grant no extra reference credit. Optional calibration `diagnostic_tasks`
bind separate host identities and report their costs without entering smoke or
reference decisions. Existing profiles without this field retain their semantics.

The [completed formulation comparison](reports/forge/FORMULATION_COMPARISON_READOUT.md)
records all five full 7k native runs as FAIL for 408.765 new paid seconds.
Independent audits pass, and the K3P control exactly reproduces the previous
clean-MoG trajectory and named training streams. BCap improves centering but
contracts further; R1/R2 improves shape but loses centering/coverage. Released
GAN v3's matched MoG adaptation improves shape but still fails the full gates;
its separate named-cloud host also fails and supplies no MoG reference credit.
The profile has 4/95 ordinary calibration cells measured (91 unknown), plus
1/5 separate diagnostic cells (4 unknown); acceptance remains blocked.
Stop these exact experimental revisions. Inspect saved critic gradients and
per-mode moments before another supported, bounded training hypothesis. No
automatic matrix filling, tuning, seed study, continuation or promotion follows.

The history, recall, compile, validate, and board commands launch no training.
The alternate `ops/formulaic_coverage.py` executor is retired after the
[2026-11-08 scientific audit](reports/gpt-audit-2026-11-08.md). It changed
recipes and host update rules outside ordinary admission. Its old receipts
remain source-bound diagnostic evidence; a reporting refresh must not repeat
them or use them to qualify a corrected implementation. Use the frozen-study
workflow above for new measurements. An unsupported public adapter remains
BLOCKED, and a correctly measured numerical FAIL remains FAIL. The ownership,
sampler, architecture and continuation boundaries in [AGENTS.md](AGENTS.md)
apply to every execution path, including coverage diagnostics.

Recall includes source-bound concluded configuration and policy publications as
non-qualifying summaries. It returns compact source identity pins and a file
count by default. Add `recall --full-provenance` when the complete source-file
hash manifest is needed; stored records retain their full provenance.
[Research memory freshness](docs/forge-research-memory.md)
is visible in plans; `forge compile --check` checks it without artifact hydration.
Use `forge compile --summaries-only` to refresh recall after a reporting or source
merge while preserving published qualifications and telemetry. Inspect missing
originals with the [artifact resolver](docs/forge-artifact-resolver.md); absence
of an archive blocks saved-state analysis rather than authorizing a rerun.

```sh
python -m experiments.forge validate
python -m experiments.forge recall --goal discriminator_stability --query "critic anchor"
python -m experiments.forge board --goal discriminator_stability
```

## Add one idea

Use an existing declaration as a parent. The scaffold inherits its formulation
settings and supplies the standard fields; edit the hypothesis, changed factors,
and actual mechanism/configuration before submission.

```sh
python -m experiments.forge new --id critic-anchor-v2 --parent k3p \
  --goal discriminator_stability \
  --hypothesis "A stronger critic anchor preserves terminal quality during hold"
```

New scaffolds write a reusable schema-v3 candidate and a separate draft study
in `configs/forge/studies/critic-anchor-v2-study.json`. Follow the
[candidate/task/study guide](docs/forge-studies.md) and its complete small example.
Edit the candidate's training mechanism and `changed_factors`; put the hypothesis,
control selection, original evidence identities, predictions/falsifiers, finite
campaign budget and stopping rules in the study. Tasks own the problem, initial prior distribution/capacity,
architecture, initialization, training allowance, sampling and sustained gates.
The recipe owns prior learning/freezing and prior regularization.
Planning generates technical hashes and execution bindings from these declarations;
users do not copy hashes or task/source bindings into a recipe. Mark the reviewed
study ready. Drafts and missing studies block admission before spend. Saved v1/v2
cards, embedded contracts, requests and receipts retain their original semantics.
A legacy embedded contract cannot be removed in place to obtain admission.

```sh
python -m experiments.forge plan critic-anchor-v2 --study critic-anchor-v2-study
# Reuse the same recipe in another explicitly declared question.
python -m experiments.forge study new --id critic-anchor-hold \
  --candidate critic-anchor-v2 --control k3p --view discriminator_stability
```

Enqueue freezes the study's generated source/runtime/task bindings. A changed
binding or declaration requires a new study ID; identical task jobs can reuse
compatible evidence. Paid retries and full reservations count against the same
scientific round across studies and campaigns in the shared queue. Readout names
the study and reports its predictions separately from scientific qualification.

Edit `configs/forge/ideas/critic-anchor-v2.json`. Describe the substantive
change in `changed_factors`, cite relevant prior work, and choose
`mechanism_class`: `structural`, `floor_constant`, or `sampling_only_patch`.
Keep floor/constant tuning visible separately from structural changes. A
sampling-only improvement must describe its training and public sampling laws.

Use `recipe_overrides` for existing package settings. Register new reusable
capabilities through the [shared formulation API](experiments/forge/api.py);
declare its extension and required capabilities on the idea. Avoid copying
another candidate's launcher or training loop. Missing host support must produce
an explicit blocker rather than silently dropping the mechanism.

New ordinary tasks declare the initial MoG distribution and capacity. Recipes
default to `prior_update: "learned"`; learning or freezing locations is an
explicit recipe decision. Width and uniform mixture weights remain fixed. Particle-cloud hosts
must explicitly declare `kind: "particle_cloud"`, `sigma: 0`, and an exception
reason. Historical sigma-zero receipts retain their original identity.

New tasks declare `execution.prior_contract: "recipe_owned_v1"` and define
`execution.prior` with `kind`, `sigma` and `standardize`; they omit `learnable`.
The initial prior cannot come from the candidate, preset or API. Legacy tasks
without this contract retain their explicit task-owned `learnable` policy and
original host objectives. Their archived evidence is not regraded.
`kind: "mog"` selects the public `MoGParticlePrior` code path, while
`kind: "particle_cloud"` selects `ParticlePrior`. A zero-width MoG and a particle
cloud can describe the same distribution but retain separate implementation
identities. Ordinary MoG tasks still require positive sigma. Particle clouds also
require `standardize: false` and their explicit exception reason. Parameter-only
controls declare `prior_applicability: "not_sampled"` in `execution`.

The [field ownership contract](docs/forge-field-boundaries.md) separates task
conditions, technique mechanisms, hyperparameters and the comparison protocol.
Each task also declares `execution.initializer`; fixed host fixtures and native
component policies remain its more specific initialization rules. An explicit
candidate initializer is a compatibility requirement and cannot replace the
task's policy. Native tasks declare their model and training resources, while
profiled and behavioral hosts retain their frozen host definitions and sources.

Inspect effective values and their owners before submission:

```sh
python -m experiments.forge plan k3p --device cpu --show-boundaries
```

Planning and adapters use the same task binding. Execution receipts record each
field's owner and source; inactive historical host optimizer settings remain
labelled provenance. Configuration searches preserve the base technique's
mechanisms and reject axes that are inactive or delegated on every tuning task.
Changing a mechanism requires a structural idea. Family labels describe lineage;
task priors and sampling laws remain distinct scientific comparison cohorts.
These declaration changes create new execution identities. Saved receipts and
qualification outcomes retain their original bindings and are not regraded.

Do not create seed-only ideas. Screening uses one fixed protocol seed with named,
isolated streams. Changing an initializer, sampling law, or stream binding changes
the evidence identity. A future public-default promotion requires the plan's
separately registered robustness stage after the candidate is frozen; a passing
screen is not a public-default promotion.

New public-API toy comparisons also default to seed `0`. Construct fixtures in
the shared isolated RNG scope and initialize learned networks and priors through
`particlegan.init`; vector priors use its R2 table instead of constructor draws.
The public runner scopes the entire execution to the fixed seed, uses one Torch
thread, requires deterministic algorithms, disables cuDNN benchmarking and TF32,
and restores the caller's RNG and execution settings after the run.
Critic and generator target batches use separate checkpointed streams and are
drawn once per update, so changing the loss cannot shift later critic batches.
Receipts expose initial model hashes and the vector batch-sequence digest.
Keep architecture, prior, data law, sampling, budget and scoring cadence fixed
within a task while comparing declared trainer changes. Use one whole global
configuration across tasks. Explicit fixed controls and initialization-profile
diagnostics keep their distinct cohorts and original evidence.

The [bounded reproducibility audit](reports/forge/reproducibility/README.md)
checks exact same-seed repeat states and matched initial models/batches across
two public trainers, with actual-training GIFs. It supplies software evidence,
not full-budget quality qualification. The new vector checkpoint and policy
search schema versions prevent old stream/seed contracts being reused silently;
reproduce archived studies from their pinned original source.

## Inspect cost, then submit

```sh
python -m experiments.forge plan critic-anchor-v2 --study critic-anchor-v2-study
python -m experiments.forge enqueue critic-anchor-v2 --study critic-anchor-v2-study
python -m experiments.forge queue
```

`plan` shows resolved tasks, task groups, reuse, blockers, and reserved worst-case
wall time. `enqueue` freezes the source request and returns without starting
training. An identical scientific request attaches to existing work or compatible
evidence. Changing only prose does not justify rerunning the same experiment.

New ordinary requests freeze `execution_policy` with `schema_version: 1` and
`mode: "complete_current_tier"`. A FAIL, INCOMPLETE, INVALID or task-local BLOCKED
result does not stop independent jobs in that tier. After runnable work finishes,
any required non-PASS or unresolved required dependency blocks the next tier.
The queue can run independent jobs concurrently, in frozen assignment order,
subject to full reservations, host/device capacity, pause and cancellation.
Known unsupported execution groups create no paid attempt or synthetic receipt.
Budget exhaustion can leave cells unmeasured; it never relaxes a full reservation.
Plans, queue summaries and submitted events expose the policy for log readers.
Saved requests without this field retain their original fail-fast behavior;
registered calibration and promotion lanes retain their frozen stopping rules.
Scheduling policy changes request identity, but does not enter scientific job
compatibility or grant qualification. Reuse still requires the exact recipe,
task, source, prior, initialization, sampling and runtime bindings.

Use the [generated tier guide](reports/forge/EXPERIMENTS_BY_TIER.md#discriminator_stability)
and [registered view](configs/forge/views/discriminator_stability.json) for the
current tasks, roles and allowances. Revision 14 declares 11 required Tier 1
tasks whose individual timeout allowances sum to 4,140 seconds. This sum is not
an execution reservation: the read-only plan resolves grouped execution, reuse,
unsupported hosts and candidate-specific bindings before admission.

```sh
python -m experiments.forge plan k3p --through-tier 1 --device cpu
```

For revision 14, this CPU K3P plan contains 12 execution groups and a
4,440-second worst-case allowance, including the 300-second clock audit
diagnostic. Review the actual plan for the intended candidate and device.

Revision 15 adds optional Tier 1 [NVFP4 and MXFP8 training toys](reports/forge/precision-stretch/README.md)
and their matched FP32 diagnostic. Required counts remain unchanged. Each new
task reserves 300 seconds; the equivalent CPU K3P plan now reserves 5,340
seconds. Existing campaigns retain their original caps. These are explicit
Blackwell-format software references; native kernel and higher-order autograd
parity remain unverified. Full-budget family results are initially unmeasured.

Historical revision 8 had six required Tier 1 tasks and a 2,220-second
required-task allowance: the 120-second scalar Gaussian can-it-pass smoke,
three 300-second behavior tasks, the 300-second ring acquisition and the
900-second confirmed word smoke. The registered `tier1-word-smoke-v1` campaign
retains its revision-8 campaign and candidate caps of 2,220 seconds; it does not
budget the expanded current Tier 1. Its Gaussian task requires any passing
scheduled state plus independent same-state confirmation, and completes all
1,000 updates. The continuing Gaussian retention/target-shift task is required
Tier 2 with its own passing smoke checkpoint dependency and 600-second allowance.
The five-word Tier 1 smoke also completes all 20,001 updates, with 24 independent
same-state confirmation pairs and the unchanged full generation/inverse bounds.
It certifies its earliest confirmed passing checkpoint. Required Tier 2 word hold
restores that exact own state and checks the restored state plus 24 scheduled
states during 4,000 additional updates; every primary and confirmation check
must pass. Its allowance is 300 seconds, and its recipe, prior, architecture,
optimizer histories, streams and original 20,000-update schedule horizon remain
unchanged. The original sustained `five_word_joint_acquisition` declaration and
its evidence remain historical. See the [split readout](reports/forge/five-word-tier-split/README.md).
Revision 8 had required counts 6/21/2; its full through-Tier-2 reservation was
42,720 seconds per candidate (43,020 including the optional 300-second clock
audit diagnostic). The revision-8 `technique-inventory-word-split-v1` campaign
retains 46,620 seconds per family through Tier 3, including the diagnostic,
and 326,340 seconds for its seven-family roster. Historical campaigns,
including `tier1-gaussian-smoke-v1` and `tier1-acquisition-v2`, remain unchanged. Revision 3's `tier1-acquisition-v1` retains its
2,100-second cap; historical `smoke.json` retains its 900-second cap. Neither
covers the complete revision-4 Tier 1.
Task timeouts live in each task's `resources.timeout_seconds`. Larger campaigns
need their own JSON definition and explicit budgets. The worker reserves a
complete task allowance before starting it; a remaining budget too small for
that reservation blocks another launch.

```sh
# One bounded worker per GPU; execution is a separate, explicit action.
python -m experiments.forge drain --gpus 0,1 --workers-per-gpu 1

# For a study declaring execution_backend cpu, submit and wait for its campaign.
python -m experiments.forge run critic-anchor-v2 --study critic-anchor-v2-study --gpus cpu
```

Use `--device cpu` or `--device cuda` when planning/enqueuing; `run` infers the
backend from `--gpus`. CUDA model, CPU model, and backend participate in evidence
identity. Boards show distinct runtime cohorts, never a pooled CPU/CUDA pass.
The compact board's Compute column names the requested cohort. CPU-only behavioral
tasks still run on CPU in a CUDA-targeted request; use each attempt's device and
cost receipt when comparing execution time.

Use `--allow-sharing` only with an intentional resource policy when increasing
workers per GPU. Training time under different hardware or contention is not a
speed leaderboard. FLOPs remain labelled unavailable until measured or estimated
with a declared method.

Inspect measured automation costs with `python -m experiments.forge stats`.
Compilation also writes `reports/forge/automation.json`: paid attempts, rejection
and qualification costs, avoided requested work, reuse, execution errors and
observed concurrency. Missing timing and memory measurements stay unavailable.
Instrumented adapters separate optimizer updates, sampling and evaluation; process
RSS and PyTorch allocator peaks are scoped measurements, not whole-device usage.

Worktrees share the main Git repository's `runs/forge` queue by default. For an
explicit coordinator location, put global options **before** the subcommand:

```sh
python -m experiments.forge --root /path/to/worktree \
  --queue-root /path/to/repository/runs/forge enqueue critic-anchor-v2
```

Use the same queue root for submitters, workers, and log viewers.
`PARTICLEGAN_FORGE_QUEUE` also sets the shared queue location. Declarations and
durable report receipts belong to the checkout selected with `--root`.

After the develop integration, scalar, image, native and adaptation scoring uses
the public prior distribution without training output noise. Learned MoG kernel
noise remains present. Behavioral hosts keep their explicitly frozen laws.
Vector tasks record their retained full-component gates; upstream finite-atom
shape exemptions have not been calibrated for Forge's MoG prior. These changes
produce new evidence identities and do not relabel previous receipts.

New idea cards set `claim_contract.sampling_law` to `task_declared`. Each task's
`evaluation` declares `sampling_contract_version: 1`, `sampling_law`, and
`eval_output_noise`; the adapter records the policy it actually executes.
Planning rejects unsupported host/policy combinations. Missing observed policy
is `INCOMPLETE`; a contradiction is `INVALID`, before scientific metrics are
graded. Behavioral scheduled-noise and parameter-only measurements retain their
explicit exceptions. Archived unversioned receipts keep their frozen semantics.

Host architecture is part of a task's evidence identity. For example,
`img_intensity2` retains its original transpose12 declaration;
`img_intensity2_residual16` explicitly binds the published residual16 profile and
its source hash. Both use the same image factory, public trainer and gates.
The opt-in `host_profile_transfer` view exposes the variant as a diagnostic;
the default views do not automatically add it. `imageprofiles.task_from_profile`
materializes the full task card from the shared resolver without runtime
inheritance. Preflight rejects unsupported profiles or mismatched model/data
declarations. Receipts record the resolved card, actual parameter shapes/counts
and initial model-state hashes. Architecture transfer does not import an archived
positive or relax the fixed named-RNG comparison policy.
The same resolver materializes the three other `img_*_residual16` diagnostics.

The same diagnostic view includes six `vector_*_published` tasks. These bind
the published critic cards through shared factories, including the public
`BatchDistanceDiscriminator`. `vectorprofiles.task_from_profile` materializes
architecture changes while preserving the original task's data, resources,
learned MoG, named initialization and full-component gates. Both source
declarations travel with frozen jobs, including the report containing the
critic cards. The profile explicitly records that historical prior scale and
draw order are not restored. Raw tasks keep their existing architectures.

Native `*_affine_square_named_v1` cards select the identity affine generator,
Fourier-3 critic and explicit uniform-square initial locations through
[`particlegan.init.initialize_`](docs/api.md#initialize_module--method-parameter_generatorsnone-distributionsnone-gain10-stricttrue).
They retain learned MoG width/masses, current recipes, named streams and the full
7k coverage/accuracy gates. Task-owned component policies, actual tensor hashes
and per-parameter stream states enter receipts and checkpoint compatibility.
An incompatible prior, supplied fixture or conflicting initialization claim is
`BLOCKED`. The catalog's matching 14k variants retain the clock-free audit and
own-state checkpoint prerequisites; select them only in a view containing those
requirements. They are not added to the scheduled transfer view automatically.

Queue submission and runtime revalidate profiles against frozen source, including
actual recipe resolution, candidate identity, grouped tasks, budgets and matching
continuation parents. Cached preflight text cannot authorize a changed host.
Older snapshots retain their recorded execution contract.

## Tiers and views

Current assignments are generated from the
[registered views](configs/forge/views/) in the
[tier guide](reports/forge/EXPERIMENTS_BY_TIER.md). The
[`discriminator_stability` revision-14 profile](reports/forge/EXPERIMENTS_BY_TIER.md#discriminator_stability)
declares:

| Qualification tier | Required and other assignments | Purpose |
| --- | --- | --- |
| 1: smoke | 11 required, 1 diagnostic | Behavior, finite-state, mechanism, confirmed acquisition and model-family checks; placement remains provisional |
| 2: quality | 18 required, 7 held | Sustained live quality and continued learning across families |
| 3: endurance | 1 required, 1 held, 1 diagnostic removed from the count | Late failure after acquisition; `ring_extension` counts the shared ring run once |
| 4: stress_and_formulations | 5 held | Smaller-network accuracy, masked-observation recovery and joint BiGAN questions |

The counted denominator is 11/18/1 for Tiers 1–3; Tier 4 has no counted
requirements. Revision 11's 6/10/1, revision 9's 6/21/2, revision 7's 6/20/2,
revision 3's 5/19/2 and revision 2's published 3/19/2 results remain historical
evidence. Adding requirements or holding tests does not regrade those results
or transfer API demonstrations into Forge qualification. Calibration remains
provisional. Check the generated guide and a read-only plan after any view change.

At the [2026-10-15 test dispositions](reports/forge/test-dispositions-20261015/README.md), the six ordinary
views counted 81 view requirements (Tier 1/2/3): adaptation 3/9/0 (revision 3), clockfree_continuous 4/9/1
(revision 4), discriminator_stability 6/10/1 (revision 11), formulation_comparison 3/9/1 (revision 2),
host_profile_transfer 3/9/1 (revision 5) and quality_coverage 3/9/0 (revision 3). At those revisions, held in all six at Tier 2:
`vector_unequal_mass`, `vector_unequal_width`, `vector_anisotropic`, `vector_overlap`, `img_bars4`,
`img_blobs4`, `img_intensity2`, `grid100`, `rotated100` and `staggered100`. Also held: `gaussian1d_stability`
(discriminator_stability, Tier 2), `target_shift_recovery` (adaptation and clockfree_continuous, Tier 3) and
`grid100_14k`, `rotated100_14k` and `staggered100_14k` (clockfree_continuous, Tier 3, behind their held 7k
parents). Removed from the count: `ring_hold` (a phase of `ring_extension` in the four ring views) and
`clockfree_audit` (a duplicate of `clockfree_audit_measurement_v1` in clockfree_continuous). At that point,
adaptation counted the same tests as quality_coverage and reported `HELD` while
`target_shift_recovery` was held. These are pinned historical dispositions; use
the generated tier guide for subsequent revisions and current roles.
`tier1_policy_coverage` remains a separate scope.

All required lower-tier tasks must pass before higher-tier work is eligible.
Failed, missing, invalid, and blocked evidence stop higher-tier spending after
the current tier's runnable work completes. Frozen legacy requests also stop
the remaining tasks in that tier.
Diagnostics remain visible without vetoing required passes. The hold and extension
share one uninterrupted execution; the extension cannot borrow another
candidate's trained state or hide behind a passing ordinary hold.

A view assigns each task a role (`required`, `stretch`, `held`, or `diagnostic`;
legacy `ranking` is retained in `IMPORTANCES` only for schema compatibility)
and declares a view-level `ranking` policy (`policy: "qualified_tier_only_with_raw_metrics"`,
`compare_compatible_cohorts: true`, `cost_separate: true` in `configs/forge/views/*.json`,
or `order_by: ["qualified_tier", "candidate_id"]`, `metric_aggregation_allowed: false`
in `particlegan_forge` — candidates rank by highest qualified tier,
never by summing raw metrics across tasks). **Stretch** (`importance: "stretch"`)
is the single canonical role for optional, extra-credit, frontier, and held
benchmark tasks: stretch tests are declared, run, graded, and displayed in their
own **`## Stretch experiments`** leaderboard table by tier (`Tier 1 stretch`–`Tier 4 stretch`,
`Total stretch`) as well as in `## Full-suite results — failures included`, and
they never block tier qualification or view-level `PASS`. A view can also mark a
previously required test as **held** (role `held`, the backwards-compatible alias
of `stretch` that carries a disposition note): it is
still declared, run, graded and shown in the stretch and full-suite tables, but it is not counted, and it never blocks
progress or becomes a next task. Its `hold` (or `stretch`) note in the view gives the reason
(`no_global_recipe`, `unsolved`, `frontier`, `extra_credit`, `optional`, or `held_prerequisite` / `stretch_prerequisite` with the parents in
`blocked_by`), a plain-language summary, an explanation link and the rule for
counting it again. A test **removed from the count** stays assigned as
`diagnostic` with a `removal` note (`phase`, `duplicate`, `superseded` or
`ill_posed`); a phase or duplicate names the counted test that carries its
question in `counted_by`; a duplicate must have the same evaluation declaration as the
counted test. A counted test may depend on a removed one only when that removed test
is a phase of it: the same uninterrupted run and tier, with the counted test
continuing the phase's own saved state (a checkpoint dependency). The phase still
gates its counted test, whose grader replays it, and the phase's own prerequisites
must be counted, so a stretch, held, or diagnostic test never decides a counted verdict.
A tier with only stretch tasks has status `STRETCH` and never blocks view-level `PASS`;
a tier with only legacy held tasks has status `HELD` and neither passes nor stops later
tiers. A view reports `PASS` when every counted tier passes and its highest non-stretch
tier is counted. If every counted tier passes but the highest non-stretch tier is held, or
nothing is counted in a held-only view, the view reports `HELD`, which is never eligible; a counted
failure still reports as a failure. These notes are view policy: changing a
summary, re-entry rule or link needs a new archived revision. Apply a recorded set of decisions with
`python -m experiments.forge test-dispositions check|apply --registry <file>`;
`check` writes nothing and exits 1 on drift, and `apply` archives each old revision
byte for byte before writing, so a second apply changes nothing.

The [experiments by tier report](reports/forge/EXPERIMENTS_BY_TIER.md) lists the
current task assignments for every view, including required, stretch, held, and
diagnostic roles, declared budgets, dependencies and tasks unassigned to any
view. Regenerate it after adding tasks or changing tier placement:

```sh
python -m experiments.forge experiments-by-tier \
  --output reports/forge/EXPERIMENTS_BY_TIER.md
# Inspect one view or print a machine-readable inventory without writing.
python -m experiments.forge experiments-by-tier --view discriminator_stability
python -m experiments.forge experiments-by-tier --json
```

The report reads validated task and view declarations plus compact published
research indexes without accessing the queue, launching training or regrading
saved evidence. Its prior column names each task's code path and absolute sigma;
recorded outcomes and related API demos show their own saved priors, with missing
bindings labelled unrecorded. Each task links to its question, declared numerical gates,
recorded Forge configuration outcomes and related public-API training GIFs.
Related demos keep their own recipe, prior, initialization, budget, sampling
and runtime; they confer no qualification on a different Forge task.

Regenerate this same report after changing tasks/views or publishing results
and media indexes. New declarations and explicit retained-question mappings
are discovered automatically, and the report freshness test catches stale
content. A task's optional `description` supplies its current question; otherwise
the guide uses the retained question audit or the declared gate. Task variants
name their host/problem explicitly, and public-API variants retain `legacy_ids`
for the related-question join. The [current solution leaderboard](reports/forge/technique-inventory.md)
remains the single ranking for its goal; use its source-bound evidence and
the compiled memory to explain a release choice after full qualification.

Boards order current rows by attained tier, then name/cohort. Raw metrics and
cost remain separate; no aggregate metric ranking is declared. Filter whole rows
without shrinking their qualification denominator:

```sh
python -m experiments.forge board --family image --evidence-quality imported_recorded
python -m experiments.forge board --scope pinned --evidence-quality certified_pinned
```

The [explicit task-recipe adaptation guide](docs/forge-task-recipe-adaptation.md)
describes the released v0.7 successor's opt-in resource/objective binding. Its
original native-host card remains frozen; adaptation never silently discards
training mechanisms or imports qualification.

For a technique inventory with one passes/total column per tier, use the
[current technique leaderboard](reports/forge/technique-inventory.md).
This is the single published technique inventory, updated in place. It includes
the [configurable Modern GAN training baseline](reports/forge/R3GAN_BASELINE_READOUT.md)
and selects one complete configuration for each formulation family. The generic
[`family-current-v1.json`](configs/forge/selections/family-current-v1.json) mapping
pins its exact measured revision, recipe, source, runtime and task identities.
Alternative configurations retain their recorded results and source identities
as unranked alternatives in the companion JSON and evidence snapshots.
For publication, a family is the high-level implementation and formulation;
optimizer choice, momentum, rates and other hyperparameters are configuration
details. The registry's editorial `reporting_family` aliases group earlier
optimizer-specific evidence identities without changing their receipts or pins.
The displayed whole configuration maximizes recorded required passes within
each exact runtime, then completed measurements; ties retain the solution's
existing selection before a stable identity tie-break. This is a best-observed
display across recorded sources, not controlled scientific superiority or
default adoption. BCAP and BCAP with K3P remain separate formulations. Detail
pages show the selected recipe, configuration alternatives and diagnostics.
`inventory_visible: false` excludes a declaration from generated navigation
while preserving its evidence and original registry identity.
The generated leaderboard has bold family totals followed by view rows, with
Tier 1, Tier 2, Tier 3 and total passes/required. Family totals sum the views;
shared experiments count once per requiring view, rather than as unique runs.
`(*)` marks incomplete evidence, including changed or unbound current contracts.
Zero recorded passes display as `0(*)/N` when results are unavailable. Every
table link opens the generated family page at its view or tier breakdown, with
recorded metrics, current criteria, missing results and original evidence links.
Diagnostic-only views and historical/API studies are separate from the totals.
Retained research recipes appear as marked rows in the same tier table, with
passes counted against their original required tasks. These rows keep each
source, recipe and seed separate and do not enter the selected family totals.
Phase observation checks remain measurements of their original benchmark gate.
Detailed comparisons, attempt ledgers and untiered historical studies live on
the family pages; they do not precede the scoreboard or create extra tier credit.
One refresh rebuilds the leaderboard, family pages and experiments-by-tier
report; the older shared score index links here instead of copying the table.
The [released v0.7 task-adaptation readout](reports/forge/RELEASE07_TASK_ADAPTATION_READOUT.md)
adds one measured successor with all 24 integration preflights ready. Its
frozen legacy request's required smoke failure stopped the remaining tasks; the original native
recipe cards and their blockers remain frozen.
Numerical source snapshots and receipt proofs remain in
`reports/forge/technique-evidence/` as provenance, without separate leaderboard
tables. Regeneration launches no training and needs no raw-log hydration;
independent regrading of a source needs its byte-exact original receipts.
The roster is
discovered from `configs/forge/ideas/*.json` and
`configs/forge/configurations/*.json`. The explicit family registry keeps
hyperparameter trials within an existing technique row. R1/R2, BCap, K3P and the other public formulations retain
their actual resolved recipes and separate source/runtime/sampling cohorts.
Solutions are reusable family candidates, evaluated through the same task ladder.
Keep one global recipe on every eligible host; task bindings supply conditions
and applicability rather than bespoke optimizer settings. A changed formulation
needs a new family. Structural control changes within an existing formulation
need ordinary idea cards and explicit deltas; they must not bypass the stricter
same-signature configuration-search validator. A new configured family standard
requires all six required Tier 1 tasks to PASS from that one candidate and
compatible evidence cohort. A failure leaves the recorded incumbent selected
and the attempted replacement visible as an unranked alternative. Retaining an
incumbent does not claim it beats a candidate from another source or runtime.
Configured standards do not change public defaults or relax calibration and
independent-confirmation requirements. Recorded-policy reconstruction preserves
its original family/cohort rows.

The ordinary `inventory` execution command uses one explicitly selected current
configuration per visible solution family from
`configs/forge/selections/family-current-v1.json`. BCAP uses the winning DualNorm
configuration selected by `current_configuration_family` in the family registry.
Optimizer variants, ablations and previous runtime cohorts remain archived
comparisons; they do not expand the default benchmark roster. The current
leaderboard displays this same whole configuration per family. Historical
reproduction wrappers retain their explicit frozen idea/configuration rosters.
Configuration grids run only through an explicitly requested `search` study.

Use the [configuration-search workflow](docs/forge-configuration-search.md) to
declare a bounded recipe grid without copying a trainer. A study freezes its
tuning tasks, objective, deterministic tie-break and campaign budgets before
execution. The leaderboard uses all tier cells from one selected configuration;
it never combines each task's best result from different settings. A search with
no complete smoke pass has no qualified winner. Later tiers remain untouched
during selection, and the provisional screen still requires calibration before
scientific ranking or default adoption.

For matched lazy-regularization comparisons, use the reusable
[cadence study workflow](docs/forge-cadence-studies.md). It varies only
`reg_every`, retains the kernel's built-in strength compensation, records actual
penalty activity and generates a paired metric report. Explicit cross-tier
coverage is a separate diagnostic scope and supplies no ordinary qualification.

```sh
# Read-only cost/coverage plan, then explicit gated execution.
python -m experiments.forge inventory plan --through-tier 3
python -m experiments.forge inventory run --through-tier 3 --gpus 0,1
# Refresh the single current publication without changing saved qualification.
python reports/forge/regenerate_technique_inventory.py
python -m experiments.forge logs --follow --campaign technique-inventory-v1
```

The [expanded Tier 1 readout](reports/forge/tier1-refresh/README.md) evaluates
all 15 existing ideas and 32 saved configurations under revision 3, then refreshes
whole-configuration selections. Its frozen roster, five finite studies and
shared campaign preserve the exact configuration recipes. `inventory` alone
includes only ideas; use the readout's reproduction wrapper to include saved
configurations through their registered studies. Those frozen historical
requests stop remaining work after failed prerequisites, including later tasks
within Tier 1; new complete-tier requests finish independent Tier 1 measurements.

The [word-task root-cause readout](reports/forge/word-root-cause/README.md) shows
full task-only K3P, KA2 and R1/R2 passes with explicit existing-control settings.
Before blaming a formulation on a new host, verify effective G/E/D/prior rates
and schedule horizons, whether output noise is added after a categorical softmax,
and the penalty's units over the full joint input. Record active controllers as
well as nominal rates. The word study repaired exact-equality grading and prior
optimizer beta binding; it preserves old evidence and ordinary qualification.
Clean/full noise and horizon ablations can cross technique activation boundaries,
so register reusable structural idea candidates instead of calling them ordinary
same-signature configuration searches. The task-only study and recipe export
remain historical evidence; they are not an active solution-selection operation
or an importer into ordinary qualification. These lessons do not add implicit
recipe adaptation or establish that noise alone caused failure.

To advance a published view policy, register its measured source explicitly with
`--advance-policy --source-commit EXECUTED_COMMIT`. Earlier numerical cohorts
and their original denominators are archived unchanged; their results cannot
fill new requirements. `--recorded-policy` only regenerates evidence for the
manifest's exact policy and cannot be combined with new-source registration.
Use `--refresh-publication` to update the displayed task coverage and standalone
API results while retaining the registered scientific rows and family selections.
It verifies the immutable evidence snapshots and exact archived policy, launches
no training and does not regrade receipts. New required cells remain unknown
until compatible ordinary evidence is explicitly registered.
The default command selects that safe publication refresh automatically only
when the registered view itself (`discriminator_stability`) is newer than the
registered evidence policy; a revision of any other view would take the ordinary
publishing path. Always pass `--refresh-publication` after a view revision, for
example after holding a test or removing one from the count. Advancing
qualification still requires the source and policy options above.

The original default inventory campaign has explicit reservation ceilings for
its 12-technique roster. New techniques require checking the expanded plan and
a new immutable campaign ID with adequate budgets. The Modern GAN recipe uses
its own one-candidate campaign; it does not rerun unchanged techniques. New ordinary
requests finish runnable tasks in a tier, then apply required failures to higher
tiers. Frozen requests without the new policy retain fail-fast behavior; unsupported
techniques reserve no training resources. Current counted denominators, including
unknown and blocked cells, follow the [registered view and tier guide](#tiers-and-views).
The publication retains its recorded revision 9 qualification (6/21/2) until
ordinary evidence explicitly advances it; earlier revision-2 3/19/2 cohorts
retain their original outcomes in its companion JSON and immutable snapshots.
A zero passes/total cell alone does not establish a scientific failure.
Unreached acquisition tasks remain unknown. New compatible evidence updates
the same published leaderboard.
Archived and calibration results appear separately and cannot fill current
qualification cells. Provisional screening still confers no default adoption.
`techniques` also renders a read-only local board or JSON. The publication wrapper
exports final metrics and receipt hashes without copying per-update diagnostics
into Git. Full new execution envelopes remain ignored and archived unchanged;
summary projections are never qualification inputs. A checkout can regenerate
the same current leaderboard from committed evidence without raw-log hydration.
For an independent regrade, hydrate exact originals using the
[current archive receipt and restoration instructions](reports/forge/tier1-refresh/README.md).
`--source-commit <executed-commit>` reconstructs and verifies the recorded
implementation, registers its measured evidence and updates the same leaderboard;
its results do not qualify a newer checkout. The raw `techniques --output` export
cannot overwrite this registered publication. Memory compilation links this goal
to the current inventory instead of generating a second table. Do not rerun unchanged science solely for a
merge or a reporting change. Inventory runs compile boards once after draining;
attempt receipts and tail-able events remain available throughout execution.

Receipt certification describes evidence identity, not a scientific pass.
Use `board --json` to inspect family/provenance options and unknowns.

`discriminator_stability`, `quality_coverage`, `adaptation`, and
`clockfree_continuous` select different requirements from shared task evidence.
Clock-free eligibility requires its explicit clock/state audits and continuation;
a constant LR or scheduled hold does not establish the claim. Forge measures
step-label, horizon, evaluation-cadence and restart perturbations on saved public
trainer states. Its source audit also blocks known delayed releases that a short
probe can miss. Native 14k tasks restore the candidate's certified 7k checkpoint,
keeping the original schedule and named RNG streams. Paired adaptation compares
an active continuation with a frozen copy of the same pre-shift state.
These adapters have bounded software verification; no full-budget current
clock-free or adaptation qualification has been recorded. Monotonicity is a
future view and has no active gate.

Change tier placement through view policy, preserving task/evaluator definitions:

```sh
python -m experiments.forge retier discriminator_stability \
  --task img_bars4 --tier 1 --reason "Recorded calibration supports earlier rejection"
```

This creates a view revision and recomputes eligibility without launching jobs.
It first copies the live revision byte for byte to
`configs/forge/view-history/<view>-v<revision>.json`, refuses if a different file is
already there, and records that archive in the new revision's `previous_revision`.
Existing campaigns retain their pinned view. Changing the task's steps, prior,
sampling, or thresholds instead changes scientific compatibility. Do not shorten
`benchmarks.toy100 --steps` to claim a preserved prefix: that option also changes
annealing schedules.

## Watch and recover work

```sh
python -m experiments.forge logs --follow --candidate critic-anchor-v2
python -m experiments.forge logs --follow --task two_pole
python -m experiments.forge queue
```

`queue` reports the resolved queue location. Its `events.jsonl` is the central
timestamped stream; each campaign also has `progress.jsonl`, and each attempt has
`run.log`. For the default queue in the primary checkout:

```sh
tail -F runs/forge/events.jsonl
tail -F runs/forge/<campaign-id>/progress.jsonl
tail -F runs/forge/<campaign-id>/<attempt-id>/run.log
```

```sh
python -m experiments.forge pause <campaign-id>
python -m experiments.forge resume <campaign-id>
python -m experiments.forge cancel <request-id>
python -m experiments.forge retry <compatibility-key> --reason "Repaired the worker environment"
```

After cancellation, explicitly enqueue the same frozen request/campaign again
to reattach its subscription, then retry its cancelled compatibility key if needed.
The cancelled receipt and cost remain visible.

Retry is for a repaired execution problem; it does not erase a scientific failure.
The retry receipt binds its predecessor and repair reason. A certified repaired
attempt replaces the infrastructure verdict for qualification while both attempts
remain in the history, readout and cost. Unlinked conflicting results stay invalid.
Preserve original errors and known parity refusals. A candidate with an unsupported
capability needs adapter/parity work before another attempt. Process leases and
completion receipts protect against duplicate work during coordinator recovery.

## Finish with a comparison and recommendation

Every attempted idea needs a readout, including cheap failures:

```sh
python -m experiments.forge board --goal discriminator_stability
python -m experiments.forge readout critic-anchor-v2 --study critic-anchor-v2-study \
  --conclusion "State the measured result and its limits" \
  --comparison "Compare the same task/protocol and scoring weights with the parent" \
  --next-action "Advance, revise a specific mechanism, investigate a blocker, or stop"
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
```

To stop pursuing a finished idea, preserve its readout and record the reason:

```sh
python -m experiments.forge abandon critic-anchor-v2@REVISION --reason "Recorded failure; no further work planned"
# Or link a genuinely revised idea:
python -m experiments.forge supersede critic-anchor-v2@REVISION \
  --successor critic-anchor-v3 --reason "The successor changes the failed mechanism"
```

Stop active work and publish its readout before either command. These immutable
administrative records preserve verdicts and costs; the disposed revision cannot
be submitted or retried again. Use an exact revision prefix when the name is
ambiguous. A successor link does not transfer qualification.

If several revisions of the name have run, use `candidate-id@revision` to bind the
readout. Finish or cancel active queued/running/paused work first. A readout covers
its exact durable attempt set; a later retry requires an updated readout. A
concluded failed idea remains useful history. Live, EMA, clean/noisy,
and calibrated-sampling outcomes remain separate; an earlier good checkpoint
cannot replace a failed required terminal window.

The summary-only refresh writes one searchable [experiment memory](reports/forge/EXPERIMENT_MEMORY.md),
its publication projection, and [input hashes and coverage](reports/forge/compilation.json).
It preserves published per-view qualification tables and automation snapshots.
Rows show exact identities, status/denominator, cost, blockers, and next actions.
Historical summaries can overlap individual attempt evidence; never add their
counts or costs as though they were independent experiments.
Pinned older revisions retain recorded verdicts and costs in the board. A changed
live evaluator does not promote them. Plain `compile` runs the receipt reducers
and can change archived grades when original envelopes are unavailable or a live
validator has changed. Use it for an intentional qualification/telemetry update
only after checking the required original envelopes and their scientific bindings;
it is not the reporting-merge refresh. See the
[memory guide](docs/forge-research-memory.md) for freshness scope and preserved
snapshot identities.

## Maintain history and coverage

The committed memory is readable immediately. Rebuilding historical imports also
needs their pinned Git objects. If the importer reports a missing revision in a
new clone, fetch the exact #155 source before rebuilding:

```sh
git fetch origin 0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b
```

Keep import gaps visible if a frozen source cannot be fetched; never replace it
with the current branch by name.

```sh
python -m experiments.forge history
python -m experiments.forge history --check
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
```

History import reads the pinned initial repository and #155 trees without
executing them. It classifies every scoped tracked file, including drivers under
`reports/`, and preserves package/fixture/configuration provenance on normalized
cards. New tracked sources must appear in the catalog; unknown families require
an explicit classification. Generated Forge records and boards never become
historical mapper inputs. Keep bulk checkpoints/samples in their declared durable
location; unavailable local artifacts remain evidence gaps.

For documentation and software maintenance, check retained publications and
refresh compiled reports without training:

```sh
# Read-only integrity check under the publications' original source bindings.
python ops/refresh_forge_readouts.py --check --reports-only
# Rebuild report displays from retained evidence without changing source pins.
python ops/refresh_forge_readouts.py --reports-only
# Separately inspect gaps against the current source and declarations.
python ops/refresh_forge_readouts.py --check
```

`--check --reports-only` verifies retained publication integrity and reports
current-source measurement gaps separately. Its `CURRENT` result preserves the
recorded cohort; it grants no evidence for changed software. Package or adapter
changes can leave the strict current-source check at `NEEDS_REFRESH` while the
original publication remains intact. Preserve the original results, source pins,
budgets and qualifications. Do not rerun unchanged experiments, rewrite their
source bindings or transfer qualification merely to make a cleanup merge current.

Strict `--check` launches no training and exits `0` when readouts and compiled
reports match the current source and declarations, `1` when evidence or reporting
gaps remain, and `2` when code or declaration changes are required (such as
preflight blockers, corrupt manifests, unregistered `family-current-v1.json`
snapshot edits, or non-gate `ERROR`/`INCOMPLETE` task outcomes).
`--reports-only` without `--check` rebuilds report displays and leaves unavailable
current-source measurements explicit; a partial result can remain `NEEDS_REFRESH`.

For a separately authorized and budgeted new-source evaluation, the execution
command is:

```sh
python ops/refresh_forge_readouts.py --gpus 0,1 --scaling-gpu auto
```

Running without `--check` or `--reports-only` drains stale stages end to end.
`--scaling-gpu auto` inspects `nvidia-smi` to select a GPU from `--gpus` with zero
external compute processes so all 48 DCGAN performance cells share one
`Contention: clear` hardware cohort and matched comparison key per task. The
publisher regenerates `technique-inventory.{json,md}` when committed
`technique-evidence/manifest.json` snapshots already satisfy `family-current-v1.json`,
preserves historical `family-current-v1.json` pins under their original frozen
sources, preserves committed goal leaderboards on clean checkouts where ignored
`reports/forge/attempts/*` envelopes are absent, and seals clean-checkout-safe
`EXPERIMENT_MEMORY.md` and `compilation.json` manifests.

## Calibrate a screen before adopting it

`calibrate` replays saved evidence without training. Follow the
[current-cohort protocol](reports/forge/CURRENT_CALIBRATION_PROTOCOL.md) to freeze
the controls, independent reference, task identities, cost criteria and prior.
Check the [read-only feasibility preflight](docs/forge-calibration-preflight.md)
before selecting paid diagnostics. New calibration-lane registrations reject
mathematically infeasible profiles and unresolved original receipt issues;
logical feasibility alone supplies no adoption or qualification credit.

```sh
python -m experiments.forge calibration-preflight --profile <current-profile> --require-feasible
python -m experiments.forge calibrate --profile initial
python -m experiments.forge calibration-lane register --contract path/to/diagnostic-contract.json
python -m experiments.forge calibration-lane plan <registration-id>
python -m experiments.forge calibration-lane enqueue <registration-id>
python -m experiments.forge drain --gpus 0,1 --campaign calibration-<registration-id>
python -m experiments.forge board --scope calibration_diagnostic
```

Registration selects explicit lineages and tasks with a reason and separate task,
candidate and campaign budgets. Only those registered diagnostics can proceed
after a failed screen. Checkpoint dependencies still require a valid passing
producer. Diagnostic jobs use a separate queue namespace: their outcomes cannot
qualify ordinary candidates. Record a readout after this work as usual.
Start from the [diagnostic contract template](configs/forge/calibration-lane-template.json)
and replace its placeholders with the frozen profile and selected lineage.

User-authorized coverage diagnostics can rebuild missing checkpoints and continue
after a producer's numerical quality FAIL. The bounded `attempt_coverage` runner
uses an explicit `checkpoint-repairs.json` plan in its state directory. It keeps
the original recipe, seed, prior distribution, schedule, observations and update
budgets. A continuation still needs its complete own checkpoint, verified files,
matching static context and RNGs, and exact restored prefix. It runs on its
producer's GPU. A missing or incomplete state is rebuilt rather than relabeled.
These receipts explicitly mark `diagnostic_continuation`; ordinary grading rejects
them for qualification and the original parent FAIL remains recorded.

Native continuation cards must inherit their producer's `recipe_owned_v1` prior
contract. Legacy `learnable=true` metadata may be aligned only when the resolved
recipe already learns that same prior and its complete initial distribution is
unchanged. The repair freezes a new request and retains the earlier error receipt.

An accepted calibration is a verified report with frozen criteria and compatible
source/task/prior/RNG/runtime evidence. Editing a status field cannot approve it.
Changing the required smoke set needs a corresponding calibration before adoption.

For a separately declared screen, explicitly import already-measured compatible
diagnostics before freezing its profile:

```sh
python -m experiments.forge calibration-lane imports <original-registration-id> \
  --lineage <original-lineage-id> --tasks <task-a> <task-b>
```

This prints bindings for the new profile's `diagnostic_imports` field. It writes
nothing and launches nothing. The bindings preserve original receipt hashes,
candidate revision, task keys, costs and retry history. Calibration verifies the
original registration and unchanged scientific cohort and criteria. Unlisted
foreign diagnostics supply no credit; imported diagnostics never qualify an
ordinary candidate. A documentation-only commit preserves a registration when
its scientific bytes and execution policy are unchanged.

## Register a finished candidate's robustness stage

After all required gates, accepted calibration and a concluded readout, use the
[promotion contract](configs/forge/promotion-template.json) to preregister the
candidate, reference controls, seed set, scoring and complete budget.

```sh
python -m experiments.forge promotion register <candidate> --contract path/to/promotion.json
python -m experiments.forge promotion plan <registration-id>
python -m experiments.forge promotion enqueue <registration-id>
python -m experiments.forge promotion report <registration-id>
```

This is the sole fixed-seed screening exception. Every registered outcome counts;
there is no in-stage tuning, seed selection or live-to-EMA scoring switch.
Registration and submission revalidate the finished candidate and calibration.

## History

These are earlier guidance and readouts, oldest material first. They keep their original
sources and results. The [technique leaderboard](reports/forge/technique-inventory.md)
supersedes their status statements.

- **BCAP repair series.** Read these before proposing a BCAP repair:
  - the [five-theory comparison](reports/forge/bcap-physics/README.md): source-bound readouts and unresolved gates;
  - [round two](reports/forge/bcap-physics/round2/README.md): a scoped, sustained unequal-mass repair;
  - [round three](reports/forge/bcap-physics/round3/README.md): two sustained conditional identity repairs;
  - [round four](reports/forge/bcap-physics/round4/README.md): a smaller conditional repair, with composition, finite overshoot and tails;
  - [round five](reports/forge/bcap-physics/round5/README.md): a combined conditional/rare repair, and a width repair with rare/native regressions.
- The [implementation plan](docs/better-experiment-automation-plan-2026-09-28.md)
  defines the migration and adoption criteria.
- The [Forge preparation review map](docs/forge-integration-review-map.md) records
  the six agent-research changes and their software acceptance. Its linked
  operating guides cover recall, bounded decisions, policy execution and artifact
  retrieval. Scientific calibration and release qualification remain separate.
- The [develop integration notes](reports/forge/DEVELOP_INTEGRATION.md) describe the
  combined E22/Atlas/Forge API. Presets, external budgets and policy serving remain
  explicit. Current Forge task definitions block E22/Atlas policy cohorts before
  reservation until separate policy-aware tasks are frozen.
- The [API model-mode research](reports/forge/api-model-toys/README.md) adds AE,
  hard/categorical VAE, full DDGAN chains and AE-DDGAN composition tasks to the
  existing tier view. Its fixed BISPF CPU research cohort, source-bound numerical
  results and actual-training GIFs remain separate from ordinary whole-suite
  qualification and from older AE/VAE/DDGAN studies.
