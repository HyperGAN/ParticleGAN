# Bounded R1/R2 repair campaign

Three prospective global recipes share the unchanged current tasks and one fresh
source/runtime. The original-source attempt was interrupted by runner defects;
its completed results and costs remain preserved. The stronger archived non-saturating
AMSGrad m110 recipe is the common reference, not a new selected default.

| Role | Candidate | Delta from its declared control |
| --- | --- | --- |
| Reference | `r1r2-repair-control-20261022` | Exact archived NS/AMSGrad m110 settings; see reference-origin.json. Its study describes the complete difference from the scheduled family incumbent. |
| R1 only | `r1r2-repair-r1only-20261022` | Fake penalty weight 1 to 0; real weight remains 1 and real half-gamma remains .1/2. |
| R1 plus joint inverse | `r1r2-repair-r1cycle-20261022` | On R1 only, global joint_reconstruction_weight 0 to 1. The word consumer adds mean token CE from raw G(E(real_words)) logits using its existing G-phase batch. |

Every recipe uses constant LR .0085, NS scalar/joint losses, global AMSGrad,
D/G+E betas (.5,.99), sampled-prior betas (0,.99), learned prior locations and
zero prior regularizer/noise/EMA. Task priors, architectures, public initializer,
seed 0, target/data law and sequence, budgets, evaluation cadence and clean/live
sampling stay fixed. No per-task recipe values are selected.

The joint inverse coefficient is technique owned. It is active on word_joint
hosts and explicitly not applicable elsewhere; the value remains 1 in the global
recipe. It is not the existing task-owned reconstruction_weight. R1/R2 weights
are technique owned and consumed by fixed penalties; they are inapplicable on
the explicit reconstruction-only AE control. No task adaptation strips them.
Neither control is allowed as an axis in a numerical search.

The queue keys include the complete candidate revision even for an inactive
field, so the R1-only and composed candidates do not automatically share their
nonword jobs. Execute their original requested jobs and audit equal saved model,
optimizer, data-stream and observation trajectories where the new coefficient
is inactive. Do not silently substitute another candidate's receipts.

## Frozen scope and budget

Existing discriminator_stability revision 14; all **13 required Tier 1** tasks
plus its separate optional clock diagnostic. Complete runnable peers in this
tier before any required failure vetoes later tiers. No task or view membership,
threshold, update count or prior condition changes. The word source-binding
migration initially changes two future evaluator-source hash fields; exact previous
task bytes and unchanged top-level scoring/data functions are preserved in
source-binding-migration.json and previous-task-definitions/.

| Task | Full reservation seconds |
| --- | ---: |
| gaussian1d_smoke | 120 |
| two_pole | 300 |
| unused_token_hold | 300 |
| ae_gan_hold | 300 |
| ring16_acquisition | 300 |
| five_word_joint_smoke | 900 |
| zero_tanh_smoke | 120 |
| model_ae_ddgan_gaussian_chain | 180 |
| model_ae_gan_multimodal | 600 |
| model_ae_reconstruction | 600 |
| model_ddgan_gaussian_chain | 600 |
| model_vae_categorical | 300 |
| model_vae_hard | 300 |
| Separate clockfree_audit_measurement_v1 | 300 |
| **Each role** | **5,220** |
| **All three roles** | **15,660** |

New campaign **r1r2-repair-20261022** has paid cap **24,000 seconds** and candidate
cap **5,220 seconds**. One finite round, no scientific retries or seed variants.
The historical 8,000-second PR #401 campaign stays concluded and unchanged.
Charge each physical attempt once; retain cancelled/incomplete attempts and
their costs. Full next-job allowances must fit the campaign and candidate caps.
Software checks and media export are separate from paid scientific attempt costs.

Ordinary Tier 2 requires all 13 Tier 1 gates. No Tier 2 execution is admitted by
this initial scope. A future full Tier 2 study needs **46,620 seconds per role**
of declared allowances (including its Tier 1 bindings); full Tier 3 needs 50,520.
The ring_hold/ring_extension execution group reserves 3,600 once. Unsupported
deeper hosts remain BLOCKED and spend nothing. A separately scoped diagnostic
cannot become ordinary qualification or replace a missing checkpoint parent.

## Infrastructure interruption and repaired source

The original source completed 18 physical attempts (384.252489 paid seconds).
The reference reproduced Ring16 covariance error 1.556546 and independently
failed DDGAN generation. Five model attempts were incomplete because frozen AE
metadata was missing or native Adam combined generator/encoder roles. The
zero-tanh attempt completed training but its evaluator dispatch was BLOCKED.
None of these original receipts is rewritten or regraded.

Version 2 includes the exact AE provider metadata in snapshots, adds opt-in
homogeneous Adam encoder groups with bit-exact numerical parity, preserves
provider error diagnostics, and routes the existing four-bound zero-tanh scorer.
The eight future model task source bindings and the word acquisition source
binding are refreshed with exact original definitions preserved in
previous-task-definitions-infra-v2/ and infrastructure-source-binding.json.
Task scoring, data, architecture, prior, initialization, budget, sampling,
cadence and gates stay unchanged. All three v1 requests were cancelled after
workers stopped. The new v2 studies use the same campaign cap and retain all
old physical charges. Only the scientific source changes; recipes stay fixed.

Before admission, 177 focused runner/model tests and 32 repair API tests passed.
An isolated frozen-source, actual NS/AMSGrad two-update integration cohort
checks every model provider, exact update counts, RNG isolation and real GIFs;
it cannot qualify or replace any full-budget task.

The isolated snapshot check passed all eight model tasks; exact receipts and
reproduction source are in infra-v2-snapshot-smoke.json. All six subsequent
studies bind the same b33e375de0a1e9cf754f4d0fa24e0fd216db4fec86b1b549ccb5ea2bb0e4f175
source and passed admission with no blockers.

The separate word-retention view places the unchanged smoke and hold tasks in
a research diagnostic cohort. It requires the diagnostic smoke's own earliest
confirmed checkpoint and the original all-25-check hold evaluator. No ordinary
Tier 1 failure is removed. The complete diagnostic reservation is 1,200 seconds
per role (900 acquisition plus 300 hold), or 3,600 seconds for three roles.

The initial admission note incorrectly expected automatic reuse of ordinary
acquisition. The additional key assertion caught that ordinary and diagnostic
smoke keys differ solely by science.evidence_use. The corrected admission proof
retains both keys and records automatic_reuse=false. Keep both original
receipts and physical charges; do not substitute an ordinary parent. The full
campaign allowance is therefore 15,660 ordinary + 3,600 diagnostic + 384.252489
already paid = **19,644.252489 seconds**, below 24,000. The 5,220-second candidate
cap still requires its actual paid cost plus the next full job allowance to fit.

## Execution and tail paths

Use /usr/bin/python (current Python 3.14.8, torch 2.14.0+cu130) with the two
physical RTX A6000 GPUs. All cooperating workers in this campaign must use the
same queue root. Raw state, stdout, JSONL and request/plan dumps stay outside Git.

```sh
export PARTICLEGAN_FORGE_QUEUE=/mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/queue
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p /mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/logs
```

After software checks and source binding, inspect all three ready studies, then
commit the execution source before enqueue. Planning launches no training.

```sh
python -m experiments.forge plan r1r2-repair-control-20261022 --study r1r2-repair-control-20261022-tier1-infra-v2 --show-boundaries
python -m experiments.forge plan r1r2-repair-r1only-20261022 --study r1r2-repair-r1only-20261022-tier1-infra-v2 --show-boundaries
python -m experiments.forge plan r1r2-repair-r1cycle-20261022 --study r1r2-repair-r1cycle-20261022-tier1-infra-v2 --show-boundaries
python -m experiments.forge enqueue r1r2-repair-control-20261022 --study r1r2-repair-control-20261022-tier1-infra-v2
python -m experiments.forge enqueue r1r2-repair-r1only-20261022 --study r1r2-repair-r1only-20261022-tier1-infra-v2
python -m experiments.forge enqueue r1r2-repair-r1cycle-20261022 --study r1r2-repair-r1cycle-20261022-tier1-infra-v2
python -u -m experiments.forge drain --gpus 0,1 --campaign r1r2-repair-20261022 > /mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/logs/drain-infra-v2.log 2>&1
```

Do not edit source, recipe, task or study between the three enqueue calls.
Enqueue freezes snapshots; later workspace edits cannot change child imports.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/queue/events.jsonl
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-repair-20261022/logs/drain-infra-v2.log
python -m experiments.forge logs --follow --campaign r1r2-repair-20261022
```

Individual execution stdout is under
queue/r1r2-repair-20261022/ATTEMPT_ID/run.log. Queue requests, attempts and full
state dumps remain local. Root-level planning/registration receipts may record
their hashes and paths; only compact final metrics and provenance are committed.

## Readout and publication

Independent Forge grades decide every unchanged gate. Report all 13 required
results and clock separately, including failures, blockers and unknowns. Record
matched initial G/E/D/prior tensor digests and named consumed data/RNG streams.
For inactive joint reconstruction hosts, compare original R1-only/composed
saved evidence without changing either recipe identity or charge ownership.

Conclude each exact study with `python -m experiments.forge readout CANDIDATE
--study STUDY --conclusion TEXT --comparison TEXT --next-action TEXT`. Retain its
source/runtime, complete recipe/prior/initialization/budget/sampling bindings,
final metrics, certificate hashes, unique paid costs and actual-training GIFs.
The existing `python -m experiments.forge.tier1_media ATTEMPT_CERTIFICATE_DIR
--output MEDIA_DIRECTORY` renders saved scored observations with no training or
sampling. API model tasks also produce their own verified api/goal.gif.

The technique inventory remains the single generated leaderboard. A static
comparison of these three frozen recipes is a campaign readout, not a second
generated ranking. Keep archived grades and original selection unchanged unless
the complete new global recipe meets the required selection contract. Screening
remains provisional; no scientific default adoption follows this campaign.
