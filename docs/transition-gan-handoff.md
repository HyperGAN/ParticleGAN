# Transition GAN continuation handoff

## PR merged; next direction is a Gym environment (2026-09-20 local)

User explicitly requested PR/merge and winner defaults. Completed PR #9:
https://github.com/HyperGAN/ParticleGAN/pull/9
Merged into master as d2a6450985282047d2d27c36eb80ff9f5200f8c9; feature commit
6bd80e8030ccd6dccb58f6e9fd3ed1432918eeee. Local checkout is now master at that
merge, matching origin/master. The old feature branch remains available.

No-argument CLI and configs/transition/default.yaml now select encoder_shared_state:
E=true, shared state critic=true, class gain8, concat D, joint+marginal feedback,
MoG1024/bcap28k/batch256. Fresh default output results/transition/default.
Historical YAML configs explicitly preserve former factors; original joint UCD
config is configs/transition/ucd_joint.yaml. Pinned completed reports unchanged.

Validation: full local CPU pytest289passed/4 opt-in CUDA skips; scoped23passed;
no-config CLI CPU smoke passed with correct winner E/D capacity and inference;
all12 leaderboard entries validated. GitHub PR CI passed Python3.10/3.11/3.12,
installed wheel smoke, and release distribution build. Generated report whitespace
cleaned; .gitattributes marks generated report data for review. Portable bundle
rebuilt and verified. No full seed-repeat training during this default change.

PR includes transition implementation/configs/docs/tests, historical reports and
shareable demo. This local handoff, AGENTS.md user edit, masked-observation proposal
and unrelated untracked files were excluded/preserved. No active work/run.

User's stated NEXT: tackle an OpenAI Gym environment with this model. No environment
selected, no Gym dependency added or environment work started. On resume, use a
new feature branch from master; retain user's MoG/bcap preference and no seed-only
experiments. Earlier recommendation is independent action coverage and a matched
supervised predictor control. Scope the environment adaptation with the user as
needed; this round was PR/merge/defaults only.


## Visual demo added (2026-09-20)

User requested both model diagram and generated states/actions, emphasizing the toy
behavior and a shareable explanation if working. Added reports/transition/demo/index.html
(self-contained offline HTML, ~8.7MB), transition_demo.zip (~3.8MB), toy_transitions
PNG/PDF and architecture PNG/SVG/PDF. Page controls both models/splits/classes/all
saved scenes and5times, synthetic vs real-input prediction, matching axes, zoom,
reference overlay and PNG export. Includes measured success and limitations plus
copy-ready description. No external publishing requested/performed; no training.

Sources: experiments/render_transition_architecture.py, experiments/render_transition_demo.py,
lib/transition_demo.html. Rebuild architecture first, then demo. Architecture source
exports also in reports/transition/architecture. Browser headless smoke passed five
interaction cases across80saved context records and PNG export; node syntax check
passed. PNG/browser rendering inspected. Static image is explicitly scene1/class0
interpolation; detail limits include all upper-branch sample/action/next points.
Broad two-route structure appears; red connectors reveal significant step errors.


## Latest completed round: encoder and shared state critic (2026-09-20)

Both authorized runs completed on feature/world-model-no-trajectory. No active
training sessions, no commits or pushes. Session11852 exited0. GPU1 unrelated work
was left alone. No seed-only repeats. No further experiments started.

- New leader encoder_shared_state: original test SW1 .0856469156, residual
  .0173622537, coverage .2481933594. Rank1/12; 14.58% better SW1 than old leader.
- encoder_separate: SW1 .0953814320, residual .0180284679, coverage .2770996094.
  Rank2/12. Shared improves SW1 10.21%, but loses 2.89 coverage percentage points.
- Shared/separate real-input next-state L2 .01665255/.01793784; synthetic SW1
  .086715/.095953, residual .014712/.013219, coverage .25244/.31797.
- Shared interpolation/extrapolation SW1 .065259/.146810; extrapolation coverage
  .00722656 (separate .01796875). Generalization remains weak.
- Shared class0/1 SW1 .0925837/.0787101, upper frequencies .837891/.303223;
  separate .1115242/.0792386, upper .864746/.278320. Targets .8/.3.
- E uses7/5 components on saved real test inputs, effective4.95/4.12. G still
  samples uniformly across1024. Continuous offsets/context also carry information;
  low usage alone is not proof of failure. Mean prediction error ~46%/49% mean step.

Implementation:
- lib/transition.py adds TransitionEncoder using repository particle_ae (.25temp,
  summed distances, bounded3sigma offset, deterministic/noKL), encoded_transition,
  composed_transition, shared-state critic role plumbing. G architecture unchanged.
- experiments/train_transition.py adds encoder losses/EMA/checkpoint and separate
  evaluate_encoder. E(st,at,observed context), NEVER real st+1. Real triple MSE and
  synthetic st/at reconstruction weights1. Adversarial mean(prior,composed) for G;
  D batch half prior/half composed. Synthetic target detached; input path live.
- Shared state D uses common physical-state normalization and actual t versus t+dt;
  joint/action unchanged; average over3 marginal roles preserved. Four roles but
  only3 unique Ds. Exact explanation docs/transition-gan-encoder.md, plain G1 -> st.
- Configs encoder_separate.yaml and encoder_shared_state.yaml; outputs
  results/transition/encoder/<name>. Same archived source, recipe, scaler and prior;
  only shared flag/output differ. G65286/E42688/prior32768 parameters. D238084 vs
 203779. Train826.9s vs954.7s; uncontrolled throughput. Same MoG1024/bcap28k/batch256,
  seed24002/fixedsigma .13111965, 14,336,000 real draws +32768 normalization draws.
- Original data/evaluate AST and exact references unchanged. Leaderboard explicitly
  marks paired-supervision encoder cohort, fixed E budget, retains old G +/-5% and
  all protocol/reference/pinning checks. All12 entries validate.

Validation/artifacts:
- 22 scoped tests pass;3 encoder tests re-passed after generalizing inference eval
  boundary. Old leader and both new checkpoints replay on CUDA exactly for first256
  prior draws; new reconstructed/composed paths and routing IDs exact, all context
  metrics within1e-6, fixedsigma verified. tests/test_transition_encoder.py added.
- Reusable experiments/verify_transition.py and experiments/audit_transition_actions.py.
  Action audit uses32 real samples/context, physical +/- .0001 action coordinate,
  held-fixed state/context, deliberately outside route manifold. Median response
  Jacobian identity error13.68shared/14.14separate; switches2.07%/2.42%. Bad response
  also without switches; near-bound offsets31%/25%. A coarser first-arm probe saved
  separately; matched diagnostic is .0001. Do not claim learned arbitrary dynamics.
- reports/transition/leaderboard/README.md, READOUT.md, encoder_comparison.json,
  encoder_{separate,shared_state}_{verification,action_audit}.json. ROUND5.md preserves
  preceding conditioning readout. Historical conditioning plots/audits unchanged.
- README/docs updated. Tail logs: tail -F results/transition/live.log.
- /tmp/transition_encoder_report.py generated comparison JSON; optional temporary
  helper. Reusable verification/action audit live in repo. Original guide updated.

Recommendation (not implemented): use shared encoder as score baseline, retain
separate encoder for coverage/consistency. Next prioritize a separate transition
benchmark with independently varied actions per state, plus a matched supervised
predictor control, to test whether joint learning helps conditional prediction.
New dataset needs its own board. Do not force E usage uniform just for count.
Preserve unrelated files and all existing results. No commits/pushes requested.

## Historical plan, now completed: encoder path, then shared state critic

User said: "ok i'll compact then you can run both" after discussion below.
Wait for resumption after compaction; no runs started for this plan. On resume,
implement and run both experiments without asking for authorization again.
This supersedes the prior recommendation to only audit frozen latent assignments.

User idea: use ae_gan routing E(st, at) -> z -> G3 -> st+1, including the
synthetic graph G1/G2 -> (st,at) -> E -> z -> G3. Also consider sharing the
state-space marginal discriminator between G1 and G3.

Proposed implementation defaults (not additional user requirements):
- Base on the current score leader concat_class8_marginals (class8/context1),
  so arm1 keeps D_joint,D_state,D_action,D_next. Arm2 changes only the two
  state marginal critics to one shared D_state; retain D_joint and D_action.
- Retain original prior sampling and independent G1 -> st,G2 -> at,G3 -> st+1.
  Add E(st,at,observed context) -> encoded z, decoded by all three Gs to
  reconstruct st/at and predict the paired real st+1. Never give E real st+1.
  Train against real transition examples; synthetic self-consistency alone can
  reinforce an incorrect model. Include/evaluate the user's synthetic composed
  path, not merely an unrelated encoder. Explicitly document the selected loss
  terms and gradient paths, including how the synthetic path is trained.
- Use repository particle_ae routing: selected MoG mean + sigma * bounded offset,
  deterministic, no KL. See particlegan/autoencoder.py and recipes.py. Do NOT
  blindly switch to ae_gan preset: it defaults to400 particles,zdim2,6k steps,
  different optimizer/bcap cadence. Adapt encoding to existing MoG1024,zdim32,
  fixedsigma,bcap every update,28k,batch256,seed24002 recipe.
- Shared state D must use common physical-state normalization and actual time:
  D_state(st,context at t), D_state(st+1,context at t+dt). Same state space does
  not imply equal time-unconditioned distributions. Preserve average weighting
  of the three marginal roles so critic sharing doesn't silently change weights.
- No forced G3=G1+G2 or physics loss. Real next-state prediction is explicitly
  added supervision; disclose that versus original adversarial-only baseline.
  Toy action is displacement so next-state prediction is deterministic/trivial
  analytically. Do not present this as evidence of learning general dynamics.
- Keep pinned prior-generated joint evaluation unchanged; add separate conditional
  prediction and synthetic-composition evaluations. Report both generation and
  prediction, residual/coverage/class breakdown, encoder routing usage, added
  E parameters/compute and real-draw budget. Any expanded leaderboard cohort or
  capacity rule must be explicit; don't silently weaken historical checks.
- Add meaningful encoder/gradient/shared-critic/checkpoint tests. Freeze sigma,
  preserve independent RNG streams and exact data references where applicable.
  No seed repeats. Fresh output directories; tail -F results/transition/live.log.
- Summarize both arms versus baseline, update leaderboard/readout/handoff, and
  recommend next steps. Recheck GPUs and leave unrelated jobs/files alone.

## Latest round: class scale 6/context scale 2 completed (2026-09-20)

This section supersedes historical recommendations below. Branch
feature/world-model-no-trajectory; no commits/pushes; no active runs.

- Config concat_class6_context2.yaml, ID concat_class6_context2, output
  results/transition/conditioning/branches_concat_class6_context2. Only class gain
  8->6 and output path change versus context2 baseline; all recipe/budget fixed.
- SW1 .1199571813, residual .0162177719, coverage .2381347656, precision .24887695;
  train242.8s. Fifth of ten. Leader concat_class8_marginals remains .10026853.
  Versus class8/context2: SW1 worse12.6%, residual better15.0%, coverage better12.1%.
- Class0/1 SW1 .1289523588/.1109620037, upper .89208984375/.34521484375.
  Class0 barely improves; class1 worsens. Interp/extrap SW1 .10222924/.17314101,
  residual .01421090/.02223838, coverage .31373698/.011328125.
- Train SW1 .09116554/resid .00546184/coverage .61298828. Test residual ~44% mean
  step. Shuffled generated test SW1 .18219623/resid .17914145.
- Added geometry_class_audit.json for leader,class8/context2,class6/context2:
  all four geometries,both classes; existing scores and midpoint route counts.
  Class0 upper excess occurs across geometries. Leader extrap class1 upper .297
  but coverage near0: mixture frequency and spatial fit are separate issues.
- Config-only; no model/trainer/evaluate edits. Prior19 tests not rerun. New CUDA
  checkpoint first256 test samples replay exactly; metrics match1e-6; sigma fixed.
  Ten registry entries verified; plots inspected. Sessions87891/85639 exited.
- READOUT current; previous readout ROUND4.md; guide updated. Plot/perclass audit
  now seven selected rows. Optional /tmp scripts: verify_transition_continuation.py,
  plot_transition_continuation.py, audit_transition_geometry.py (three selected IDs).
  /tmp/transition_class6_context2_verification.json has detailed verification.
- NEXT: pause gain tuning; paired-latent audit on frozen class8_marginals and
  class8_context2 checkpoints. Reuse exact noisy z across classes/train-test geometry,
  inspect route allocation,branch agreement,residual,within-route spatial errors
  away from endpoints. Preserve G1 -> st,G2 -> at,G3 -> st+1 with shared noisy z.
  No new training config/run yet. Use findings to select a structural intervention.
  No seed repeats. Logs tail -F results/transition/live.log. Reserve untouched
  geometries for eventual validation. Preserve unrelated files.

## Latest round: context scale 2 completed (2026-09-20)

This section supersedes the historical status/recommendations below. Branch
feature/world-model-no-trajectory; no commits/pushes; no active runs.

- Config concat_class8_context2.yaml, registered ID concat_class8_context2, output
  results/transition/conditioning/branches_concat_class8_context2. Config-only:
  class8/context2, joint concat D, MoG1024/bcap,28k/batch256/seed24002 unchanged.
- SW1 .1065498878, residual .0190887991, coverage .2123535156, precision .2375;
  train252.1s. Third of nine. Leader still concat_class8_marginals .10026853.
  Versus context1: SW1 worse0.9%, residual better12.3%, coverage better8.9%.
- Interp/extrap SW1 .0870079501/.1651757009, residual .0167650990/.0260599051,
  coverage .2779296875/.015625. Intermediate SW1/residual tradeoff; coverage
  slightly exceeds context1 and4. Extrap coverage still only1.6%.
- Class0/1 SW1 .1303018127/.0827979630; upper frequencies .9008789063/.2900390625.
  Class1 closer, class0 worse. Train SW1 .07446216,resid .005393868,coverage .57114.
- No model/trainer code changes; preceding19-test pass still applies, not rerun.
  New checkpoint first256 test samples replay exactly, metrics within1e-6, sigma
  fixed. Nine entries pass registry protocol/reference checks. Plots inspected.
- READOUT.md current, previous context4 readout preserved as ROUND3.md. Plot and
  class_distance_audit include six selected rows. Optional /tmp verification/plot
  scripts still available. Run/verification sessions exited; GPU0 used, GPU1 left
  alone. Easy log tail -F results/transition/live.log.
- NEXT recommendation: class scale6/context2, single joint concat D and fixed
  recipe/budget, to test class0/class1 compromise. No config/run yet. If this fails,
  prioritize conditional mode-allocation diagnosis over more input-scale tuning.
  No seed-only repeats. Preserve independent G1 -> st, G2 -> at, G3 -> st+1 sharing
  the same noisy z, no physics loss/hard identity. Keep all diagnostics visible.


## Latest round: context scale 4 completed (2026-09-20)

This section supersedes the recommendations/status below, which remain history.
Branch feature/world-model-no-trajectory; no commits/pushes. No active runs.

- Added `g_context_scale` (default1), passed as `context_scale` to G. Scales only
  four geometry/time inputs; same shared noisy z and class input to independent Gs.
  Plain float, not state_dict data: restore from checkpoint config, default1 for
  old checkpoints, like class_scale. All parameter counts unchanged.
- Config concat_class8_context4.yaml; run
  results/transition/conditioning/branches_concat_class8_context4; registered ID
  concat_class8_context4. Class8, context4, joint concat D, fixed MoG1024/bcap recipe,
  28k updates/batch256/seed24002. Training265.8s. Eight completed board entries.
- SW1 .10692454 (third), residual .01550325, coverage .19438477. Versus matched
  context1: SW1 worsens1.2%, residual improves28.8%, coverage essentially flat.
  Score leader still concat_class8_marginals at .10026853.
- Interp/extrap SW1 .09278800/.14933416 (interp worse9.8%, extrap better11.6%).
  Residual .01292754/.02323039, coverage .25397135/.015625.
- Class0/1 SW1 .12479490/.08905418, upper frequencies .884765625/.31298828125.
  Class0 still overshoots. Train SW1 .0725516, residual .0054094, coverage .63408.
  Test residual ~42% of mean reference displacement; coverage gap remains.
- 19 scoped tests pass. Checkpoint replay for new run AND old class8 exact on
  first256 saved test draws with updated G; all test metrics within1e-6; sigma fixed.
  Eight board entries pass pinned protocol/reference checks. Plots inspected.
  A preliminary AST check under system Python mismatched due to Python-version
  AST differences; project .venv Python check and registry both match all nodes.
- Leaderboard renderer exposes geometry/time scale; canonical config fills old
  g_context_scale=1 to detect duplicates. No data/evaluation AST changes.
- READOUT.md is current; prior readout preserved as ROUND2.md. Comparison PNG and
  class_distance_audit.json now include new run. docs/transition-gan.md updated.
- Optional /tmp/verify_transition_continuation.py restores both scales now;
  /tmp/plot_transition_continuation.py includes five selected runs. Temporary
  helpers are not repository dependencies.
- NEXT recommendation: context scale2, class scale8, single joint concat D and
  all budgets fixed. Tests midpoint tradeoff, not launched or implemented config.
  Do not repeat seeds. Keep independent G1 -> st, G2 -> at, G3 -> st+1, same noisy
  z; no physics loss/hard identity. Continue perclass, residual, coverage audits.
- Logs: tail -F results/transition/live.log. GPU0 was free at start; GPU1 busy with
  unrelated work. Recheck on resume. Preserve unrelated untracked files.

## Prior round status

Status: completed two more experiments on 2026-09-20. Latest best is **0.10027
joint SW1**, with important consistency/coverage regressions. All runs are complete.
No commits or pushes; branch feature/world-model-no-trajectory. Follow this newest
section; the prior rounds below remain as history.

## Latest round: scale 8 and marginal feedback

- New configs: configs/transition/concat_class8.yaml and
  configs/transition/concat_class8_marginals.yaml. No model/trainer changes needed.
- New outputs: results/transition/conditioning/branches_concat_class8 and
  branches_concat_class8_marginals. Registered IDs concat_class8 and
  concat_class8_marginals. Seven entries now; fixed data/recipe/protocol verified.
- Scale8 joint: SW1 .10562508, residual .02176602, coverage .19492, upper class
  frequencies .87158/.32373. Interp/extrap SW1 .08454/.16888; train283.1s.
- Scale8 joint+marginals: SW1 **.10026853**, residual .02393071, coverage .15752,
  upper frequencies .88184/.30859. Interp/extrap .08008/.16084; train565.1s.
  G65,286 / D238,084 params; marginal_weight1. Same shared noisy MoG1024 and seed.
- Score gain from prior .14179 leader: 29.3%; additional marginal gain vs scale8
  joint: 5.1%. But held-out consistency and coverage worsen, so no all-metric win.
  Scale4 remains stronger on those diagnostics among conditioning challengers.
- Scale8 gain mostly fixes class1: class1 SW1 .19946 -> .09340 -> .08275 for
  scale4, scale8, scale8+marginals. Class0 .08412 -> .11785 -> .11779.
- Winner train residual .00578 vs test .02393 (~65% of mean step .03656).
  Train coverage .68877 vs test .15752, extrap coverage .000195. This is a
  geometry generalization problem as well as a class-mixture problem.
- Full interpretation: reports/transition/leaderboard/READOUT.md. Previous readout
  preserved as ROUND1.md. New conditioning_tradeoffs.png and class_distance_audit.json.
- 19 scoped tests pass. Both new checkpoints exactly replayed first256 test draws
  on CUDA, recomputed metrics matched within1e-6, fixed sigma verified. Comparison
  plot inspected. Temporary helper scripts /tmp/verify_transition_continuation.py
  and /tmp/plot_transition_continuation.py are optional, not repository dependencies.
- GPUs had unrelated jobs; did not touch them. Wall times are not controlled
  throughput comparisons. Training sessions have exited. Logs remain easy to tail.
- NEXT RECOMMENDATION (not implemented/run): use concat_class8 single joint D as
  economical base; add optional G geometry/time context scaling, try4 versus
  current1 with class scale8 and all budgets fixed. Hypothesis: geometry conditioning
  became relatively weak as class scale increased; not established by these runs.
  Keep independent Gs, same noisy z, no physics loss or hard G3 identity. Inspect
  held-out coverage/consistency as well as SW1. Don't launch seed-only repeats.

## Prior round: leaderboard established and beaten


- Persistent registry: configs/transition/leaderboard.json. Tool:
  experiments/transition_leaderboard.py. Report:
  reports/transition/leaderboard/README.md; explanation: READOUT.md beside it.
- Registry pins summaries/source archives and verifies unchanged data/evaluation
  ASTs, recipe (except critic conditioning), normalization, prior metadata, exact
  reference arrays, 28k updates, and G parameter budget within 5%. Model/trainer
  revisions may differ. These held-out geometries are now a development benchmark
  because we have used them repeatedly for model selection.
- Added d_conditioning=ucd/concat. Concat feeds one-hot class inputs into a scalar
  D and omits UCD CE. All optional marginal critics use the selected mode.
- First challenger configs/transition/concat.yaml: .25182 SW1, .01532 residual,
  upper fractions .613/.619. Did not beat the board or fix conditioning.
- Added g_class_scale (default1), scaling only G's class indicators, not the
  shared MoG draw or geometry/time. No additional parameters. Constructor setting
  must be restored from checkpoint config; old checkpoints default to1.
- Winner configs/transition/concat_class4.yaml: three independent Gs, shared noisy
  MoG draw, scalar concat joint D, G class scale4. **.14179 SW1**, .01464 residual,
  midpoint upper fractions **.826/.485** (targets .8/.3). G65,286 / D135,169 params.
  Interpolation/extrapolation SW1 .12405/.19503; train time318.2s. Class1 still biased,
  support coverage .301 vs reference .954, residual ~40% of mean step length.
- Runs: results/transition/conditioning/{branches_concat,branches_concat_class4}.
  Five completed entries on the board. No seed-only repeats. Default.yaml remains
  the original MoG/UCD setup; use concat_class4.yaml for the current best.
- 19 scoped tests pass. Replayed both challenger checkpoints to reproduce saved
  samples, recomputed their scores, checked fixed sigma and viewer JavaScript.
- Next recommended experiment: G class gain8 with the same concat joint D, MoG,
  networks and budget. Then reconsider marginal critics once label use works.
  Do not enforce G3 = G1 + G2 or change the user's shared-z architecture silently.

The original "proposed next experiment" section below has now been attempted;
explicit D conditioning alone was insufficient. Follow this latest section when
resuming, rather than repeating that experiment.

## Intent and architecture

Explore whether jointly learning states, actions and outcomes helps learn their
distribution, inspired by MisGAN. Preserve the user's shared-latent architecture:

```text
G1 -> st
G2 -> at
G3 -> st+1

D_joint(st, at, st+1)
optional D_state(st), D_action(at), D_next(st+1)
```

All Gs receive the SAME sampled MoG center and Gaussian noise draw, plus observed
geometry, physical time and preference class. Three independent MLPs, no shared
trunk. Do not replace this with G2(st,z) or G3(st,at,z) without new steering.
Marginal Ds receive only their own output coordinates plus observed context.

Reference triples evaluate two adjacent points in the analytic Routes family.
State is 2D position; action is displacement; next state is independently
generated. Reference relation: st+1 = st + at. No physics loss, no overwriting
G3 outputs, no trajectories constructed during training. Complete records only
for now; this is not yet missing-data recovery or arbitrary state/action inference.

## Current implementation

- Branch: feature/world-model-no-trajectory. No commits/pushes made.
- examples/transition_gan.py: runnable entry point.
- experiments/train_transition.py: recipe, objectives, training, evaluation.
- lib/transition.py: sampler, scaler, generators, joint/marginal critics, metrics.
- lib/transition_visuals.py: PNG and standalone interactive canvas viewer.
- experiments/analyze_transition.py: comparison guards, leaderboard, preference audit.
- tests/test_transition.py: 7 tests; combined with trajectory tests, 17 pass.
- docs/transition-gan.md: current user-facing setup and commands.
- docs/transition-gan-plan.md: original rationale; superseded numerical choices.

The user corrected the initial prior choice: it MUST now be MoG with about 1k
particles, not the original point-particle GAN preset.

Historical handoff settings (before the shared-default cleanup): 1,024 components, z_dim=32, sigma_rel=.025,
standardized sampled means, fixed calibrated sigma (~.13112 for this initialization).
28k updates, batch256, Rp logistic, UCD class heads, bcap cap1/weight1/every step.
G LR=.0006, D multiplier1.5, prior multiplier100 (LR=.06); G/D betas(0,.999),
prior betas(.5,.999); default cosine LR schedule and EMA .995.
Regularize RAW centers, full table at <=1024, once per G update. MoG forward(ids)
adds noise, so never use it to retrieve raw centers for the regularizer.

Every D independently minimizes Rp + UCD + bcap, with penalty in its own input
space. G objective is L_joint + marginal_weight * mean(three marginal losses),
plus the prior regularizer. marginal_weight=1. D objectives are summed across
independent networks. Joint critic remains width256; marginal critics width128.

G sizes: branches width128 =65,286 parameters; monolithic width234 =65,526.
D sizes: joint134,914; joint plus marginals237,448. Additional critics add compute;
updates and real draws match, total compute does not. One fixed training seed24002.

## Completed MoG comparison

Configs: configs/transition/default.yaml, marginals.yaml, monolithic.yaml.
Runs: results/transition/mog_1024/{branches_joint,branches_joint_marginals,monolithic_joint}.
Report: reports/transition/mog_1024/README.md and leaderboard.json, plus viewers.

| Setup | Test joint SW1 | Mean consistency error | Train seconds |
|---|---:|---:|---:|
| Three G, joint + marginals | .24999 | .01486 | 609.6 |
| Three G, joint | .25501 | .01529 | 248.2 |
| One G, joint | .26215 | .00519 | 216.1 |

Reference SW1 floor .03788; mean reference step .03656. Shuffling generated blocks
within context increases joint SW1 and inconsistency, proving some coordination
was learned. It does not prove accurate conditional distribution fitting.

The central finding: ALL THREE MoG configurations largely ignore preference class.
Midpoint state upper-side fractions, averaged over held-out geometries:

| Setup | Class0 upper | Class1 upper |
|---|---:|---:|
| Three G, joint + marginals | .577 | .571 |
| Three G, joint | .659 | .665 |
| One G, joint | .566 | .587 |
| Reference draws | .815 | .295 |
| Analytic target | .800 | .300 |

The audit counts state y above the analytic centerline at tick31/63. It checks
mixture weights, not support validity. A class-agnostic sampler can fit the pooled
upper probability .55 while missing both conditional distributions. This explains
why good physical consistency alone is insufficient. Added marginals help only
slightly and do not fix conditioning; no statistical generalization claim from
these single-seed runs.

Older reports/transition/README.md records POINT particles (20k atoms, 7k steps).
Its monolithic model did better (.0842 SW1) than its branches (.2535). Do not mix
these into the MoG leaderboard or attribute changes solely to Gaussian noise:
the recipe also changed optimizer, normalization of the prior and training horizon.

## Proposed next experiment (not launched)

User said they will compact and then continue trying to solve this. Recommendation
already given: compare explicit class input to critics against current UCD head
selection, keeping MoG, G architecture and update/data budgets fixed. Current Gs
already receive the class explicitly. Current D backbone receives geometry/time;
class only selects its UCD output score and supplies discriminator CE supervision.

This is a hypothesis about optimization/conditioning, not an established bug or
proof UCD causes the failure. Add a configurable conditional/concat discriminator
mode and test whether preference separation and joint SW1 improve. Preserve joint
and marginal options. Match parameter counts approximately and disclose differences.
Given the tiny gain and extra cost of marginal Ds, a joint-only comparison is a
reasonable first diagnostic. Keep the shared-z three-generator idea available.
Do not launch a seed-only repeat. Further marginal-only or missing-observation
experiments should follow diagnosis of the preference failure.

## Validation and operating notes

17 scoped tests passed (transition + trajectory). All three completed MoG EMA
checkpoints were loaded and reproduced the first 256 saved test samples on CUDA
with the original evaluation RNG. Sigma matched initial calibration exactly.
Reference samples, contexts and times match exactly across all three runs.
Viewer JavaScript executed through initial/final selections with mocked canvas
under Node; PNGs visually inspected. Analyzer rejects mixed recipes or duplicates.

Easy log: tail -F results/transition/live.log. Per-run log.txt and metrics.jsonl
also flush continuously. All runs are complete; no training process remains from
this work. Last GPU check showed GPU0 available before these runs, GPU1 busy with
unrelated work; recheck before execution. Use fresh output directories.

Preserve unrelated pre-existing untracked files, especially other experiments,
results/motion, runs/, .claude/, and the earlier masked-observation proposal.
New transition code/docs/reports remain untracked; README.md and .gitignore have
our tracked edits. Do not mistake untracked files for permission to delete them.
AGENTS.md: no seed experiments, be token efficient, easy-to-tail logs, summarize
experiments with explanations, leaderboard and recommendations. No subagents
unless the user or applicable instructions explicitly authorizes delegation.
