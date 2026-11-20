**Experiment work: read [EXPERIMENTATION.md](EXPERIMENTATION.md) first.**
Query prior evidence with `python -m experiments.forge recall --query "<topic>"` and read
[reports/forge/problems.md](reports/forge/problems.md) and
[reports/forge/technique-inventory.md](reports/forge/technique-inventory.md) before
proposing another idea (use [compiled experiment memory](reports/forge/EXPERIMENT_MEMORY.md)
as a reference index rather than loading it in full). Use Forge's declared gates and budgets; the initial
profile is provisional until its calibration criteria pass.

Use the terms and ownership rules in [RESEARCH_GLOSSARY.md](RESEARCH_GLOSSARY.md).

dont do seed experiments(same thing except different seed)
be token efficient
make it easy to tail the logs
summarize and give explanations, leaderboard, recommendations on experiments after completion
establish metrics/leaderboards and use them over viewing images
only keep one generated leaderboard per goal, the current one

For new comparisons, use protocol seed 0 and the repository's public
deterministic initializer. Hold each task's architecture, target/data law,
seen batch sequence, initial prior distribution and capacity, sampling, update
budget and evaluation cadence fixed across trainer candidates. Use one global
trainer configuration across
tasks; declare the trainer delta explicitly, including changes to the recipe's
prior update policy or regularizer. Isolate constructor, data,
training-noise and evaluation RNGs, and checkpoint every consumed stream.
Fixed identity/zero fixtures and initialization diagnostics must be explicit
separate cohorts, never silently substituted for the shared baseline. Preserve
archived evidence under its original source, seed and initialization contract.

Keep bulk research logs and per-update metric/event streams out of Git. Store
raw stdout, JSONL traces, JUnit logs, checkpoints, and tensor/state dumps locally
or in an artifact archive. Commit compact reports, final metrics, provenance
receipts, and reproduction sources instead. When removing tracked logs, retain
their exact archive commit/blob identities and repair report links; do not
rewrite qualification results or rerun unchanged experiments just for a merge.
Never force-add ignored bulk logs. Keep new execution logs easy to tail.

Bind trained gates to their actual recipe, prior, initialization, budget and
sampling law. Check these against the qualified evidence before attributing a
failure to a formulation change. Clean and noisy served results are separate
cohorts; matching training settings alone does not make their gates comparable.

**A faithful numerical FAIL is better evidence than a PASS obtained by changing
the formula.** Keep these boundaries explicit in implementation, tests and reports:

- The recipe owns every training update and intervention, including prior moves,
  gradient transformations, controllers, regularizers and noise. Task hosts must
  not add cluster fitting, particle placement, weight resets, auxiliary solvers,
  surrogate gradients or other post-optimizer repairs to make a recipe pass.
  A useful new mechanism needs its own public recipe capability, declared delta
  and evidence identity before comparison.
- Execute the exact resolved recipe. Never substitute a different preset,
  schedule, prior, guard or noise policy to accommodate a host. Implement the
  missing capability or report BLOCKED; do not silently drop the mechanism.
- Evaluate the object and law named by the question: a neural-imputer test must
  call that neural imputer; no-posterior-resampling means no fallback to posterior
  resampling. Live, EMA, averaged and clean/noisy serving are separate contracts.
  Record the observed sampler and resolved recipe, not just the requested labels.
- Architecture and initialization belong to the task. Imports must not replace
  shared model classes or change constructor defaults. Normalized, scaled,
  identity-initialized and assisted variants need explicit separate task/cohort
  identities. Custom backward functions must implement their declared derivative;
  a surrogate update is a recipe mechanism, not an invisible architecture repair.
- Checkpoint every behavior-changing state and consumed RNG stream, including
  host caches, control flags, penalty clocks and actual EMA histories. Verify
  continuation in a fresh object/process across mechanism activation boundaries;
  reject incompatible old checkpoints rather than reconstructing missing history.
- Keep numerical FAIL distinct from BLOCKED, ERROR, INVALID, INCOMPLETE, TIMEOUT
  and unmeasured outcomes. A negative eligibility decision is not a numerical
  failure. Diagnostic, assisted, adapted or incompatible-source results cannot
  fill ordinary qualification or a selected recipe's matched comparison cells.
- Preserve affected historical receipts byte for byte. Publish explicit,
  source-bound scope corrections instead of rewriting old grades or presenting
  old assisted passes as evidence for a corrected implementation. Keep one
  current leaderboard and derive its current explanations from the same records.
- Add semantic controls that would detect the wrong formula or evaluator:
  forbidden-fallback checks, deliberate damage to the measured model, fresh-process
  import-order checks and exact continuation checks. Passing software tests, an
  oracle witness or a training GIF does not itself establish learned quality.

Scientific superiority and default adoption also require accepted compatible
calibration and independent confirmation. Keep unequal tuning budgets, selection
tasks, untouched task-level holdouts and unresolved uncertainty visible; never
manufacture calibration acceptance or run seed-only searches to improve a score.

Toy tests must follow the [public-API test contract](reports/toy_audit/api_contract/README.md):
execute through the ParticleGAN API, declare a numerical pass/fail metric, and
provide an actual-training GIF that illustrates the test's goal. Reform an
incompatible question into an explicitly scoped, runnable variant, retaining its
useful controls and the original evidence identity.

Each agent can queue experiments on the GPUs without waiting for it to be free. It's ok to run multiple in parallel.

---
develop->master PR should always exist and have a link to the up to date https://github.com/255BITS/ParticleGAN/blob/develop/reports/forge/technique-inventory.md
master should always reflect whats in pypi
