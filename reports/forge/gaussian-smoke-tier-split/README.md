# Gaussian acquisition smoke and Tier 2 stability

The new Tier 1 question is **can training produce a correct Gaussian?** Any one
of 24 scheduled states must pass the unchanged finite, location, width and KS
bounds, with a second independent draw passing at the same state. All 1,000
updates and all 24 paired observations still finish. No learning shutdown or
best-checkpoint serving is introduced.

The new Tier 2 question requires every one of 72 stationary observations between
updates 1,000 and 4,000 to pass. Training then changes the target mean from 2 to 3,
continues for 2,000 updates without resetting optimizer/history or the original
schedule, reacquires by update 5,000 at five terminal checks, and passes every
remaining 24 shifted hold checks. A frozen copy of the learner's own update-4,000
state receives matched evaluation draws.

These are new task identities with 256 learned uniform MoG particles, sigma `.1`,
latent dimension 2, width 32, depth 2 and two critic Fourier frequencies. The
original sigma-`.025` `gaussian1d_acquisition` task and historical five-terminal
verdicts remain unchanged. The ordinary view is revision 7 with required counts
6 / 20 / 2. Screening remains provisional until calibration passes.

A zero-update confirmation of the fixed historical BCAP update-1,000 checkpoint
is being published separately. It supplies no ordinary qualification and retains
the original failed five-terminal acquisition verdict. The parent inventory run
will measure the new ordinary tasks under their exact bindings.

The public shared host is `experiments.forge.gaussian_tasks`; architecture
variants supply explicit task cards to `benchmarks.toy_audit.gaussian_smoke_study`.
All model/trainer/sample/restore execution requires CUDA. CPU reference scoring
and rendering are explicit exceptions. Constructor, target, trainer sampling,
primary evaluation and confirmation streams remain isolated and checkpointed.

## Fixed-checkpoint evidence

The original BCAP sigma-`.1` update-1,000 state passes a fresh independent CUDA
confirmation. This establishes a source-bound **can-pass** result; it does not
supply the new ordinary task's 24 paired observations or replace its inventory
execution.

| Full bound | Original scheduled draw | Independent same-state draw | Requirement |
| --- | ---: | ---: | --- |
| Samples | 4,096 | 4,096 | ≥ 4,096 |
| Finite fraction | 1 | 1 | 1 |
| Mean error / target sigma | .013037 | .009982 | ≤ .2 |
| Standard-deviation ratio | .957388 | .932642 | .8–1.2 |
| KS against N(2, .5²) | .042974 | .047590 | ≤ .05 |

The historical five-terminal acquisition remains **FAIL**. No new training
updates were used; the budget allowed one independent draw at this prespecified
checkpoint and no repeat. The complete original training curve and
[actual-training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-smoke/mog100-n256-gaussian1d_acquisition.gif)
retain their original source and verdict. The existing
[duration evidence](../tier1-prior-duration/README.md) still shows instability
through update 4,000; Tier 1 success makes no Tier 2 claim.

The CUDA draw and unchanged complete training state were saved before a compact
receipt serialization error (`KeyError: 'verdict'`; the historical key is
`full_verdict`). [The saved-output publisher](publish.py) recovers the compact
result by scoring that saved tensor and comparing the entire before/after
training state, excluding consumed evaluation streams. It performs no training,
model forward, sampling or repeat. The original error stdout and executed source
are retained in the archive. The numerical diagnostic loop duration was not
serialized; the 120-second reservation remains declared and no speed claim is
made. [Results](results.json) and [provenance](provenance.json) bind the recovery.

## Software verification and execution

The new Gaussian contracts have 22 passing checks, including CUDA stream
isolation, exact trained-state restoration/continuation, strict certified
artifact verification, and rejection of extra artifact files. The remaining
checks reject missing/off-cadence/duplicate observations, mismatched confirmation
states, nonfinite outputs, failed mechanisms, RNG deviations and malformed or
incomplete actual optimizer counts. The inherited exact Gaussian scorer's oracle
and destructive controls pass. The focused Forge/view/word/task-recipe regression
suite passes 93 checks.

An early integration fixture exposed a self-receipt artifact-manifest problem.
The final host stores tensors under `evaluator/` and the adapter receipt outside
that certified tree; strict artifact verification remains unchanged. Existing
architecture-study prefixes can be continued only with an explicit source-bound
amendment and byte-identical saved inputs; no prefix rerun is required.

Byte-exact restoration itself is verified. A separate pristine-state next-step
comparison found a near-rank-deficient polar/SVD sensitivity in the inherited
normalized optimizer: identical restored initial tensors could produce different
first hidden-layer corrections. The trained-state next-step comparison passes.
We therefore do not claim that every pristine first-step repeat is byte-exact;
this does not change the independently confirmed historical checkpoint.

Architecture diagnostics use the same public host with explicit task variants:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python -u -m benchmarks.toy_audit.gaussian_smoke_study \
  --task configs/forge/tasks/gaussian1d_smoke.json \
  --stability-task configs/forge/tasks/gaussian1d_stability.json \
  --candidate configs/forge/configurations/bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9.json \
  --through-stability --device cuda:0 --output runs/api/DECLARED_NEW_STUDY \
  > runs/api/DECLARED_NEW_STUDY.log 2>&1
```

This is a reproduction interface, not authorization for an undeclared new run.
Diagnostic continuation remains explicit after smoke failure and gives no
ordinary qualification. Ordinary Tier 2 requires the same candidate's compatible
passing smoke receipt and exact checkpoint. Every output/confirmation stream,
model, optimizer, history and data cursor is checkpointed. For saved-state
publication only, hydrate the archive and run:

```sh
/usr/bin/python reports/forge/gaussian-smoke-tier-split/publish.py
tail -F runs/api/gaussian-smoke-tier-split-v1/publication.log
```

The only generated goal leaderboard remains
[technique-inventory](../technique-inventory.md). The parent integration will
merge successful changes, run the current roster under the new bindings, execute
eligible Tier 2 families and regenerate that publication with its script.
