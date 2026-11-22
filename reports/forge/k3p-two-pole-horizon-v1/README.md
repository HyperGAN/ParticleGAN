# K3P two-pole budget and schedule diagnostic

**Increasing execution to 800 updates did not rescue the word-positive global recipe.** Its two variants FAIL movement; the existing movement control passes both. The stretched word run reaches a small-radius force balance, while the original-schedule run continues moving slowly. These different trajectories explain why extending time alone is insufficient in this measured round. Retain the original 80-update Tier 1 gate and current family selection; investigate existing critic/particle balance before another bounded global search.

| Unchanged global recipe | Schedule horizon | Movement at update 80 | Movement at update 800 | 800-update diagnostic |
| --- | ---: | ---: | ---: | --- |
| Word-positive, coefficient 170 / LR .0006 | 80 | .046871 | .089698 | FAIL |
| Word-positive, coefficient 170 / LR .0006 | 800 | .059627 | .111102 | FAIL |
| Movement control, coefficient 1 / LR .006375 | 80 | .592836 | 1.054229 | PASS |
| Movement control, coefficient 1 / LR .006375 | 800 | .108209 | .975970 | PASS |

The unchanged numerical bounds are mean absolute movement ≥ .3 and median critic input gradient ≤ 1. All four final gradient bounds pass. Each run executes 800 updates and the declared 24 scoring checks; both passing controls satisfy the five-check terminal requirement. Neither word run ever reaches .3. Extra observations at 80/200/400/800 and the original 24-check prefix are diagnostic and do not replace the scorer's cadence. [Exact final metrics, decisions and comparisons](summary.json) and [the frozen plan](plans.json) retain their bindings.

These are explicitly scoped diagnostic tasks, not new task-specific solutions. The word recipe is the unchanged `k3p-global-repair-v1`; its existing fields exactly match the normalized historical word-winning recipe. The control is unchanged `k3p--01eca360219ea5225a6e30a31800c7bbe70ca08c3800e259c17a67f0ea528ab5`. Original solution cards remain byte-identical. Study declarations replace only their bounded decision contracts. No candidate technique, initializer, prior, host objective, sampling law, ordinary task or main-view gate changed. Historical word evidence remains valid under its original identity and gives this diagnostic no new word qualification.

## What the saved forces show

[Independent force analysis](force-analysis.json) validates actual update tensors and the existing rowwise relativistic loss derivative against the saved critics, with maximum absolute error ≤ 3.73e-9. The task already adds `.02 * mean(x²)`, whose coordinate gradient is `.04*x/12`. The observer subtracts that exact term from the already-computed total generator gradient; it records the resulting adversarial force, actual learning rate/gain, displacement and critic payoff/penalty gradients without altering training.

For the stretched word run, peak movement is .136512 at update 183, then falls to .111102. At update 800, adversarial and L2 gradient norms are .001282865 and .001282828, with opposite directions (cosine approximately −1) and total-gradient norm approximately 1.22e-7. Their cancellation remains strong over the last 100 updates; movement decreases by .001818 in that window. The finite optimizer path and measured drift are retained, so small gradients alone are not presented as proof of convergence. The recorded force balance supports the local plateau diagnosis within this run.

The original-schedule word run has a different limit: it reaches the scheduled rate floor after 80 updates, retains outward net force, and gains .005946 movement over updates 701–800. It is still progressing slowly; the experiment does not prove it could never pass at a much larger budget. That uncertainty does not justify another unchanged continuation after this finite round.

Both word variants keep all 12 particles negative, with very small spread. The control finishes with six particles on each side. Clean zero-origin coordinates can separate through the existing paired relativistic payoff: `softplus(D(real_i) - D(x_i))` weights different real rows differently. Noise is not the only possible source of separation. Movement and sign balance remain distinct from a complete two-mode distribution-quality claim.

No critic guard clips occur in any of the four runs, despite eligibility after update 200. Applied gradients and actual update directions rule out an unconsumed optimizer setting or guard clipping as explanations here. Word-run critic payoff and penalty gradients oppose; the initial 1D real-gradient penalty prefactor is 85 for coefficient 170, compared with .5 for coefficient 1. This supports investigating critic shape and penalty balance. The recipes also differ in rates and noise, so this comparison does not isolate coefficient causality or establish a broken formulation.

## What changing the horizon means

Schedule horizon 80 retains the original schedule and adds 720 updates at its floors. Horizon 800 stretches the declared schedule bundle. For the noisy control, the input-noise window grows from 8 to 80 updates; learning-rate decay and LR-dependent K3P behavior also change. The paired arms start from identical component states, but the contrast is not an isolated LR intervention.

[Historical prefix controls](summary.json) verify every one of the 24 original 80-step observations and the full effective recipe for both schedule-80 continuations. Historical critic/optimizer checkpoints were unavailable, so full historical state equality is not claimed. Software parity tests independently verify complete model, optimizer, moment and RNG equality with and without observation hooks on small clean/noisy public-host runs.

## Recommendation and scope

Keep the original gate for now. This round does not show that replacing 80 with 800 would produce a global winner: the word-compatible recipe still fails, and these diagnostic results grant no ordinary qualification. The low-rate 80-step displacement bound remains an important screening-calibration limitation; it is separate from the measured longer-run force balance.

The next useful hypothesis should target existing global critic/particle balance, predicting stronger outward adversarial force at the same radius. Tune existing coefficients and critic/prior controls with a finite declaration, preserve the task's L2 objective, and evaluate one complete recipe across every ordinary task. Recipes already passing movement still face ring tails and unmeasured word acquisition. No seed sweep, new GAN technique, automatic further search, higher-tier execution or public-default adoption follows this diagnostic.

The [single current family leaderboard](../technique-inventory.md) remains the solution ranking. Diagnostic jobs have a separate scientific evidence-use marker, always qualify at tier zero, cannot fill ordinary/calibration cells, and produce compact concluded recall without another leaderboard. The [word-recipe readout](../records/readout-dd6ff59094d1d766c5b0fe2b.json) and [control readout](../records/readout-a91dbccf086504f9accaee6b.json) retain exact attempt identities.

## Evidence and reproduction

The four runs charged **34.666967 seconds** against 300 seconds per task, 600 per candidate and 1,200 campaign-wide. They ran sequentially on CPU with one Torch thread and named seed 0. Execution used commit `d376a1d0426dce945b51e7f662c0a65aa69ca740`, source digest `966d7d0b0cb55bd88c42791b1120bd27c02f184ee1b62ae97873cd5b3bb25bb0`, and 1,143 verified Git blobs. Cost is not a convergence-speed ranking.

Actual coordinate GIFs use retained updates and common fixed axes, showing the target poles and numerical movement/slope status: [word / schedule80](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/k3p-two-pole-horizon-v1/media/word_positive_schedule80.gif), [word / schedule800](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/k3p-two-pole-horizon-v1/media/word_positive_schedule800.gif), [control / schedule80](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/k3p-two-pole-horizon-v1/media/movement_control_schedule80.gif), [control / schedule800](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/k3p-two-pole-horizon-v1/media/movement_control_schedule800.gif). Media and forensics add zero training updates or sample draws.

[The immutable archive](archive.json) retains exact stdout, per-update traces, state checkpoints, certificates, historical reference envelopes, queue records, source snapshots, declarations, compact readouts and publication sources. [Independent archive verification](archive-audit.json) checks hashes and executed Git blobs. Bulk logs and tensors stay out of Git. Tail `runs/forge/k3p-two-pole-horizon-v1/worker.log`, its queue's `events.jsonl`, or an attempt's `run.log`/`force-trace.jsonl`.

After hydrating the exact original artifacts, the first two commands reproduce publication/analysis before archiving. The last command verifies the frozen archive:

```sh
/usr/bin/python reports/forge/k3p-two-pole-horizon-v1/publish.py
/usr/bin/python reports/forge/k3p-two-pole-horizon-v1/force_analysis.py
/usr/bin/python reports/forge/k3p-two-pole-horizon-v1/archive.py --verify --card reports/forge/k3p-two-pole-horizon-v1/archive.json
```

Publication is frozen once archived; use the archive verifier thereafter. The committed `prepare.py` and `run.py` retain the original setup/execution procedure and refuse preparation or fresh execution of the admitted study. Do not rerun unchanged science for publication or a merge.
