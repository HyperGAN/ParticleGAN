# Raw assignment on the remaining RHBI retention tasks

Raw assignment sustained rare-mode width for 19 observations. Moving-patch sharpness missed once, and saddle interior quality missed four times in the five terminal checks. Both feature-attraction recipes failed these two interpolation gates. Raw RHBI passed both. The comparison used 8,400 updates and 473.46 worker-seconds. [Five-task comparison](../README.md) · [Matched metrics and provenance](results.json)

**Unequal mass.** For the candidate, the smallest component eigenvalue ratio stayed between 0.416 and 0.529 over the terminal window, above the 0.15 floor. The 2% target component’s core width ratio was the same at each of those five observations. At update 1,200 it received 137 of 4,096 samples and overall mass TV was 0.0875 against a ceiling of 0.15. The parent also passed 19 consecutive observations. Raw RHBI fell to 0.133 and 0.142 at updates 1,100 and 1,150. Its final check passed and its full 24-observation trace exactly reproduced the previous raw control. [Width training GIF](width-training.gif)

**Moving patch.** Only the mean peak of the decoded chords failed for the candidate. It reached 0.77374 at update 800 against the 0.78 floor. Chord RMSE was 0.04565 against a ceiling of 0.10. The parent missed peak at update 734 and recovered to 0.83438 at update 800. Raw RHBI passed all five checks and finished at 0.85739. [Patch training GIF](patch-training.gif)

**Saddle.** The candidate passed once at update 767. At update 800, the pinned CUDA evaluator reduced 141 finite rows to a fraction of 0.99999994 and emitted placeholder metrics. Replaying the same live model and outputs gave 53/71 qualifying barycenters (74.65%, floor 78%), surface MAE 0.07314 (ceiling 0.09) and coverage of all 16 spatial bins. The genuine final miss was barycenter quality. The parent passed zero terminal checks and raw RHBI passed all five. The original candidate GIF includes the evaluator's placeholder frame. [Saddle training GIF](saddle-training.gif)

## What the learned neighborhoods explain

An exploratory check used each model's final generator and prior with the triangle or edge row IDs from each arm. On the candidate’s 71 saddle triangles, raw RHBI passed 59 interiors, the parent passed 49 and the candidate passed 53. On the parent's 74 triangles the candidate passed 59, exceeding the 78% floor on that metric. Raw RHBI exceeded the floor on all three triangle sets. For moving patch each model scored highest on its own edges, and the candidate’s own peak was already below the floor. Row identities are comparable because initial tables match, while the trained latent coordinates can differ. [All graph substitutions](results.json)

Keep the synthetic texture recovery and investigate the image and saddle misses within RHBI. A useful next probe would hold a saved model, batch and assignment fixed, then compare the effect of raw and feature attraction on the failing interiors. This would test whether to change the attraction term before spending a CIFAR run. CIFAR and untouched task-level confirmation remain unmeasured.

## Frozen comparison

Compare raw assignment and feature attraction with its immediate feature-assignment parent and raw RHBI-v12 on three existing tasks. The texture and sparse-arc measurements admitted these runs. Use the exact same three public recipes, prior policies and regularizers as in that study. Only the candidate’s detached marginal generator cross-assignment cost differs from its immediate parent. [Recipe and prior results](../README.md)

Run every arm on every task at protocol seed 0, using the task's deterministic initialization, data sequence, prior, clean live sampler, update budget and 24 observations. The hypothesis requires every declared bound at each of the last five observations on all three tasks. The study card's final patch peak prediction is one component of this decision. Control outcomes contextualize attribution. Reserve 7,200 worker-seconds for the nine cells.

| Task | Updates per arm | Question |
| --- | ---: | --- |
| Unequal mixture mass | 1,200 | Are mass, coverage and component widths sustained under clean sampling from the learned mixture prior? |
| Moving image patch | 800 | Are learned endpoints and decoded latent chords sharp, accurate and diverse? |
| Saddle barycenters | 800 | Do the learned endpoints and interior points cover and follow the saddle? |

A passive observer saves the initial state and five pre-evaluation states and hashes every training batch. Keep official grades from the pinned evaluator. For interpolation, separately check finite counts on the outputs of each restored live model and identify any sentinel metrics from the floating-point reduction. These diagnostics cannot change a grade. Stop before external CIFAR training if any candidate gate fails. [Task contracts](../../../../../configs/forge/tasks)

## Reproduction

Training used source commit `7d4b9d0b57b6b01ca2a55dc9a504ea337392d02a`. Each study binds the public recipe, task hashes, protocol and budget. The compact results include the attempt IDs and artifact hashes. Matched hashes cover the initial models and prior, all seen batches, other recipe fields, named streams, live weights and clean serving law. Fresh-process replay matched all 30 official terminal interpolation observations exactly. The diagnostic uses the same CUDA-generated tensors and moves the finiteness reduction to CPU. It checks an integer count of all endpoint and primary interior rows. [Candidate receipt](../../../records/readout-47ae4e9c8c2e782e4747f222.json) · [Parent receipt](../../../records/readout-e77a1c6bb3e4b9bf185d0dc1.json) · [Raw receipt](../../../records/readout-4eed35ca04c58e48dcea5195.json)

Run from the repository root with access to the local Forge artifact archive and a compatible CUDA device. The reducer independently grades each gate, verifies retained artifact hashes and checks the diagonal of both graph comparisons against the corresponding native scores.

```sh
study=reports/forge/rhbi-texture-feature/raw-assignment/retention
PYTHONPATH=. python "$study/compare.py" /path/to/runs/forge --media "$study" > /tmp/rhbi-retention.json
cmp /tmp/rhbi-retention.json "$study/results.json"
```

Tail `runs/forge/rhbi-raw-assignment-retention-v1/progress.jsonl` for campaign events, or an attempt’s `run.log` for per-observation output.
