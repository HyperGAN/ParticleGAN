# DDGAN conditional-law repair on the selected BISPF recipe

Three bounded repair rounds concluded with **seven full-budget FAILs and two
resource-INCOMPLETE attempts**. The energy objective and annealing substantially
improve independent generation, including five passing Gaussian marginal checks,
but no one global recipe passes either complete task. Conditional distributions
and bimodal local spread remain unresolved. This is a completed negative
research result, with no ordinary qualification or public-default credit.

The [single current leaderboard](../leaderboards/ddgan_conditional_repair.md)
retains every measured failure and incomplete attempt, numerical endpoints and
actual-training GIFs. [Compact results](results.json) contain exact failed bounds,
full-budget and terminal-window status, source/task/recipe identities, model and
batch hashes, optimizer guards, actual artifact hashes and measured costs.

## Diagnosis and proposed mechanism

The baseline is exactly `bispf-optimized-quintic-v1`, selected by the technique
inventory; its declared recipe SHA is
`60ba0af155bfe843f31fc11606a9b37d690b74be69ffd7ea4243c4b0a891541e`.
Its historical CPU Gaussian FAIL and blocked bimodal full run retain their
original identity. New matched GPU 0 runs provide this comparison.

The [saved-state investigation](../ddgan-failure-investigation/README.md) found
low-noise conditional errors at central and fresh contexts. High-noise reverse
kernels masked poor clean predictions. Local critic gradients opposed needed
mean correction on seven of eight fixed panels; they do not establish actual
parameter-step direction, critic lag or a temporal cause.

The [DDGAN paper](https://arxiv.org/abs/2112.07804) and
[official implementation](https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py)
model reverse transitions with a stochastic conditional clean predictor and a
Gaussian posterior kernel. This experiment retains that transition game and all
four reverse sampling steps. It adds the negative beta-1
[energy score, equation 22](https://doi.org/10.1198/016214506000001437):

`mean ||Xi-y|| - sum(i != j)||Xi-Xj|| / (2*K*(K-1))`.

Eight independent prior draws share each existing generator training context;
`y` is that forward pair's real clean example. The off-diagonal U-statistic
avoids finite-ensemble underdispersion bias, while separation rewards conditional
spread. No analytic posterior, privileged target parameters or evaluation query
enters training. The public `conditional_energy_score` owns this sample-only
loss. The existing public DDGAN fixture, Recipe factories, learned MoG, SBNP,
Born–Infeld critic penalty, DualNorm and ordered UpdatePolicy own actual updates.
The adversarial objective retains coefficient 1 throughout.

## Frozen comparison and bounded rounds

Both tasks retain seed 0 and public deterministic initialization; two-layer
width-32 G/D MLPs; 128 initial four-dimensional MoG locations with absolute
sigma 0.25 and no standardization; batch 64; separate D/G clean and timestep
streams; the schedule `[1,.9,.5,.05,.0001]`; clean live four-step ancestral
sampling with fresh latent, kernel and reverse noise at every step; 2,048
samples per panel; 24 scheduled observations; and five passing terminal checks.
Gaussian trains 2,500 updates and bimodal 5,000. Every original mean, covariance,
projection-CDF, mode-mass, quality and local-spread gate remains exact.

Auxiliary draws have independent checkpointed latent/kernel streams. Tests and
the exporter verify identical initial model hashes and complete consumed clean
and timestep batch digests across recipe arms. Evaluation leaves training state
and RNGs unchanged. The scientific code digest is identical across all rounds:
`db83af24303b269185064c69aa6837b3c56fe1fa5460b7da39aecc0672c0fe32`.
One global recipe is used across both tasks in each round.

- Round 1 adds only conditional energy weight 1 with 8 draws to the selected
  recipe. Ready studies `ddgan-repair-baseline-r1` and `ddgan-repair-energy-r1`
  reserved 2,400 seconds before execution at commit `71275a4c7`.
- Round 2 retains those objectives and globally changes learning rate
  `0.012 -> 0.003`, cosine annealing start `0.6 -> 0.3`, and positive network/prior
  floors `1 -> 0.05`. Studies `ddgan-repair-annealed-r2` and
  `ddgan-repair-baseline-wall-r2` were frozen at `34b7a7d6b`.
- Round 3 changes only `constraint_geometry_mode: direction_blend -> none`,
  minimizing the same summed GAN, energy and SBNP objectives with the same rates
  and core BISPF mechanisms. It removes the optional per-objective common-descent
  blend. This tests whether that constraint limits conditional correction; its
  conflict counts alone do not prove a cause or a generic nonascent guarantee.
  Study `ddgan-repair-unconstrained-r3` was frozen at `223d24568` before execution.

Round-1 bimodal workers exhausted their frozen 600-second allowance before
5,000 updates. The last actual logged observations were baseline 3,542 and
energy 3,334; exact terminal update counts, checkpoints and full receipts were
not written before supervisor termination. They remain **INCOMPLETE resource
outcomes**, with no full-budget quality verdict or substituted GIF.

The explicit timeout-only bimodal variant raises allowance to 1,200 seconds,
holding all 5,000 updates and scientific gates fixed. View revision 1 is
[archived byte-for-byte](../../../configs/forge/view-history/ddgan-conditional-repair-v1-v1.json).
Revision 2 selects that resource variant and the unchanged Gaussian as research
diagnostics. The completed Gaussian baseline is reused exactly, never rerun.
Ordinary deeper-task prerequisites remain failed; these diagnostics grant no
qualification and cannot fill the original blocked cells.

After the resource failures, the finite amendment permits at most 9 attempts,
35,000 maximum declared generator updates including incomplete allowances, and
7,800 reserved seconds across all rounds. Round 2 reserves up to 3,600 seconds;
the final round reserves 1,800. No active cap, scientific gate or recorded result
is relaxed. GPU 0 is shared and under load; costs are recorded without a speed
comparison. No seed trial, continuation or automatic extension follows.

## Measured findings and current recommendation

The matched Gaussian baseline fails with final CDF 0.5905, mean error 1.4801 and
worst conditional CDF 0.6144. Round 1 improves those endpoints to 0.0937, 0.2198
and 0.1289, but covariance maximum 1.5563 exceeds 1.4. Its five terminal final
CDF values span 0.0578–0.2836, and none passes every task bound. Endpoint
improvement is not sustained success. The saved optimizer records 1,448 blended
conflict steps and zero Pareto stalls; [the diagnosis](round1-diagnosis.json)
adds no updates or sampling.

Round 2 stabilizes the Gaussian marginal: all five final-chain checks pass,
with CDF 0.0451–0.0855. Endpoint CDF is 0.0617, mean error 0.1348 and covariance
[0.9905,1.0117]. The endpoint's remaining failures are fixed panel `t1_q0`:
mean error 0.6195, covariance maximum 1.4775 and CDF 0.1911. Its optimizer records
1,605 blended conflicts and zero Pareto stalls.
[Saved-state readout](round2-diagnosis.json) adds no updates or sampling.

The complete matched bimodal baseline fails with final CDF 0.3738, worst
conditional CDF 0.4728, mean error 0.7641, local mean error 1.066 sigma and minimum
local covariance 0.3522. Round 2 improves final CDF to 0.0450, mean error to 0.0376,
local mean error to 0.208 sigma and worst conditional CDF to 0.1272. It still
fails: high-quality coverage 0.9390 is below 0.95 and maximum local covariance
1.7867 exceeds 1.5, alongside conditional and sustained-window bounds.

Keep every complete task labelled FAIL and every capped attempt INCOMPLETE.
Do not select separate recipes per target or promote passing marginal subgates
to a task pass. Disabling the optional direction blend does not repair the conditional
failure: it worsens the Gaussian fixed query. A next substantive
question would test explicit timestep conditioning or task capacity in a new
architecture cohort, with a matched selected-recipe control and the same
scientific gates. Current evidence does not establish a capacity limit. Stop
these exact revisions; no additional tuning, seed trial or promotion follows.

The complete round-3 Gaussian also fails its fixed `t1_q0` panel. Marginal CDF
is 0.0316 and mean error 0.0113, but conditional CDF worsens to 0.3707, mean error
to 1.1197 and covariance maximum to 1.6634. Round-3 bimodal CDF is 0.0435, mean
error 0.0146 and worst conditional CDF 0.1024, while quality 0.9453, local
covariance maximum 1.6982 and conditional covariance/mean bounds still fail.
Neither task has a passing five-observation terminal window.

All nine attempts cost **4,793.664 supervised seconds**, including capped work,
startup, training, scoring, media and independent grading. Actual attempt
reservations total 7,200 seconds within the 7,800-second campaign ceiling.
The exporter verifies exact matched model initialization and complete target/
timestep sequences, one global recipe per candidate, seven real full-training
GIFs and finite complete checkpoints. Costs supply accounting only.

## Software validation and reproduction

**72 focused tests pass**, covering proper-score oracle versus collapsed/wrong
mean/wrong spread controls, off-diagonal normalization, independent same-context
draws, rejection of analytic-posterior leakage, public component execution,
exact checkpoint replay for both optimizer geometries, unchanged initial models
and target batches, observation purity, and unsupported-host refusal.
**124 training/recipe/model-toy regression tests pass**. The committed-page
freshness guard passes after a display-only publication refresh; a
[provenance receipt](publication-refresh.json) verifies every registered
scientific row and selected recipe remains unchanged. Forge declarations
validate and compiled memory is CURRENT. Original DDGAN scorer controls reject wrong contexts, unconditional
samples, posterior-mean collapse, mode collapse and nonfinite output at the
actual evaluation sample count. The source-only compatibility migration retains
[all four original task Git blobs and source hashes](source-migration.json);
no historical receipt or qualification was rewritten or rerun.

**66 strict receipt-compatibility checks pass** against frozen develop and the
maintained full-packet expectations. Inactive conditional-energy defaults are
omitted from historical applied packets while older fields remain present. A
[source-only receipt migration](receipt-compatibility-migration.json) preserves
the complete revision ancestry and exact previous Git blobs for ten current
task bindings; ten historical prior-refactor task cards remain byte-identical.
This software repair changes no recorded training source, metrics or verdicts.

Use a detached checkout of the pinned preregistration commit for the desired
round and a new queue directory. For example, round 3 uses the public Forge
pipeline and the repository's Python environment:

```sh
python -m experiments.forge --queue-root /outside-git/ddgan-reproduction/queue \
  plan bispf-ddgan-energy-unconstrained-v3 --study ddgan-repair-unconstrained-r3
python -m experiments.forge --queue-root /outside-git/ddgan-reproduction/queue \
  enqueue bispf-ddgan-energy-unconstrained-v3 --study ddgan-repair-unconstrained-r3
python -m experiments.forge --queue-root /outside-git/ddgan-reproduction/queue \
  drain --gpus 0 --allow-sharing --workers-per-gpu 2 \
  --campaign ddgan-conditional-repair-r3
```

`export.py` reads saved requests/receipts, verifies artifact hashes and shared
model/batch/task/global-recipe invariants, and copies actual saved GIFs. It
constructs no model, executes no update, draws no sample and does not regrade.
It overwrites only this goal's current compact publication and leaderboard.

Raw stdout, per-observation arrays, checkpoints and traces stay outside Git.
The single progress log aggregates real job steps and statuses:

```sh
tail -F /nvme/dev/particlegan/artifacts/ddgan-bispf-20261019/progress.log
```

Full execution logs are under
`/nvme/dev/particlegan/ParticleGAN/runs/forge/ddgan-conditional-repair-r{1,2,3}/*/run.log`.
