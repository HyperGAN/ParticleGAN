# Resume here after compact

## Latest: joint timestep/class UCD scout completed

Read [joint_ucd/READOUT.md](joint_ucd/READOUT.md). Both56k runs and both GPU
smokes finished successfully. Joint UCD is competitive, not a clear winner.
User prefers it as a tie-break and it is now the CIFAR candidate default.
The next prepared experiment is the full CIFAR class-only versus joint pair;
see [CIFAR handoff](../cifar-ddgan/RUNBOOK.md). Nothing is active or queued.
The historical statements below about deferring toy work are superseded.


## Next task switched to CIFAR-10

The user switched the next experiment to an image baseline and confirmed
**DCGAN-style convolutional networks inside the four-step DDGAN**, with UCD,
learned latent particles, and Gaussian step noise. Establish that baseline,
then transition to a transformer. The longer-term target is approximately
3.8 CIFAR FID. Further toy sweeps are deferred.

**Resume from [the CIFAR handoff](../cifar-ddgan/RUNBOOK.md).** It records the
confirmed formulation, planned YAML/no-argument entry point, repo/dependency
inspection, evaluation protocol needs, GPU workflow, and implementation order.
CIFAR implementation and baseline rounds are complete; the CIFAR handoff records current results.
The toy work was committed and pushed as `9a4aff0` on `master`.

## Current status: selected default ready for the next round

The user explicitly selected **DDGAN + UCD, learned latent prior, Gaussian
reverse-step noise** as the new experiment's working default after round two.
This is a chosen baseline, not a claim of universal superiority.

Name: **100 Gaussians DDGAN + UCD**.
Script: `experiments/train_denoising.py`.
Review/edit the complete config: `configs/denoising/ddgan_ucd.yaml`.
Run with no arguments:

```bash
.venv/bin/python experiments/train_denoising.py
```

The no-argument CLI reads that YAML; `--config PATH` remains available. Selected
settings exactly match `budget56k/ddgan_ucd_learned_56k_s24002.yaml` apart from
output directory: 56000 updates, seed 24002, four classes/four reverse steps,
20000 learned 4-D latent particles, Gaussian step noise, UCD lambda .02, and
the established bcap/optimizer recipe. Outputs go to
`results/denoising/ddgan_ucd`. Trainer literal DEFAULTS mirror this config.

**No experiments launched after selecting defaults.** Existing eight denoising
tests pass. Mocked training dispatch verifies no-argument CLI from an unrelated
working directory and explicit config overrides. The default config matches
the tested run, and all 65 historical screen/diagnostic configs regenerate
unchanged (verified in a temporary directory). Generators explicitly preserve
their historical steps/D/prior-table choices as CLI defaults evolve.

Changing the trainer defaults/docstring/CLI changes the source fingerprint for
future runs. The training algorithm itself is unchanged. Old results keep their
old source archives; use a new stage for the next experiment round and compare
source revisions explicitly. The historical note below about a 4096 fallback
is superseded: the trainer fallback is now the selected 20000-particle model.

Latest user research direction: noise may let each latent particle support more
useful outputs, and the network may need more training to exploit that capacity.
Suggested diagnostic: hold starting noise and latent-particle sequence fixed,
vary only step noise, and track distinct valid outputs, within-mode spread,
and invalid bridges over training. Frozen-checkpoint covariance substitutions
do not establish whether learned particles helped during training. Preserve this
question alongside the next-round ideas in budget56k/READOUT.md.

## Completed experiments: round two (2026-09-14)

**No active runs. 17 new runs, zero failures; 89 successful training runs total.**
Both GPUs used (9 new runs on GPU 0, 8 on GPU 1). User authorized continuing
experiments after compact; nine diagnostics, six 56k-budget variants, and two
28k confirmation seeds finished. The latter overlapped the 56k grid. Main log:
`results/denoising/diagnostics.log`. No training source changed: all new results
match the original trainer/lib/runner fingerprint and retain the audited bcap
recipe. Historical trainer fallback stays 4096; new explicit configs use 20000.

Start with **[round-two readout](budget56k/READOUT.md)** and
[56k samples](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/budget56k/samples.png). Key results:

- 28k UCD DDGAN + learned latent + Gaussian step noise: **77.8 ± 1.6% HQ**,
  100 modes in all three seeds 24001–24003, versus ~1% HQ at 7k.
- 56k seed 24002: one-shot learned-particle GAN 99.8% HQ / .103 mode TV;
  DDGAN concat Gaussian latent 89.1% / .065; concat learned latent 88.9% / .072;
  UCD learned latent + Gaussian noise 91.2% / .059;
  UCD + fixed step noise 90.9% / .065; UCD + learned step noise 93.4% / .070.
  All cover 100 modes. These 56k rankings are one-seed exploratory findings.
- Learned noise covariance eigenvalues shrink to .250/.354. For this frozen
  checkpoint, three evaluation draws give learned-table HQ 93.48% versus
  covariance-matched Gaussian 93.22%; SW1 .126 versus .132. Covariance explains
  much of the inference benefit; it does NOT establish equivalence of training.
- Oracle chain probes: UCD Gaussian-noise final-step-only learned HQ rises
  4.38% -> 92.43% -> 96.43% at 7k/28k/56k. Its full learned chain reaches
  .93% -> 78.06% -> 91.25%. Final-step error and earlier-step inputs both matter.
- Fixed latent one-shot control 19.3% at 7k; class-free DDGAN 10.5% at 28k;
  one-step DDGAN 99.2% at 7k (close to a GAN with extra random input), two-step
  DDGAN 43.0% at 14k. Details and caveats in diagnostics/READOUT.md.
- No universal winner: GAN is sharper, DDGAN has better mode proportions;
  all have calibration defects. No default baked in, no pure diffusion added.

Next candidates: repeat 56k comparisons on more training seeds, train a simple
learned Gaussian noise scale/covariance control, then target timestep allocation
or denoiser parameterization. Keep bcap established defaults; do not start a
broad hyperparameter search. Extending a run also extends high-LR exposure,
since cosine decay starts at 60% of total updates. Saved checkpoints are EMA
for evaluation, not optimizer resumes. Changing trainer/lib for a next stage
requires a new revision and coherent comparison manifests.

New helpers (all outside training fingerprint):
- `experiments/gen_denoising_diagnostics.py --stage diagnostics|budget56k|confirm28k --seeds ...`
- `experiments/plot_denoising_diagnostics.py --manifest ... --out ... --seed 24002`
- `experiments/probe_denoising_chain.py --checkpoint ... --out DIRECTORY` (CUDA)
- `experiments/probe_denoising_noise.py --checkpoint ... --out FILE --seed 99000` (CUDA)
  Noise counterfactuals use one frozen learned-noise checkpoint. Evaluation seeds
  99000/99010/99020 each draw 20k samples. See budget56k/noise_probe_summary.json.
- Analyzer tables now show updates, transition count, classes; GAN noise is N/A.
  Old screen sample labels also corrected to N/A. Every new script compiles;
  all 17 runner certificates and the unchanged training fingerprint verified.

Exact configs under `configs/denoising/{diagnostics,budget56k,confirm28k}`.
The confirm28k `manifest.json` has two new runs; `comparison_manifest.json`
adds the seed-24002 28k diagnostic, giving the three-seed comparison. Reports
are separate by stage, same training source provenance. No experiments left
running; no commits made; preserve unrelated `.claude/` and `sparse-ucd.log`.

## Historical status: first round complete (2026-09-14)

The user resumed and authorized running experiments, then requested a subagent
audit of the established `100gaussians.py` bcap defaults. All 48 initial runs
finished, then 24 learned-prior runs were corrected from 4096 to 20,000 latent
particles and finished. Zero failures; both physical GPUs used, two workers
per GPU. No active runs remain. The original pause/readiness notes below are
historical.

Start with `screen_20k/READOUT.md`, `screen_20k/TABLE.md`, and
`screen_20k/samples.png`. `BCAP_AUDIT.md` records the independent audit: exact
penalty values and D gradients verified, inherited recipe matches apart from
the corrected latent-table count. Training sources were never edited during
these runs. Future configs should explicitly use `num_particles: 20000`;
the trainer's historical fallback remains 4096 to preserve source provenance.

The authoritative comparison manifest is
`configs/denoising/screen_20k/manifest.json` (24 reused Gaussian controls plus
24 corrected learned runs). `reruns.json` contains just the latter. Results
are certified under one source fingerprint. Analyzer prior contrasts ignore
the unused Gaussian table count, record both counts, and reject ambiguity.

Results: concat one-shot GAN + learned prior achieves 99.2% joint HQ and all
100 modes across three seeds, but has shape/tail/mass defects. UCD worsens
this baseline. All diffusion GAN cells remain around 1% HQ; neither latent
nor step-noise particles rescue them at 7000 updates. Conditional SW1 alone
can favor blurred outputs: inspect HQ, class fidelity, TV, and shape too.
Next round: targeted denoiser/last-transition, timestep exposure, class-free,
and fixed latent controls, retaining the known bcap recipe. Do not bake in a
diffusion default based on this screen; no model default has been replaced.

Tail-friendly helper (use a distinct scheduler log per round):

```bash
.venv/bin/python -u experiments/follow_grid.py \
  --root results/denoising/screen_20k --log results/denoising/screen.log \
  --runner-log results/denoising/screen_20k.runner.log -- \
  --config_manifest configs/denoising/screen_20k/reruns.json \
  --trainer experiments/train_denoising.py --gpus 0,1 --workers_per_gpu 2
```

Rebuild reports with `experiments/analyze_denoising.py --manifest
configs/denoising/screen_20k/manifest.json --out reports/denoising-toy/screen_20k`;
the sample panel uses `experiments/plot_denoising_screen.py` with the same
manifest and `--out reports/denoising-toy/screen_20k/samples.png`.

## Historical preparation notes

The user wants to scope **diffusion GAN / UCD / latent particles / learned
diffusion-step noise**, then bake in the best confirmed performer. They hope
diffusion GAN wins, but explicitly want to question the ingredients and follow
the evidence. **No pure diffusion.** They authorized both GPUs and asked for
repeatable config files. They subsequently asked to pause when the code was
ready, before full experiments, so they could run `/compact` and proceed in
rounds. Only correctness checks and short GPU smoke tests have been run.

## Implemented

- `lib/denoising_toy.py`: four-class, 100-Gaussian grid; Gaussian forward
  schedule; exact clean posterior oracle; one-shot and diffusion GAN MLPs;
  independent Gaussian/fixed/learned latent and reverse-noise sources; metrics.
- `experiments/train_denoising.py`: CUDA-only trainer, seeded independent RNG
  streams, matched relativistic GAN losses and cap penalties, EMA of G and
  learned tables, complete config/environment/provenance, final samples and
  checkpoint, marginal and conditional evaluation, plots.
- `experiments/gen_denoising_configs.py`: all concrete YAML configurations.
- `experiments/analyze_denoising.py`: verifies completed-run certificates;
  per-cell means/SDs, paired effects, and learning curves. It rejects mixtures
  of source revisions.
- `tests/test_denoising.py`: posterior correctness, final-step/noise gradients,
  finite-table sampling, UCD conditioning, metric failure cases, validation.
- Existing `experiments/run_grid.py` supplies concurrent GPU scheduling,
  source fingerprints, output locks, preserved old attempts, and certified reuse.

Neither the old `100gaussians.py` default nor its results have been replaced.
Results and checkpoints live under gitignored `results/denoising/`.

## Exact factor meanings

- **One-shot / diffusion GAN:** `model: gan / ddgan`. DDGAN predicts clean
  `x0_hat` from `(xt, z, t, c)`, then constructs the reverse transition using
  Gaussian posterior coefficients. All training is adversarial.
- **UCD:** `d_mode: concat / ucd`. UCD removes the semantic class from the
  backbone, uses four class scores with label-indexed adversarial loss, and
  adds real/fake CE (`ucd_lambda: 0.02`). It retains `xt,t` in the diffusion GAN.
  This is an adaptation, not a proof that continuous denoising conditions can
  be removed. Noisy transitions may make CE harmful; `ucd_lambda: 0` and
  `drop_xt: true` are available follow-up controls.
- **Latent prior:** `prior: gaussian / fixed / learned / zero`, sampled as
  the generator's auxiliary `z`. Initial factorial uses Gaussian and learned;
  `fixed` is a follow-up control. Shared 4-D latent table, 4,096 particles.
- **Step noise:** `noise: gaussian / fixed / learned / zero`, sampled independently
  as `eta` in `A_t*x0_hat + B_t*xt + sqrt(beta_tilde)*eta`. It is NOT jitter
  around latent particles. Shared 2-D noise table, 1,024 particles. Learned
  rows receive adversarial gradients. Its last-step multiplier is zero.
- Forward corruption and starting `x_T` remain fresh Gaussian. Changing forward
  noise would change the target and invalidate these Gaussian coefficients.

## Ready configs and commands

`configs/denoising/screen/manifest.json` lists **48 complete YAML configs**:
4 one-shot + 12 diffusion GAN cells, each at seeds 24001–24003, 7,000 updates.
Smoke seed 23999 is separate. The main default is width 128, depth 3, batch 256,
Rp logistic, Fourier-2 D, cap penalty 1, LR .0006, EMA .995. This is a starting
recipe, not an optimized diffusion GAN claim.

Full initial factorial, when the user resumes:

```bash
.venv/bin/python experiments/run_grid.py \
  --config_manifest configs/denoising/screen/manifest.json \
  --trainer experiments/train_denoising.py \
  --gpus 0,1 --workers_per_gpu 2
```

For rounds, first create a separate manifest selecting desired existing YAML
paths (e.g. the ordinary-Gaussian reverse-noise cells), then pass that manifest.
Do not edit configs in place after runs. New settings get new config/output
directories. Do not edit trainer/lib sources during a running grid: the runner
will correctly reject results whose source changes in flight.

```bash
.venv/bin/python experiments/analyze_denoising.py \
  --manifest configs/denoising/screen/manifest.json \
  --out reports/denoising-toy/screen
```

The analyzer reports missing runs, so it can collect a partially completed
factorial. Analyze intentionally different source revisions separately.

## Before choosing a default

Compare learning speed and final quality, not just a single scalar ranking.
Report joint HQ, all-mode coverage, conditional SW1, class accuracy, mode mass,
core width, both covariance eigenvalues, tails, and oracle transition error.
The final checkpoint is for evaluation, not optimizer-resume training.

After screening, run equal tuning budgets and targeted controls (frozen latent
table, UCD without CE, noise-moment regularization, two/eight reverse steps,
class-free generation). Confirm selected candidates on fresh seeds, e.g.
25001–25003, before changing any default. Consider a transfer target with
nonuniform weights/anisotropy; this extension is designed but not implemented.
The original recipe was tuned for particles, so losing with its defaults alone
does not establish that diffusion GAN or a Gaussian latent is inferior.

GPU smoke timing is a pipeline check, not a convergence or performance result.
Measure full-run throughput and GPU utilization before increasing concurrency.

## Verified before this pause

- Full test suite: **62 passed, 22 subtests passed**. One existing warning in
  `test_regularizers.py` concerns converting a gradient-bearing tensor to float.
- Final smoke runs: 200 updates each, GAN on physical GPU 0 and DDGAN + UCD +
  both learned tables on physical GPU 1 (both RTX A6000). About 2.2–2.4 seconds
  for training and 6–7 seconds including startup, final evaluation, and plots.
- Both smoke runs have certified completion, exact `source.zip` archives,
  saved configs, samples, and final evaluation checkpoints. The analyzer loaded
  both successfully. Smoke metrics are not convergence evidence.
- All 48 full configs passed runner preflight. **Zero full-sweep runs launched.**
- The first smoke attempt exposed a metric keyword mismatch, now fixed and
  covered by the new metric test. Old failed attempts are preserved by the
  runner under the smoke directory's `.run_grid_history`.
