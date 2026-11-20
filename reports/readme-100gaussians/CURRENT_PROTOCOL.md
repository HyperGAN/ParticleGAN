# Current-default README animation

This is one bounded illustration of the selected public default, resolved with
`get_recipe()` at execution. For **0.11.0** this is `unified_v11`. No technique
recipe field is tuned for the animation. The 0.11.0 illustration uses the
recipe's default 7,000 updates, batch 2,048, the Forge `grid100` task-owned
mixture-of-Gaussians particle prior (`MoGParticlePrior(num_particles=20000, z_dim=2, sigma=0.025, standardize=False)`),
the example's width-128, three-layer MLP generator and Fourier-feature critic,
and a uniform 10×10 mixture of Gaussians with σ = 0.03. This MoG prior cohort
is explicitly declared in `contract.json` and `summary-current.json` and is a
separate cohort from the archived 0.9.0 zero-sigma `ParticlePrior(20000)` runs.

Protocol seed is 0, with the public deterministic initializer on whole G/D/prior
modules (seeds 0/1/2). Constructor, data, latent, penalty, noise, model, evaluation
and reference draws have separate recorded streams. Initial and final checkpoints
include the trainer and every consumed stream. Plotting cannot consume training
RNGs. Samples are live and clean (`output_noise=0`, preserving the task prior's
intrinsic σ = 0.025); 4,096 fixed row IDs establish visual continuity.

The mode-acquisition gate requires 100/100 covered modes and at least 95% of 20,000
fixed evaluation draws within 3σ of a mode center, at each of the last five
scoring observations. The 7,000-update run has 24 evenly spaced scoring checks.
A covered mode contains at least ten such samples. An independent oracle draw
must pass and a collapsed control must fail before training. The declared
execution budget (at least `default.total_steps = 7,000`) is required; short
software smokes cannot pass. Additional media observations cannot remove or
change a scoring check.

This gate describes mode acquisition, not full Gaussian covariance calibration.
Final mode TV, sliced W1, tails and per-mode shape metrics are also reported.
The GIF includes the endpoint and a hold, without best-checkpoint selection.
Its caption and receipt preserve any failed gate. This supplies no Forge
qualification and makes no controlled comparison to the archived 0.9.0
`ParticlePrior(20000)` runs or historical K3P/seed-1234 runs.

Archived 0.9.0 budget note: the earlier 0.9.0 `bispf_optimized` illustration
used `ParticlePrior(20000)` for 7,000 updates plus one user-requested
continuation via `GANTrainer.extend_execution(20_000)` to 20,000 updates; both
its 7k and 20k `FAIL` receipts and GIFs remain archived in Git history under
their original commits and blob hashes.

The reproduction script (`record.py`) preserves backward-compatible CLI defaults
while exposing explicit task-owned prior flags (`--prior-kind mog --prior-sigma 0.025 --no-standardize`).
Bounded software smokes may use four updates to verify the public API, RNG
isolation, checkpoint and renderer. Raw logs, observations and checkpoints stay
outside Git. Re-render saved observations for visual iterations rather than
retraining.
