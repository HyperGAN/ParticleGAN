# Three-generator Lunar Lander world model

This example learns shuffled individual `(st, at, st+1)` records from continuous
`LunarLander-v3`. The experiment asks whether learning the joint distribution
helps predict the next observation. The main model retains our three-generator
solution and the MoG + bcap recipe. It is inspired by MisGAN's coordinated
generators; these records are fully observed pairs, with no learned missingness mask.

```text
One shared draw z from learned MoG1024:
G1 -> st
G2 -> at
G3 -> st+1

Observed input:
E(st, at) -> z_hat -> G3 -> predicted st+1
                    G1 -> reconstructed st
                    G2 -> reconstructed at

Synthetic composition:
G1 -> st --+
G2 -> at --+-> E -> z_hat -> G3 -> composed st+1

D_joint(st, at, st+1)
D_action(at)
D_state(st, current_role)
D_state(st+1, next_role)  # shared weights
```

Every G, E and D also receives eleven terrain heights at fixed horizontal
coordinates. This is **privileged terrain context**, beyond the standard
eight-coordinate observation. G1 and G3 output six continuous observation fields
and two contact logits; G2 outputs two bounded engine commands. All three Gs are
independent MLPs with the same latent draw, including its noise. G2 models the
collected action distribution; it is not a trained control policy.

E uses `particle_ae` routing to a learned component plus a bounded offset. It sees
only current observation, chosen action, and terrain. E → G3 gives a deterministic
continuous point prediction and contact probabilities. Random prior triples do
not establish calibrated conditional uncertainty.

## Reproduce the bounded comparison

Install the optional simulator and experiment dependencies in the repository's
Python environment. Some platforms require SWIG to build Box2D:

```bash
python -m pip install swig
python -m pip install -e '.[gym,experiments,dev]'
python -u experiments/collect_gym_transition.py
python -u experiments/probe_gym_simulator.py
python -u examples/gym_world_model.py --config configs/gym/lunar_lander/direct.yaml --device cuda:1
python -u examples/gym_world_model.py --config configs/gym/lunar_lander/reconstruction.yaml --device cuda:1
python -u examples/gym_world_model.py --config configs/gym/lunar_lander/adversarial.yaml --device cuda:1
```

Use fresh output directories for new runs; existing completed runs are protected.
The default example runs the full adversarial model. GPU 1 is the default for this
experiment; use `--device cpu` for a CPU smoke. Logs are flushed:

```bash
tail -F results/gym/lunar_lander/live.log
tail -F results/gym/lunar_lander/collection.log
```

The comparison trains each learned arm for 10,000 updates, batch 256, with
checkpoints at 1,000/2,500/5,000/10,000. Validation continuous MSE selects `best.pt`;
`final.pt` is reported separately. No experiments repeat only the training seed.

```bash
python -u experiments/evaluate_gym_transition.py \
  --checkpoint direct_best=results/gym/lunar_lander/direct/best.pt \
  --checkpoint direct_final=results/gym/lunar_lander/direct/final.pt \
  --checkpoint reconstruction_best=results/gym/lunar_lander/reconstruction/best.pt \
  --checkpoint reconstruction_final=results/gym/lunar_lander/reconstruction/final.pt \
  --checkpoint adversarial_best=results/gym/lunar_lander/adversarial/best.pt \
  --checkpoint adversarial_final=results/gym/lunar_lander/adversarial/final.pt
```

Evaluation automatically includes persistence. It verifies training-data and
checkpoint hashes and the inference source before ranking. Saved checkpoints
contain EMA inference weights and scalers, not optimizer resume state. Raw data
and checkpoints live in ignored `results/gym/lunar_lander/`; the numerical
[leaderboard](../reports/gym/lunar_lander/baseline/README.md) and
[readout](../reports/gym/lunar_lander/READOUT.md) are separate from the old route benchmark.

Generate the offline viewer and optional GIF after evaluation:

```bash
python experiments/render_gym_transition.py --gif
python experiments/diagnose_gym_transition.py
```

Open [the HTML demo](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/gym/lunar_lander/demo/index.html) locally. It offers
eleven recorded engine commands per fixed anchor, real/learned comparisons, and
the shared-latent transition gallery. The [downloadable bundle](../reports/gym/lunar_lander/demo/lunar_lander_demo.zip)
works without model checkpoints or a simulator installation.

## Data and simulator checks

The frozen dataset contains 32,768 train, 4,096 validation, and 8,192 test records.
Each anchor has its behavior command and three alternative commands spanning
main-engine off/on and lateral-engine off/left/right. Source episodes and terrain
instances are disjoint across splits. A fixed mix of heuristic, perturbed
heuristic, and exploratory behavior supplies approaches and failures; phase
sampling targets 50% flight, 35% approach, and 15% contact. Metadata records actual
counts and any redistribution when a phase has too few anchors.

Counterfactuals reset to the same episode seed and replay the exact command
prefix. This restores the full Box2D state and pre-step engine RNG. The trainer
does not load episode identities or prefixes. Collected trajectories are
provenance and evaluation references, never training sequences.

The adapter pins Gymnasium 1.2.3 and records the Box2D version and simulator source
hash. Tests verify replay through live contact and exact activation boundaries:
the main engine fires for `a[0] > 0`; lateral firing requires `abs(a[1]) > .5`.
Wind is disabled, but random engine dispersion remains. Articulated leg state,
contact solver state, and sleep state are not fully captured by eight observations.
The [simulator probe](../reports/gym/lunar_lander/simulator_probe.json) measures
noise variability at replayed worlds; this is not a proven Bayes-error floor.

## Losses and comparison capacity

Continuous coordinates use one scaler fitted on the union of training current
and next states; contacts remain binary. Actions have a separate training scaler.
Each state reconstruction combines mean standardized six-field MSE and mean
two-contact BCE, each weight 1. Real reconstruction averages current/action/next
roles; synthetic reconstruction averages current/action roles, with detached
targets and live encoder-input gradients. Both reconstruction losses have weight 1.

Fake adversarial contact values are Bernoulli bits with a straight-through
sigmoid derivative for G updates. D therefore cannot win just by distinguishing
real binary contacts from soft fake probabilities. Conditional metrics use
probabilities. Recursive feedback thresholds them at `p >= .5`.

The full arm uses Rp logistic losses, bcap for all four discriminator roles,
joint plus mean marginal generator feedback, and mean original/composed
adversarial feedback. It retains the MoG optimizer groups, calibrated fixed sigma,
raw-center spread, cosine schedule, and EMA. G/E width is 128; joint D width 256,
marginal D width 128; latent dimension 32; 1,024 learned components.

The reconstruction arm has identical G/E/prior initialization and capacity, with
the same reconstruction and prior losses. The direct supervised control uses the
same next-state loss and approximately matches E+G3+prior parameter count:
99,940 versus 100,040 parameters. Total G/E/prior capacity is 145,618; the full
model adds 210,307 discriminator parameters. Each learned arm draws 2,560,000
training records for generator/predictor updates; the GAN draws another 2,560,000
for D. This is a finite-data comparison, not a matched-compute comparison.
The current prediction helper evaluates all G branches before selecting G3;
reported throughput includes this overhead, while inference capacity denotes the
E+G3+prior dependency path.

## Reading the measurements and demo

The primary metric is test MSE over six continuous fields standardized by
training scales. Contact Brier/BCE and precision/recall stay separate. Matched
action differences measure engine response. Recursive 1/5/20/50-step evaluation
never refreshes real observations or snaps predictions to the terrain, and ends
at reference termination with counts reported.

Joint SW1 and coverage compare original/composed samples within fixed test terrain
contexts, equally weighted. Contact-pattern and phase frequencies accompany them.
Reference-half distances characterize finite-reference variation, not an iid
lower bound: some records share anchors. Arbitrary G1 states cannot uniquely
identify full Box2D worlds, so unrestricted prior triples receive distribution
scores rather than purported exact simulator-consistency scores.

The offline demo compares recorded real transitions with checkpoint predictions
under the same commands. Ghost leg positions are schematic. Recursive failures
remain visible. The model has no reward/termination head or planner; a successful
reference-controller landing is not evidence of learned model-based control.

## Live simulator driven by G2

To compare controllers generating commands for an actual running simulator:

```bash
python -u examples/gym_lander_live.py
```

Open **http://localhost:8787** and press **Play**. Pause holds the simulator at its
current step; Single Step advances once. Reset restarts the displayed seed.
The display uses Gymnasium's real rendered frames, including articulated legs.
The browser requests one physics step at a time and stops at termination or
truncation. It runs local inference on CPU by default, with no retraining.

```text
previous action = [-1, 0]  # engines off initially

repeat:
    E(st, previous action, terrain) -> z
    G2(z, terrain) -> at
    simulator.step(at) -> st+1
    previous action = at
```

This deliberately reuses the paired encoder with a **previous** command in its
action input. It is a prototype feedback loop, not the encoder's original
next-state prediction evaluation. The checkpoint was trained on a mixture of
controller and exploratory commands, with no landing objective. No G3 prediction
replaces the simulator's state. The default checkpoint is the validation-selected
three-generator GAN when no control results are available. The separate
[control experiment](gym-control.md) trains `E_control(st, previous at) -> G2`
with expert actions, comparing imitation alone with joint three-generator
training. When its controller manifest is present, the viewer defaults to the
validation-selected learned controller and offers all four controllers in a
selector. Switching controllers pauses and resets the same world. Use
`--controller original` for the prototype above, or `--port` / `--device`
to change serving options.

The local server holds one shared episode. Its flushed startup/reset/end log is:

```bash
tail -F results/gym/lunar_lander/live_controller_server.log
```
