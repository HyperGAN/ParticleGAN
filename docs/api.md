# ParticleGAN API reference

ParticleGAN provides independent PyTorch priors, losses, and diffusion helpers.
You supply models and data; use an optional GAN trainer or compose your own loop. Install the package
with `python -m pip install particlegan`; the core dependencies are PyTorch and NumPy.

## A minimal training loop

This complete example learns a synthetic 2D distribution. Replace `real_batch`
with your pipeline and the MLPs with your networks. The helper applies one
shared recipe to optimizers, losses, regularization, decay and EMA. The recipe
contains configuration and component factories; the separately constructed
trainer owns the training lifecycle.

```python
import torch
from torch import nn
from particlegan import BatchDistanceDiscriminator, GANTrainer, get_recipe, init

torch.manual_seed(0)
device = torch.device("cpu")
recipe = get_recipe(total_steps=1000)
G = nn.Sequential(nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(.2),
                  nn.Linear(64, 64), nn.LeakyReLU(.2), nn.Linear(64, 2)).to(device)
D = BatchDistanceDiscriminator().to(device)
init.deterministic_orthogonal_(G, seed=0)   # optional; see Initialization
init.deterministic_orthogonal_(D, seed=1)
prior = init.deterministic_orthogonal_(recipe.make_prior()).to(device)
trainer = GANTrainer(recipe, G, D, prior=prior, seed=0)

def real_batch():
    return .2 * torch.randn(recipe.batch_size, 2, device=device) + 1

for step in range(recipe.total_steps):
    stats = trainer.step(real_batch(), generator_real=real_batch)
    if (step + 1) % 100 == 0:
        print(step + 1, stats["loss_d"].item(), stats["loss_g"].item(), flush=True)

samples = trainer.sample(256)            # Live weights by default.
ema_samples = trainer.sample(256, ema=True)
torch.save(trainer.state_dict(), "trainer.pt")
```

[The runnable example](../examples/quickstart_gan.py) adds CLI options, flushed
JSON logs and a data-RNG checkpoint. To verify continuation:

```bash
python -u examples/quickstart_gan.py --steps 1000 --stop-after 500 --output run.pt
python -u examples/quickstart_gan.py --steps 1000 --resume run.pt --output run.pt
```

The explicit [component-based loop](../examples/pytorch_loop.py) remains available
for applications that manage their own updates.

For E22, use the installed `get_recipe("e22", **task_overrides)` preset and
`E22Policy` to coordinate a caller-owned loop. The policy used by `GANTrainer`
owns controller observations, group learning rates, row evidence, birth/death,
learned output noise and serving averages. Its lifecycle hooks, checkpoint
contract and served snapshots are documented in [the E22 guide](e22.md), with
a [runnable external loop](../examples/e22_external_loop.py).
Conditional densely blended banks use `get_recipe("e22_routed", ...)` and
an explicit `RoutedRows` binding, with `RoutedBatch` inputs providing paired
targets and separate guard contexts. The [routed adaptation](e22_routed.md)
defines its evidence and move rules and includes a paired-error example.
For multiple token routing sites in a single model forward, provide
`RoutedRows(model_forward=..., features=..., sites=(...))`. The callback uses
`routing.mix(site_name, logits)` at each declared site with the explicit
candidate table and row state. Each counterfactual reruns the entire model;
evidence and guards evaluate its final output. See the
[shared-bank site contract](e22_routed_sites.md) for perturbation placement,
usage attribution and a two-site example.
`RoutedRows(probe_interval=K)` schedules costly probes, proposals and guards at
least K observed updates apart (default 1), independently of
`min_observations`. Its observation/probe clocks are checkpointed; gradient
evidence continues every update. `output_error_guard=True` adds clean paired
output-MSE protection on separate guard contexts for both fast and averaged
models. Output tolerances are `max_output_error_increase` and
`max_output_context_harm`, each measured in per-context MSE units and defaulting
to zero. Routed row moves support Adam, AdamW and `K3PGeneratorAdam`: both split
rows inherit half-parent first moments and quarter-parent second/AMSGrad
moments while preserving the optimizer age.
The [whole-model checkpoint replay example](e22_routed_sites.md#activation-checkpointed-whole-model-replay)
recreates routing on recomputation and restores a private DV12 stream without
advancing the training stream or repeating observations.
DV12 diagnostics materialize on observation, retaining only two detached
applications. Accelerator routing validates finite values once per complete
forward. See the [many-site synchronization measurement](e22_routed_readbacks.md)
for the validation timing, pending-memory bound and checkpoint compatibility.
Routed DV12 uses represented mass and active support, collapsing exact latent
duplicates before estimating bandwidth. Its checkpoint configuration records
`routed_geometry="mass_atoms_v1"`; former raw-row checkpoints require the prior
release for exact recovery. See the [noisy-game qualification guide](e22_routed_game.md)
for the migration boundary, pooled-token KA2 units, private diagnostic replay
and matched longer-training results. Clean feature gains alone do not establish
persistent support or gradient-conditioning repair.
Frozen module parameters and buffers may retain BF16 or other precision;
trainable floating tensors share the table's dtype. Frozen weights are copied
exactly into the serving averages.
Routed evidence refreshes independently of birth/death. With
`row_evidence_gate=True, particle_birth_death=False`, deletion probes update
support and persistence diagnostics without making structural proposals.
With both controls disabled, a routed forward can use a frozen bank and
`begin_step(real)` does not require fitting or guard observations.

## Initialization

`particlegan.init` is optional, explicit tooling in the style of
`torch.nn.init`. Nothing else in the package writes network weights: the
recipe factories, `make_prior` and `GANTrainer` use modules exactly as you
pass them. The examples initialize like this:

```python
from particlegan import get_recipe, init

recipe = get_recipe()
init.deterministic_orthogonal_(G, seed=0)                    # in place; returns G
init.deterministic_orthogonal_(D, seed=1)
init.deterministic_orthogonal_(E, seed=2)                    # an encoder, if any
prior = init.deterministic_orthogonal_(recipe.make_prior())  # R2 particle table
opt_g, opt_d = recipe.make_optimizers(G, D, prior, encoder=E, ema_critic=copy.deepcopy(D))
```

Call it on fresh networks, **before** loading trained weights, taking EMA
copies, or building optimizers; it does not touch optimizer state or copies
made earlier. Importing ParticleGAN changes nothing in PyTorch.

### `deterministic_orthogonal_(module, *, seed=0, strict=True)`

Rewrites the trainable parameters of `module` and its submodules in place and
returns `module`. Each layer class declares, per parameter, the distribution
its PyTorch constructor draws from; the replacement keeps that distribution's
scale but not its randomness:

| Parameter | Value |
| --- | --- |
| Matrix or kernel (2+ dims), `Uniform`/`Normal` | Semi-orthogonal matrix over `[shape[0], prod(shape[1:])]`, scaled so the entry RMS equals the declared distribution's RMS; reshaped back |
| Vector or scalar, `Uniform`/`Normal` | Deterministic pattern with the declared mean and standard deviation |
| Particle table, `R2Normal(mean, std)` | Row i is the i-th R2 low-discrepancy point mapped through the normal quantile |

`seed` is a nonnegative integer that keys a hash, not an RNG seed: values
come from hashing `(seed, parameter index, shape)` in CPU float64, then are
cast to each parameter's dtype and device. No RNG state is read or consumed,
and results do not depend on the device. The same seed and architecture give
the same weights, so networks with matching shapes need different seeds (the
examples use G=0, D=1, E=2). R2 tables ignore the seed. Repeating the call
rewrites the same values. CPU QR can be slow for very large matrices.

> **Values depend on where a parameter sits in the module you pass.** The
> parameter index in the hash is its position in `module.named_parameters()`
> of that module, not a property of the parameter itself. So:
>
> - initializing a submodule on its own (`deterministic_orthogonal_(G.head)`)
>   gives different values than the same submodule gets when you initialize
>   the whole network (`deterministic_orthogonal_(G)`);
> - adding, removing or reordering parameters shifts the values of every
>   parameter after them.
>
> Initialize each whole network in one call, and treat an architecture change
> as a change of initial weights.

Left as they are:

- frozen parameters (`requires_grad=False`) and buffers, so a frozen
  pretrained backbone keeps its weights while its new trainable head is
  initialized;
- parameters declared `KEEP` (normalization scales and shifts, `PReLU`
  slopes, `MultiheadAttention.in_proj_bias`);
- declared zero vectors and constant or identity matrices, which constructors
  set on purpose (e.g. zero attention output biases);
- undeclared parameters when `strict=False`.

**Strict mode.** With `strict=True` (the default), any other trainable
parameter no declaration covers raises `ValueError` before anything is
written. The message lists each one as `'path' (OwningClass)`:

```text
ValueError: deterministic_orthogonal_ has no declaration for 2 trainable parameter(s):
'1.patterns' (Hopfield), '1.beta' (Hopfield). Declare them with
particlegan.init.register(<layer class>, {name: Uniform/Normal/R2Normal/KEEP}),
or pass strict=False to leave them as-is.
```

Declare the layer with `register`, or pass `strict=False` to leave undeclared
parameters on their constructor's init. Lazy layers must be materialized
first, otherwise `ValueError`.

### Initializing priors

For comparisons with changing architectures,
`init.deterministic_orthogonal_(module, parameter_seeds={name: seed, ...})`
accepts one explicit seed for every nonempty trainable named parameter. This
mode removes positional-index coupling: adding an unrelated parameter does not
shift the values of shared components. Shapes and declared distributions still
matter; different architectures are not asserted to have identical weights.
The default call without `parameter_seeds` retains the existing whole-network
initialization exactly. `prior_capabilities(prior)` and
`prior_mechanisms(prior, latent_damping_max_rate=..., prior_beta1=...)` expose
the supplied prior's sampling and A2 support as JSON-compatible records.

`ParticlePrior` declares its table `z` as `R2Normal(0, init_std)`, so

```python
prior = init.deterministic_orthogonal_(recipe.make_prior())
```

gives a learnable table the R2 cloud at the prior's `init_std`. For a
`MoGParticlePrior` whose spacing was calibrated (`d0 > 0`, as
`recipe.make_prior()` does for MoG recipes, even at `sigma_rel=0`), the call
recalibrates `sigma` and `d0` on the new centers; an explicitly given `sigma`
is kept. A
`learnable=False` table is a buffer and is not changed. `GaussianPrior` has
no parameters. A learned `DrawSource` table (DDGAN) is declared
`R2Normal(0, 1)`.

### `initialize_(module, *, method, parameter_generators=None, distributions=None, gain=1.0, strict=True)`

An explicit, in-place initializer in `particlegan.init`; returns the same module.
It is independent of `Recipe` and `GANTrainer`, and does not change
`deterministic_orthogonal_` or its historical values.

| Method | Applied operation |
| --- | --- |
| `identity_linear_v1` | Identity weight and zero bias for a square `nn.Linear` |
| `xavier_uniform_zero_bias_v1` | Public PyTorch Xavier-uniform weights, with positive `gain`, and zero biases; trainable parameters must belong to supported Linear weights/biases |
| `sample_distributions_v1` | Literal PyTorch uniform/normal draws from registered `Uniform`/`Normal` descriptors; `KEEP` remains unchanged |

For sampled distributions, `distributions={full_parameter_name: descriptor}`
provides call-local overrides without changing the registry. An `R2Normal`
table needs an explicit literal distribution override. Uniform endpoints and
normal parameters must be finite, with `low < high` and `std > 0`; these methods
do not interpret a zero-width distribution as a constant. Use identity/zero
operations or `KEEP` for constants. Xavier `gain` must be finite and positive;
other methods require its default value.

`parameter_generators` maps **exactly** the randomly drawn parameter names to
distinct CPU `torch.Generator` objects. Exclude biases zeroed by Xavier, identity
parameters, `KEEP`, frozen and empty parameters. Callers own seed derivation and
checkpointing; the initializer neither creates hidden random streams nor reads
global randomness. Each draw uses CPU scratch in the parameter's dtype, then
copies to its destination device. A matching named stream therefore does not
depend on the physical GPU index or an unrelated parameter's draws.
Passing `torch.default_generator` is rejected; each stream must be owned by the
caller independently of the global RNG.

```python
from particlegan import MoGParticlePrior, init

prior = MoGParticlePrior(32, 2, sigma=0.025, standardize=False)
# The caller supplies its existing named initialization stream for prior/z.
init.initialize_(prior, method="sample_distributions_v1",
                 distributions={"z": init.Uniform(-5.0, 5.0)},
                 parameter_generators={"z": prior_location_generator})
```

Input declarations, shapes, generator keys/devices and unsupported aliases are
validated before drawing. Operations and registered finalizers then run on a
staged CPU module with cloned streams. The result must preserve all unselected
and frozen parameters, buffers, parameter metadata and module parameter/buffer
names before selected values and RNG states are committed. Validation or
finalizer failure leaves caller tensors and supplied stream states unchanged.
Finalizers that consume the global CPU RNG are rejected. Parameter aliases,
shared parameter storage, and parameter/buffer storage sharing are unsupported.
Trainable parameters must be contiguous. This conservatively rejects transposed
views as well as internally overlapping views, preventing an in-place commit
failure after another parameter has already changed.
`strict=False` permits undeclared parameters only in the sampling method; it
does not relax malformed distributions, unsupported methods or stream checks.

Registered trainable layout fixes still apply, such as the batch-distance
critic's zero batch-feature readout. Explicit-sigma MoG initialization preserves
its width, masses and standardization policy. A spacing-calibrated MoG whose
finalizer would change `sigma`/`d0` is rejected by this API; calibrating noise is
a separate explicit operation. Frozen prior tables remain untouched.

`Uniform`/`Normal` descriptors have different applications in the two public
functions: `initialize_(..., method="sample_distributions_v1")` draws the
literal distribution, while `deterministic_orthogonal_` uses its moments to
construct QR matrices/patterns. Neither method implies historical fixture parity.

### `register(cls, declarations=None, *, finalize=None)`

Declares distributions for parameters that `cls` itself owns (not those of its
submodules). `deterministic_orthogonal_` uses their moments, and the explicit
`sample_distributions_v1` method draws supported distributions literally.
`declarations` maps each parameter name to a spec:

| Spec | Meaning |
| --- | --- |
| `init.Uniform(low, high)` | The constructor draws U(low, high) |
| `init.Normal(mean=0, std=1)` | The constructor draws N(mean, std²) |
| `init.R2Normal(mean=0, std=1)` | 2-D particle table: R2 points through the N(mean, std²) quantile |
| `init.KEEP` | The constructor's value is deliberate; never changed |

Pass a callable `module -> mapping` when a spec depends on the instance, such
as its fan-in. Declarations merge along the class MRO, base classes first, so a
subclass declares only the parameters it adds or overrides; a subclass of
`nn.Linear` inherits the `weight`/`bias` declarations. Declaring a name the
module lacks, or a spec of another type, raises when `declarations` or
`deterministic_orthogonal_` reaches the module. `finalize(module)` runs under `no_grad` after the call has
written any parameter inside the module, for layout fix-ups (the built-in
`nn.Embedding` rezeroes its `padding_idx` row). Registering a class again
replaces its entry.

A Hopfield-style memory layer with raw stored patterns and a learnable inverse
temperature:

```python
import math
import torch
from torch import nn
from particlegan import init

class Hopfield(nn.Module):
    def __init__(self, dim, count, beta=8.0):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(count, dim) / math.sqrt(dim))
        self.beta = nn.Parameter(torch.tensor(beta))  # inverse temperature

    def forward(self, x):
        weights = torch.softmax(self.beta * x @ self.patterns.T, dim=-1)
        return weights @ self.patterns

init.register(Hopfield, lambda layer: {
    "patterns": init.Normal(0.0, 1 / math.sqrt(layer.patterns.shape[1])),
    "beta": init.KEEP,     # a chosen temperature, not a random draw
})

net = nn.Sequential(nn.Linear(16, 32), Hopfield(32, 64), nn.Linear(32, 1))
init.deterministic_orthogonal_(net, seed=1)
```

`patterns` becomes a 64×32 semi-orthogonal matrix at RMS `1/sqrt(32)`;
without `KEEP`, the scalar `beta` would be replaced by a pattern value.

### `declarations(module)`

Returns `{path: spec}` for every trainable parameter, with `None` for an
undeclared one, and changes nothing. Use it to check a custom network before
initializing it:

```python
>>> init.declarations(net)
{'0.weight': Uniform(low=-0.25, high=0.25), '0.bias': Uniform(low=-0.25, high=0.25),
 '1.patterns': Normal(mean=0.0, std=0.1767...), '1.beta': KEEP,
 '2.weight': Uniform(low=-0.1767..., high=0.1767...), '2.bias': Uniform(...)}
```

**Recommended downstream check.** A project with custom networks can keep a
test that fails as soon as a new layer or parameter lacks a declaration,
instead of finding out when `deterministic_orthogonal_` raises in a run:

```python
from particlegan import init

def test_networks_fully_declared():
    for net in (MyGenerator(), MyCritic()):
        undeclared = [p for p, spec in init.declarations(net).items() if spec is None]
        assert not undeclared, undeclared
```

### Built-in declarations

| Layer | Declaration |
| --- | --- |
| `nn.Linear`, `nn.Conv1d/2d/3d`, `nn.ConvTranspose1d/2d/3d` | `weight`, `bias`: `Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`, fan-in = `prod(weight.shape[1:])` |
| `nn.Embedding` | `weight`: `Normal(0, 1)`; the `padding_idx` row is zeroed afterwards |
| `nn.MultiheadAttention` | packed `in_proj_weight` or `q/k/v_proj_weight`: Xavier-uniform bound; `bias_k/bias_v`: Xavier-normal std; `in_proj_bias`: `KEEP`; `out_proj` is an `nn.Linear` |
| `nn.LayerNorm`, `nn.GroupNorm`, BatchNorm, InstanceNorm, `nn.RMSNorm` | `weight`, `bias`: `KEEP` |
| `nn.PReLU` | `weight`: `KEEP` |
| `ParticlePrior` (and `MoGParticlePrior`) | `z`: `R2Normal(0, init_std)`; MoG recalibrates calibrated spacing |
| `particlegan.diffusion.DrawSource` | `table`: `R2Normal(0, 1)` |
| `BatchDistanceDiscriminator` | its layers use the rules above; the head's batch-distance coefficients are then zeroed, leaving the per-point score path active |

Packed attention QKV uses one QR over its stored tensor. Parametrized weights,
fused layouts and other custom parameters need their own declaration.

### Migrating from `initialize_` and `Recipe.initialization`

Earlier development builds had `Recipe.initialization="batch_feature_zero"`,
which `make_optimizers`/`GANTrainer` applied to fresh G/D/E weights (keys 0, 1,
2) and `make_prior` to learnable tables, plus `particlegan.initialize_`. Both
are removed; `get_recipe(initialization=...)` is now a `TypeError`. The new
`particlegan.init.initialize_(..., method=...)` is a separate explicit API;
it does not restore the old top-level `particlegan.initialize_(..., key=...)`
or implicit optimizer/trainer initialization.

| Before | Now |
| --- | --- |
| `initialize_(module, key=k)` | `init.deterministic_orthogonal_(module, seed=k)` (same values) |
| implicit init in `make_optimizers` / `GANTrainer` | call `deterministic_orthogonal_` on G (0), D (1), E (2) first, then build the EMA critic |
| implicit R2 table in `recipe.make_prior()` | `init.deterministic_orthogonal_(recipe.make_prior())`, then pass `prior=` to `GANTrainer` |
| `get_recipe(initialization=None)` | `get_recipe()`; weights are already left alone |

The old path skipped undeclared custom parameters silently; strict mode now
reports them. `GANTrainer.load_state_dict` still accepts checkpoints whose saved
recipe records `initialization` (`None` or `"batch_feature_zero"`); saved
weights replace construction-time values as before.

The [math guide](initialization.md) describes the QR identities, targeted
batch-feature correction, convolution storage, attention, and LoRA. The
[research report](../reports/toy100/batch-feature-init/README.md) holds the
22/22 frozen-suite evidence for the construction this API reproduces.

The historical `benchmarks.toy100 --init NAME` runner records its installed
construction hook as `deterministic_init` in the run manifest and resolved
config. This field is replay metadata: passing it to `train(config, output)`
does not install an initializer. Direct benchmark callers must install the
declared hook before model construction. Affine receipt audits rebuild that
initializer on the recorded device in a separate interpreter and check the
actual prior and generator tensors; named initializer source files are pinned
in the run archive.

## GANTrainer

`GANTrainer(recipe, G, D, *, prior=None, seed=0, latent_generator=None,
penalty_generator=None, noise_generator=None, input_noise_generator=None,
prior_noise_generator=None, eval_generator=None, model_generator=None,
require_latent_damping=None, max_steps=None, optimizer_options=None,
penalty_options=None, serial_backward=True)` is an
explicitly imported helper, separate from `Recipe`. Move networks to the same
device and floating dtype first. When `prior=` is omitted, the helper constructs
the prior with `recipe.make_prior()`, a plain randomly drawn table. `seed` controls owned sampling streams. The helper
never changes the weights it receives; for deterministic starting weights,
call [`init.deterministic_orthogonal_`](#initialization) on G, D and a
recipe-made prior first and pass `prior=`, as in the minimal loop above.

ParticleGAN disables autograd's multithreaded backward scheduling on import.
`GANTrainer.step` enforces this for the entire update and restores the caller's
setting afterward. CPU operation thread pools and CUDA parallelism are unchanged.
`serial_backward=True` remains accepted for compatibility; False is rejected.
Checkpoints record True. Unmarked or False historical checkpoints must be
resumed using their pinned original source because changing scheduling can
change gradient accumulation rounding. For caller-owned component loops in a
new thread or an explicitly enabled external context, wrap the whole update in
`with particlegan.serial_autograd():`, including its forward passes.

The helper supports scalar, unconditional GANs with `ParticlePrior` or
`MoGParticlePrior`, matching the recipe's `prior_kind` and `standardize` policy.
Encoders, conditional GANs and DDGAN use the component API. A step performs one
D update, then one G/prior update with fresh latent samples. During the G phase,
D is evaluated with frozen parameters; each original gradient flag is restored.
Small particle tables (at most 1,024 rows) are regularized in full; larger tables
use unique sampled rows. There is no particle L2 term.

MoG training and sampling keep the declared Gaussian noise, including under
`eval()` and EMA sampling. Pass an explicit prior from
`recipe.make_prior(sigma=...)` to avoid spacing calibration. A2 supports learned
row-local locations (`standardize=False`). Standardized reads couple all rows;
disable A2 explicitly with `latent_damping_max_rate=0` for that formulation.
The trainer requires configured A2 by default for a learnable prior. Its
`prior_mechanisms` receipt reports eligibility, activation, and the reason.

Optional generators isolate component indices (`latent_generator`), MoG draws
(`prior_noise_generator`), critic input noise (`input_noise_generator`), output
noise (`noise_generator`), evaluation (`eval_generator`), and stochastic layers
such as dropout (`model_generator`). Direct-draw training streams may share a
generator when a nonadaptive legacy host requires one draw sequence. Adaptive
policies keep their four public policy streams distinct. Evaluation and model
streams must each remain separate from the other streams. When `model_generator`
is supplied, no training stream may use the process-default generator, because
its state is temporarily replaced for stochastic layers. Forge always supplies
distinct named bindings. Recreate the same stream bindings when restoring a
checkpoint, since legacy checkpoints do not encode alias topology. Conflicting
saved states for a shared stream are rejected before restoration. The old
particle-cloud defaults keep their original streams and schema-4 checkpoints;
MoG and explicitly supplied additional streams are checkpointed too. Data RNG
remains caller-owned. Evaluation cannot borrow a training stream.

- `step(real, generator_real=None, collect_stats=False)` returns detached scalar
  tensors `loss_d`, `loss_g`, `loss_gan`, `prior_regularization`, `penalty`, and
  integer `step`. The reported prior term is unweighted; `loss_g` includes its
  recipe weight. `generator_real` can supply a fresh tensor or zero-argument
  callback for RP/RA; otherwise the real batch is reused. RP requires equal
  batch sizes. `collect_stats=True` also returns penalty diagnostics.
- `sample(n, ema=False, generator=None, output_noise=False, fixed_first_n=False, offset=0)`
  defaults to fast weights for ordinary recipes and omits training output noise.
  E22 and Atlas retain DV12 latent perturbation and their state-selected served
  fast/averaged weights; `ema=False` does not force the fast iterate for those policies.
  `output_noise=True`
  adds the current training output noise using the sampling stream. Its separate
  RNG and temporary evaluation mode preserve training randomness and module modes.
  `fixed_first_n=True` enumerates component indices from `offset`; the requested
  block must fit within the table bounds. Learned MoG kernel noise remains part of the prior law; a
  zero-width cloud without output noise consumes no evaluation RNG in this mode.
  EMA averages G/prior parameters and copies their buffers, including integer
  counters. EMA never determines a live leaderboard pass.
- `state_dict()` includes G, D, prior, EMA, optimizers, initial learning rates,
  update count and RNG states. `load_state_dict(state)` restores them, including
  global PyTorch RNG. Recreate the same recipe, architecture, options, dtype
  and device, with the same parameter freezing, before loading. Save your data-loader position or separate data RNG
  alongside it. Loading on CPU first works for a compatible CUDA trainer:
  `trainer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))`.
  Save and restore only between completed updates. Both methods raise
  `RuntimeError` during an active trainer step or policy hook sequence, including
  inside `generator_real` callbacks, before changing live state.
  When `generator_gradient_response` is enabled, the checkpoint also includes
  its float64 gradient history, normalization weights, gain diagnostics and
  applied-step count. Older checkpoints that enabled this response but omitted
  its history cannot resume exactly and raise `ValueError`; use their original
  source or start a new run. Checkpoints with response `none` retain their
  existing schema and compatibility.

For scheduled recipes, `total_steps` fixes every schedule horizon. By default it is also
the execution bound. After restoring a checkpoint with its original bound, call
`trainer.extend_execution(new_max_steps)` to increase only the execution
allowance while keeping model, optimizer, prior, EMA, and RNG state intact.
The new allowance must exceed the current one. Set `max_steps` explicitly to allow bounded continuation
past that horizon; this preserves the original prefix and leaves schedules at
their terminal values. It does not stretch or restart a schedule. Save and
resume with the same explicit bound (included in the checkpoint when different
from `total_steps`). Further steps after the execution bound raise an error. A failed user
callback can occur after D has updated, so restore a checkpoint before retrying
that interrupted update. AMP, distributed training and custom update ratios
require a caller-owned loop.
Schedule-free E22 and Atlas retain `total_steps=None`; pass `max_steps` to bound
their execution without inserting a schedule. Their policy hooks use that
external allowance without changing the formulation.

`get_recipe()` constructs the selected **optimized BISPF** research recipe: non-saturating
loss, the Born-Infeld chord penalty, DualNorm, direction blending and constant
G/E, D and prior rates `.012`, `.018` and `.03`. `GANTrainer` binds sampled
rows and the protected adversarial objective automatically. Additive training
noise, EMA serving and KA2 interventions are disabled. SBNP prior regularization
has weight `.01`.
See [the selected settings and evidence](#recipes-and-defaults).

`get_recipe("ka2")` retains **KA2** ([details](ka2.md)): Rp logistic, the KA2
critic penalty (coefficient 1, κ 1, EMA-critic anchor gated by the critic's Adam
moment surprise), critic spike guard
(ratio 5 after 200 steps), A2 latent-row damping, Adam (0,.999), G/D LR .00425
and particle LR .0085. G/D rates hold for 60% of a 1,600-update horizon, then
cosine to 1% (`network_lr_horizon_cap`, `network_lr_floor`); particle rates hold
for 60% of the budget, then cosine toward 5%. The critic sees annealed input
noise and the generator output carries warmed-up noise (also in `sample`). There
is no particle spread or L2 term. Live sampling is the default; EMA is explicit.

`GANTrainer` builds everything through the recipe: `trainer.opt_g, trainer.opt_d
= recipe.make_optimizers(G, D, prior, ema_critic=...)` (the trainer allocates
`trainer.ema_D`, a frozen deep copy) and `trainer.penalty =
recipe.make_critic_penalty(trainer.opt_d)`. Checkpoints use schema 4 (the KA2
controller and EMA critic are inside the optimizer states, plus a noise
stream); schema 1–3 checkpoints come from older formulations and raise
`ValueError` (resume them with the release that wrote them). Caller-owned loops use the same objects with an ordinary loop:
`penalty(D, real, fake)` in the critic loss, then `opt_d.step()` and
`opt_g.step()` as usual. See [regularization factories](#regularization-factories). `learning_rate_scales(step, recipe)` returns the
`(network, prior)` LR multipliers.

### Optional vector discriminators

`BatchDistanceDiscriminator(in_dim=2, hidden_dim=96, n_hidden=3,
scales=(.1,.25,.5,1.), beta=6., eps=1e-5)` accepts nonempty flat
`[batch, in_dim]` inputs and returns one score per sample (**19,013 parameters**
at the 2D defaults): per-example hidden feature centering, Softplus β6, and
differentiable kernel-weighted neighbor distances appended to the final head.
Self-pairs are excluded.

Scores depend on other samples in the current input batch, including during
G updates and gradient-cap differentiation. Real/fake calls compute separate
features; there are no running statistics. Cost is quadratic in batch size,
and scales are in input-coordinate units. The measured witness uses 2D inputs
and batch size 128. You supply D explicitly.

`LinearSkipDiscriminator(in_dim=2, hidden_dim=96, n_hidden=2, fourier=2, beta=5.)`
is a smooth Fourier MLP plus a zero-initialized raw linear branch, 10,467
parameters. It supports the double backward the critic penalty needs.
Both classes can be passed to `GANTrainer` or used in a custom loop.

## A minimal DDGAN + UCD loop

This standalone example learns two conditional 2D distributions. G predicts
clean data; DDGAN constructs a reverse transition; UCD selects the requested
class score without feeding the class label into D's network. Replace the
synthetic `real` and `labels` with batches from your pipeline.

```python
import copy
import torch
from torch import nn
from torch.nn import functional as F
from particlegan import DDGAN, UCD, get_recipe, init, scale_learning_rates, ucd_loss
from particlegan.optim.constraint_geometry import constraint_geometry_backward

device = torch.device("cpu")
recipe = get_recipe(model="ddgan", conditioning="ucd", num_classes=2)  # Add total_steps=5 for a smoke check.
process = DDGAN(recipe.alpha_bar).to(device)

class Generator(nn.Module):
    def __init__(self, z_dim, classes, steps):
        super().__init__()
        self.classes, self.steps = classes, steps
        self.net = nn.Sequential(nn.Linear(z_dim + classes + 3, 64),
                                 nn.LeakyReLU(0.2), nn.Linear(64, 2))

    def forward(self, z, labels, *, xt, t):
        c = F.one_hot(labels, self.classes).to(z)
        time = t[:, None].to(z) / self.steps
        return self.net(torch.cat((z, c, xt, time), dim=1))

class LogitNetwork(nn.Module):
    def __init__(self, classes, steps):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(nn.Linear(5, 64), nn.LeakyReLU(0.2),
                                 nn.Linear(64, classes))

    def forward(self, x, *, xt, t):
        return self.net(torch.cat((x, xt, t[:, None].to(x) / self.steps), dim=1))

G = Generator(recipe.z_dim, recipe.num_classes, process.steps).to(device)
D = UCD(LogitNetwork(recipe.num_classes, process.steps), recipe.num_classes).to(device)
init.deterministic_orthogonal_(G, seed=0)   # optional deterministic start
init.deterministic_orthogonal_(D, seed=1)
prior = init.deterministic_orthogonal_(recipe.make_prior()).to(device)
gan = recipe.make_loss()
# Optimizers follow the shared trainer preset, including module update semantics.
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
spread = recipe.make_prior_regularizer(prior=prior)
base_lrs = [[group["lr"] for group in opt.param_groups] for opt in (opt_g, opt_d)]
penalty = recipe.make_critic_penalty(opt_d)
ema_g = copy.deepcopy(G).eval().requires_grad_(False)
ema_prior = copy.deepcopy(prior).eval().requires_grad_(False)

for step in range(recipe.total_steps):
    # G/D follow the network schedule, the prior its own.
    scale_learning_rates(step, recipe, (opt_g, opt_d), base_lrs, prior)

    labels = torch.randint(recipe.num_classes, (recipe.batch_size,), device=device)
    real = 0.2 * torch.randn(len(labels), 2, device=device) + (2 * labels[:, None] - 1)
    t = torch.randint(1, process.steps + 1, (len(real),), device=device)
    x_prev, xt = process.forward_pair(real, t)
    z, indices = prior.sample(len(real))
    clean = G(z, labels, xt=xt, t=t)
    fake = process.reverse(clean, xt, t, torch.randn_like(xt))

    opt_d.zero_grad(set_to_none=True)
    real_score, real_logits = D(x_prev, labels, xt=xt, t=t)
    fake_score, fake_logits = D(fake.detach(), labels, xt=xt, t=t)
    d_loss = gan.d_loss(real_score, fake_score)
    d_loss += ucd_loss(real_logits, fake_logits, labels, weight=recipe.ucd_weight)
    d_loss = d_loss + penalty(D, x_prev, fake.detach(), labels, xt=xt, t=t)
    d_loss.backward()
    opt_d.step()

    D.requires_grad_(False)
    opt_g.zero_grad(set_to_none=True)
    fake_score = D(fake, labels, xt=xt, t=t)[0]
    real_score = D(x_prev, labels, xt=xt, t=t)[0].detach()
    adversarial_g = gan.g_loss(fake_score, real_score)
    g_loss = adversarial_g + spread(prior.z[indices.unique()])
    if hasattr(opt_g, "set_sampled_rows"):
        opt_g.set_sampled_rows(prior.z, indices)
    constraint_geometry_backward(g_loss, opt_g, (adversarial_g,))
    opt_g.step()
    D.requires_grad_(True)

    with torch.no_grad():
        for average, current in ((ema_g, G), (ema_prior, prior)):
            for target, source in zip(average.parameters(), current.parameters()):
                target.lerp_(source, 1 - recipe.ema_decay)
    if step % 100 == 0 or step + 1 == recipe.total_steps:
        print(f"step={step + 1} d={d_loss.item():.4f} g={g_loss.item():.4f}", flush=True)

# Conditional inference: fresh latent particles and Gaussian noise at each step.
with torch.inference_mode():
    labels = torch.arange(64, device=device) % recipe.num_classes
    samples = torch.randn(len(labels), 2, device=device)
    for step in range(process.steps, 0, -1):
        t = torch.full_like(labels, step)
        z, _ = ema_prior.sample(len(labels))
        clean = ema_g(z, labels, xt=samples, t=t)
        samples = process.reverse(clean, samples, t, torch.randn_like(samples))
```

The D update combines adversarial loss, class cross-entropy, and the candidate
gradient penalty. The G/prior update combines adversarial loss and particle
regularization. One random transition per example is used during training;
inference walks through every reverse step. Class-only UCD is the default;
[joint time/class heads](#ucd) are an independent option.

## Reference index

| Component | Purpose |
| --- | --- |
| [Priors](#priors) | Learnable particles, fixed-sigma Gaussian mixtures, or fresh Gaussian samples |
| [Losses](#losses-and-regularizers) | Adversarial objectives and additive regularizers |
| [DDGAN](#ddgan) | Forward corruption and reverse transitions |
| [UCD](#ucd) | Class-score selection and class supervision |
| [Recipes](#recipes-and-defaults) | Inspectable defaults and optional factories |
| [GANTrainer](#gantrainer) | Optional unconditional GAN updates, sampling and checkpoints |
| [TOML](#toml-configuration) | Pass loaded dictionaries to constructors |
| [Other pipelines](#loss-augmentation-and-teacherstudent-pipelines) | Compose with existing objectives |
| [Inference](#inference-and-checkpoints) | Generate from saved G and prior states |

All names below are exported from `particlegan`. Modules use ordinary
`.to(device, dtype)`, `.parameters()`, and `.state_dict()` behavior. Move models
and priors before constructing optimizers. Stateless loss helpers need no device
setup. These primitives never step an optimizer. The optional `GANTrainer`
manages updates and restores RNG state when loading checkpoints.

## Priors

### `ParticlePrior`

```python
ParticlePrior(num_particles=20_000, z_dim=4, init_std=1.0,
              device=None, dtype=None, learnable=True, generator=None)
```

An `nn.Module` with table `prior.z` of shape `[num_particles, z_dim]`, initialized
from a zero-mean Gaussian with standard deviation `init_std`. By default the
table is a parameter. With `learnable=False`, it is a fixed buffer.
`recipe.make_prior()` keeps this random draw;
[`init.deterministic_orthogonal_(prior)`](#initializing-priors) replaces a learnable table
with a deterministic R2 cloud at `init_std`.

| Method / attribute | Result |
| --- | --- |
| `sample(batch_size, generator=None)` | `(z, indices)`: `[B, z_dim]` codes and `[B]` long indices, sampled uniformly with replacement |
| `sample(..., fixed_first_n=True, offset=0)` | Consecutive rows starting at `offset`; requires a block within the table |
| `sample_indices(batch_size, generator=None)` | Only the random indices, on the table's device |
| `prior(indices)` | Indexed codes with gradients to the selected rows |
| `num_particles`, `z_dim` | Dimensions of the table |

Sampling does not detach latent codes. Include the prior parameters in your
optimizer to learn them. Use `prior(indices)` through your distributed wrapper's
forward path when applicable. An explicit `torch.Generator` controls initialization
or sampling without consuming the global RNG; use a generator for the same device.

### `MoGParticlePrior`

```python
MoGParticlePrior(num_particles=400, z_dim=4, init_std=1.0,
                 device=None, dtype=None, learnable=True, generator=None,
                 *, sigma, standardize=True)
```

An equal-weight mixture: choose component `i` uniformly, then draw
`z = means()[i] + sigma * eps`, with standard-normal epsilon. The raw component
centers are the parameter `prior.z`. Sigma is a **required keyword argument**:
a finite, nonnegative real scalar stored as one shared isotropic buffer, fixed
during training. Construction draws the centers once and performs no calibration
or nearest-neighbor search. `learnable=False` freezes the table as a buffer.
Standardized reads require at least two components; raw reads allow one.
Coincident centers and `init_std=0` are valid with an explicit sigma.

With `standardize=True`, each read centers and divides the table by its
per-dimension sample standard deviation plus `1e-6`. This is differentiable;
all rows can receive gradients. With `standardize=False`, means are the raw
table, as in `ParticlePrior`. Learning and EMA updates do not recalibrate sigma.

| Method / attribute | Result |
| --- | --- |
| `sample(batch_size, generator=None, *, fixed_first_n=False, offset=0, eps=None, noise_generator=None)` | Noisy codes and selected component indices |
| `prior(indices, generator=None, *, eps=None)` | Noisy draws for supplied indices; use this forward path through DDP |
| `means()` | Differentiable read-space centers, with no noise |
| `z` | Raw learned table; the input to particle regularization |
| `sigma` | Fixed scalar buffer, moving with `.to(...)` |
| `d0`, `sigma_rel` | Legacy calibration metadata; zero for explicitly supplied sigma |
| `set_sigma(sigma)` | Explicitly replace the fixed scale and update zero-noise RNG handling |

`fixed_first_n=True` fixes component indices, **not epsilon**. For a stable scatter,
save a fixed epsilon tensor too:

```python
from particlegan import MoGParticlePrior

prior = MoGParticlePrior(num_particles=65536, z_dim=128, sigma=0.212616428732872)
eps = torch.randn(64, prior.z_dim, device=prior.z.device, dtype=prior.z.dtype)
z, indices = prior.sample(64, fixed_first_n=True, eps=eps)
# Reuse eps at every snapshot; use prior.means()[indices] only for a centers-only audit.
```

Explicit epsilon must match the sampled codes' shape, device and dtype. An explicit
generator controls both component selection and Gaussian draws without touching
global RNG. Supply `noise_generator` to isolate Gaussian draws from component
selection. `sigma=0, standardize=False` preserves `ParticlePrior` outputs and
RNG consumption; zero sigma never draws noise. `eval()` keeps Gaussian noise on.

For DDP, sample indices from the unwrapped prior, then call the wrapped module:
`z = wrapped_prior(indices, generator=rng)`. Use `prior.z` for VICReg rather than
`prior(indices)` or sampled codes. The one-shot MoG benchmark regularizes the
full raw table at N ≤ 1024, otherwise `prior.z[indices.unique()]`. The denoising
trainer and loops above regularize sampled unique raw rows for either prior.

For EMA, deepcopy the prior and average its learned `z`; the fixed buffers retain
their fixed values. Standardization is computed from the EMA table itself.
State dicts include `z`, `sigma`, `d0`, and `_extra_state` containing `sigma_rel`
and `standardize`. Reconstruct with matching dimensions and an explicit placeholder `sigma=0`, then
`load_state_dict`:
read settings and noise are restored even if constructor defaults differ.
Legacy experimental checkpoints containing only z/sigma/d0 are accepted; supply
their original `standardize` setting when constructing the prior.

Optional spacing calibration is a standalone helper:

```python
from particlegan import MoGParticlePrior, calibrate_mog_sigma

prior = MoGParticlePrior(num_particles=400, z_dim=4, sigma=0)
sigma, d0 = calibrate_mog_sigma(prior.means(), sigma_rel=1/40)
prior.set_sigma(sigma)
prior.d0.copy_(d0)        # Optional historical metric/checkpoint metadata.
prior.sigma_rel = 1/40
```

This calibrates the **already initialized centers without redrawing them**.
`recipe.make_prior()` with a MoG recipe (or `prior_kind="mog", sigma_rel=...`)
performs these steps on the constructor's random centers.
`init.deterministic_orthogonal_(prior)` later moves the centers to an R2 cloud
and repeats the calibration for a calibrated prior.
The helper accepts supplied read-space centers; it does not standardize, mutate
centers, or consume RNG. It returns detached scalar tensors `(sigma, d0)` on the
centers' device and dtype. The exact median averages the two middle nearest-neighbor
distances for even component counts. As before, `d0` is rounded to the centers'
dtype before multiplication by `sigma_rel`. Calibration requires at least two
finite centers and a positive median spacing, even when `sigma_rel=0`.

**Calibration can be very expensive**, especially for 65,536 centers in 128
dimensions. It copies centers to CPU float64 and uses SciPy's exact CPU tree
when installed (`pip install 'particlegan[mog]'`), or memory-bounded, quadratic
Torch distances otherwise. Trees can also be slow in high dimensions. Choose
an explicit sigma to avoid this work. Sampling needs neither SciPy nor NumPy.

Migration from 0.5: replace constructor `sigma_rel=...` with `sigma=...`; the
`calibrate()` method is replaced by `calibrate_mog_sigma`. Historical MoG recipes
retain spacing calibration explicitly; `recipe.make_prior(sigma=...)` skips it.
For HyperGAN, pass `sigma=fixed_sigma` directly and remove the subsequent buffer
overwrite. Checkpoint loading must restore its saved sigma, even when it differs
from the constructor's value. Do not call the helper when loading a checkpoint.

The default experiment is [configs/mog/default.toml](../configs/mog/default.toml),
run via `python -u experiments/train_100gaussians.py --config configs/mog/default.toml`.
It retains the benchmark's networks, Fourier discriminator and full metric suite.
Use `get_recipe(prior_kind="mog", sigma_rel=.025)`; a recipe alone does not
reproduce benchmark quality with arbitrary networks or data.

### `GaussianPrior`

```python
GaussianPrior(z_dim=4, init_std=1.0, device=None, dtype=None)
```

An `nn.Module` that draws fresh Gaussian codes. `sample(batch_size,
generator=None)` returns `(z, None)`. Calling the module as
`prior(batch_size, generator=None)` returns only `z`. It has no trainable
parameters or particle table, so omit particle regularization. Its empty buffer
tracks device and dtype.

## Losses and regularizers

### `GANLoss`

```python
gan = recipe.make_loss()  # GANLoss(recipe.loss, labels=recipe.loss_labels)
recipe = get_recipe("bcap", loss="hinge")
gan = recipe.make_loss()
```

`d_loss(real_logits, fake_logits)` and `g_loss(fake_logits, real_logits=None)`
return scalar tensors to minimize. Both preserve their input gradient paths;
the caller decides which scores to detach. Inputs are critic scores, without
a sigmoid. Use matching shapes such as `[B]` or `[B, 1]` for paired scores.

`Recipe.loss` selects one of these objectives. `r` and `f` are raw real/fake
scores, and `mean` averages each score tensor independently except for paired
`relativistic` differences.

| `loss` | D loss | G loss |
| --- | --- | --- |
| `relativistic` (default) | `mean(softplus(f-r))` | `mean(softplus(r-f))` |
| `non_saturating` | `mean(softplus(-r)) + mean(softplus(f))` | `mean(softplus(-f))` |
| `hinge` | `mean(relu(1-r)) + mean(relu(1+f))` | `-mean(f)` |
| `wasserstein` | `mean(f) - mean(r)` | `-mean(f)` |
| `least_squares` | `(mean((r-1)**2) + mean(f**2))/2` | `mean((f-1)**2)/2` |

Least squares accepts `loss_labels=(fake, real, generator)`, default `(0,1,1)`.
For `(a,b,c)`, D minimizes `(mean((r-b)**2)+mean((f-a)**2))/2` and G minimizes
`mean((f-c)**2)/2`. `joint_g_loss` also reverses the real-stream target to `a`.
Use `GANLoss("least_squares", labels=(-1,1,1))` or the corresponding Recipe
field; nondefault labels on other objectives are rejected.

Only `relativistic` requires real scores in `g_loss`, paired row by row. The
other losses ignore that optional argument. Recompute D's scores after updating
D; freeze its weights during the G step while retaining the gradient path
through the fake input. Unknown loss names raise `ValueError` when constructing
`Recipe` or `GANLoss`. Alternative objectives are explicit in `recipe.to_dict()`;
the historical default stays implicit to preserve older configuration packets.

These are the paired objective from
[the relativistic discriminator paper](https://arxiv.org/abs/1807.00734),
the non-saturating objective from
[the original GAN paper](https://arxiv.org/abs/1406.2661), the hinge objective from
[SAGAN](https://arxiv.org/abs/1805.08318), the linear critic objective from
[WGAN](https://arxiv.org/abs/1701.07875), and the real=1/fake=0/generator=1
parameterization in [LSGAN](https://arxiv.org/abs/1611.04076). Selecting
`wasserstein` changes the objective only; it does not add weight clipping,
WGAN-GP or a global Lipschitz constraint. Critic regularization is independent.

Joint [BiGAN](https://arxiv.org/abs/1605.09782) loops train an encoder through
real pairs `(x, E(x))` as well as a generator through fake pairs `(G(z), z)`.
Use `joint_g_loss(fake_logits, real_logits)` for that shared generator/encoder
step, with D frozen. It preserves gradient paths through both streams and
reverses both discriminator labels. The paired relativistic objective is
identical to `g_loss(fake, real)`. For other losses, it adds an encoder term to
the scalar G loss: `mean(softplus(real))` for `non_saturating`, `mean(real)` for
`hinge`/`wasserstein`, and `mean(real**2)/2` for `least_squares`. Real scores are
required. Ordinary scalar GANs continue to use `g_loss`, whose unpaired losses
depend only on fake scores. The five-word joint fixture uses `joint_g_loss`.

### Critic penalty

```python
penalty = recipe.make_critic_penalty(opt_d)   # opt_d from recipe.make_optimizers(...)
d_loss = gan.d_loss(D(real), D(fake.detach())) + penalty(D, real, fake.detach())
```

The penalty is a loss term: call it with the critic, reals and detached fakes
(plus any conditioning, forwarded to the critic and its EMA) and add the
result to the critic loss. It reads the state it needs from its critic
optimizer, so build it from the optimizer the recipe made for that critic and
pass `ema_critic=copy.deepcopy(D)` there. It penalizes the critic's input
gradient: R1 on reals plus a cap on fakes for its first 799 calls, then an even
blend with caps on both plus an EMA-critic gradient anchor gated by the critic's
Adam moment surprise ([how it works](ka2.md)). `reg_coeff`, `reg_kappa` and `reg_every` set its
strength, cap and lazy interval. It recomputes D on detached inputs and builds
gradients only for the critic's parameters.

`get_recipe("ka2", reg_arm="a_r1r2")` selects the fixed zero-centered squared L2
gradient penalty on real and fake inputs. `reg_arm="b_cap"` selects the fixed
one-sided L2 cap, `relu(norm(grad D) - reg_kappa) ** 2`, on both. Each uses
`reg_coeff / 2` times the sum of the two mean penalties, with autograd and the
same lazy interval. These arms reproduce the v0.7 kernels and do not evaluate
the K3P blend or critic EMA anchor. Optimizer, prior and noise settings remain
explicit recipe choices; selecting BCap alone does not restore the full v0.7
recipe. Explicit legacy arms select the K3P optimizer; `reg_arm="k3p"` selects
the earlier K3P penalty too. `reg_arm=None` follows `critic_formulation`, whose
restoration default is KA2. These choices are recorded in new source/formulation cohorts.

For the fixed arms, `reg_real_weight` and `reg_fake_weight` separately multiply
the two mean penalties. Both default to `1`; for example,
`Recipe(reg_arm="a_r1r2", reg_fake_weight=0)` applies real-only R1 at the
unchanged `reg_coeff / 2` coefficient. A zero-weight term performs no critic
forward. Nondefault weights require an explicit fixed arm and form a new
recipe/checkpoint identity.

`joint_reconstruction_weight` defaults to `0`. An explicitly bound joint
inverse host can add `recipe.joint_reconstruction_loss(reconstruction, target)`
to its generator/encoder loss. Continuous values use mean squared error;
passing `categorical_dim` interprets the reconstruction as raw logits and uses
stable categorical cross entropy. Targets are detached. The five-word joint
fixture consumes this objective as `G(E(real_words))` using its existing real
batch, without extra draws or a change to generation. Scalar hosts without a
joint inverse are inapplicable; ordinary AE/VAE reconstruction keeps its own
declared objective.

`get_recipe("bcap")` selects fixed BCAP with zero-momentum `dualnorm`,
`constraint_geometry_mode="direction_blend"`, and constant G/D/prior step sizes.
Direction blending protects existing task objectives across the joint
generator/encoder/prior optimizer; the discriminator retains its DualNorm step
and fixed cap above. `GANTrainer` binds its existing adversarial objective
automatically. [Custom component loops](#protected-backward-for-bcap-component-loops)
must bind their existing protected losses before the generator-side step.
The preset disables the critic spike
guard, EMA anchor, A2 latent damping, direct-particle gain, prior regularization,
EMA averaging, and additive input/output training noise. Its defaults are
coefficient 1, cap 1, penalty every update, G/E step `.012`, D step `.018`
(`d_lr_mult=1.5`), and sampled-prior row step `.03` (`prior_lr_mult=2.5`).
`optimizer_momentum=0`; `loss` defaults to `non_saturating`, with
`optimizer_smoothing=.001` and `optimizer_convolution="per_offset"`.
Transport weights remain zero, `critic_step_mode="none"`, and
`optimizer_svd_backend="native"`. Use
`get_recipe("bcap", constraint_geometry_mode="none")` for the earlier winning
DualNorm recipe. Pin `trainer="bcap"` on a model family to select these settings;
the shared default is optimized BISPF. `Recipe()` and partial constructors are
disabled. Use `get_recipe()` for new runs and `Recipe.from_dict(saved_fields)` for checkpoint restoration.
Each setting can be overridden explicitly.
`optimizer_svd_backend="newton_schulz"` selects a separate numerical backend
for annotated per-offset convolutions. It requires full DualNorm, positive
`optimizer_smoothing`, and `optimizer_convolution="per_offset"`. It batches
the smoothed polar computation with at most 32 augmented Newton-Schulz iterations,
checking convergence every four iterations from iteration 8. Optimizer-local
CUDA graphs replay four-iteration blocks. Orthogonality and the
relation to the original gradient must pass numerical checks; failing slices
fall back to native SVD. CPU kernels and kernels whose largest channel-matrix
dimension is at most 32 also retain native SVD.
Dense weights retain native SVD; prior, bias, learning rates, and momentum laws
are unchanged. The accelerated branch omits native SVD's hard numerical-rank
cutoff; finite iterations and floating-point rounding also differ. This option
is a declared trainer change requiring its own quality evidence. Its checkpoints
bind solver version 3 and reject earlier experimental solver states. CUDA graph
caches are disposable execution state and are rebuilt after restoration. Native
remains the explicit `get_recipe("ka2")` backend.

`optimizer_svd_backend="batched"` is the `bispf_optimized` CUDA execution backend for
per-offset DualNorm convolutions. Channel matrices with both dimensions at
most 32 use one batched native SVD per kernel, preserving the per-matrix rank
cutoff and absolute smoothing. With the quintic rule, larger matrices replay
the existing FP32 polynomial in optimizer-local CUDA graphs. Dense, bias,
prior-row and CPU updates retain their original laws. Batched SVD may select
different numerical kernels; graph input copies use contiguous storage, which
can change reduction rounding for strided kernels. Bitwise agreement with the
native backend is not promised. Checkpoints bind backend version 1; graph caches are rebuilt after
restore and are omitted from copies and checkpoints.

`optimizer_dense_profile="gram_fp64_guarded_v1"` explicitly selects the
versioned guarded FP64 Gram/eigh profile for ordinary FP32 dense matrices:

```python
recipe = get_recipe("bispf_optimized",
                    optimizer_dense_profile="gram_fp64_guarded_v1")
# Direct component-loop equivalent:
from particlegan.optim import NormalizedOptimizer
optimizer = NormalizedOptimizer(parameters, family="dualnorm", momentum=0,
                                smoothing=.001,
                                dense_profile="gram_fp64_guarded_v1")
```

It requires full DualNorm, zero momentum, smoothing exactly `.001`, and the
`native` or `batched` SVD backend. Only unannotated, two-dimensional strided
FP32 CPU/CUDA parameters take this path. Annotated convolutions, vectors/biases
and sampled-prior rows retain their laws, including batched/quintic convolution
execution. No global precision, random-state or backend setting changes.

The profile promotes **before** forming a scaled FP64 Gram matrix, then computes
the full hard-thresholded, smoothed polar direction. The strict cutoff remains
`max(G.shape) * eps_float32 * largest_singular_value`, with original dimensions
and absolute damping `.001`. The direction is cast to FP32 before the existing
parameter update. A fixed-margin error model checks orthogonality, factor
residual, negative eigenvalues and cutoff ambiguity on that same decomposition.
This screen is not an outward-rounded interval certificate or quality proof.

`GramGuardError` rejects an unqualified direction; technical solver errors
propagate. There is **no fallback, retry or transactional rollback**. Earlier
leaves, clocks or protected state may already have changed. Discard the failed
update and restore complete training state, including weights, optimizers,
pending protection and caller-owned RNG/sampling state, before continuing.

This is a numerical variant, with rounding that can differ from default SVD.
`optimizer_dense_profile="default"` retains the existing path and omits the new
field from legacy recipe/optimizer packets. Selected packets bind the profile
and reject cross-profile restore. Restore saved resolved recipes; do not relabel
an existing checkpoint. See the [qualification and reproduction commands](../reports/gram-profile/README.md).

`training_precision="bf16_channels_last"` configures `GANTrainer` for
BF16 training for CUDA generator/critic networks containing a Conv2d or
ConvTranspose2d whose largest channel-matrix dimension exceeds 32. These
networks use channels-last parameter storage and critic inputs. Only network
forwards use autocast; outputs, losses, master parameters, optimizer arithmetic
and prior regularization remain FP32. Sampling and evaluation run outside this
training autocast. Smaller, CPU and nonconvolutional networks retain their
original execution, as do CUDA devices without native BF16 support. This is a numerical trainer change requiring its own
quality evidence. Caller-owned component loops do not acquire autocast merely
by carrying this field; they must explicitly implement the declared policy.

`emulate_linear_training_(module, "nvfp4" | "mxfp8")` is a separate opt-in
software reference for Blackwell-format Linear forward and backward operands.
Assign its return value if the root is itself a Linear. It preserves FP32
master parameters, uses BF16 GEMM outputs and checkpoints private stochastic
rounding streams. It provides no native speed, storage or kernel-parity claim;
second derivatives use explicit straight-through quantizer surrogates. The
[4-bit/8-bit stretch protocol](../reports/forge/precision-stretch/README.md)
defines the format/scaling rules, higher-precision operations, gates and native
validation still needed. Existing precision settings and recipes are unchanged.

The public BISPF default selects both policies:

```python
recipe = get_recipe()
# Previous 0.9.0 execution settings, including for matching native checkpoints:
previous = get_recipe("bispf_optimized", optimizer_svd_backend="native",
                      training_precision="float32")
```

See the [source-bound throughput study](../reports/forge/bispf-throughput/README.md)
for measured workload limits and quality gates. Save resolved fields and restore
them with `Recipe.from_dict(saved_fields)`; do not resolve a new default when resuming.

`optimizer_convolution_rule="quintic"` selects a separately declared hybrid
optimizer: large CUDA per-offset channel matrices use five fixed FP32 polynomial
steps on `[M; optimizer_smoothing I]`, with coefficients
`(3.4445, -4.775, 2.0315)` and final scaling by `1/1.21`. This bounded filter
changes the update law; it is not an exact polar solve or PyTorch Muon. The
experimental `"frobenius"` rule uses `M / sqrt(||M||_F^2 + smoothing^2)` and
failed the direct wide-channel mode-coverage controls. Neither rule changes the
original channel/kernel factor, momentum, learning rates or sampled-prior law.
They require full DualNorm, per-offset convolution and the native SVD backend.
Only CUDA kernels whose largest channel count per group exceeds 32 use the new
rule. CPU, tiny convolution, dense, bias and prior paths retain native behavior.
The default `"polar"` preserves existing serialization and native checkpoints;
nondefault rules bind their own checkpoint versions. See the
[BISPF investigation](../reports/forge/bispf-optimized/README.md) for measured
quality and latency, scope and reproduction sources.

When selecting another optimizer family, use `bcap_adam` as the base or also
set `constraint_geometry_mode="none"`, `optimizer_smoothing=0.0`, and
`optimizer_convolution="none"`. Direction blending requires zero-momentum full
DualNorm; a positive `optimizer_momentum` likewise requires explicitly setting
`constraint_geometry_mode="none"`.
The resolved formulation is `bcap`. Selecting this preset does not change
historical `Recipe(reg_arm="b_cap")` configurations. A declared MoG prior still
has its kernel noise; that distribution is independent of additive training
noise. Models, initialization, prior and execution budget belong to the caller.

The selected direction-only recipe passed **6/6 Tier 1** and **9/21 Tier 2**
in the [ordinary matched comparison](../reports/forge/bcap-default-baseline/README.md).
Its matched `constraint_geometry_mode="none"` control passed **6/6** and
**7/21**, respectively. The [default selection](../reports/forge/bcap-default-baseline/DEFAULT_SELECTION.md)
records the owner-directed named-preset and benchmark-family choice, including
the independent saved-evidence audit. Remaining failures stay in the required
denominator; calibration and scale transfer remain unestablished.

The earlier `constraint_geometry_mode="none"` recipe also passed **6/6 Tier 1**
and **7/21 Tier 2** in the separate
[96-configuration search](../reports/forge/bcap-tier2-search/README.md), tying two
alternatives and improving on that study's relativistic/smoothing-1e-5 control's
6/21. Its [historical default selection](../reports/forge/bcap-tier2-search/DEFAULT_SELECTION.md)
and original source, task and scoring contracts retain their meaning.
`get_recipe("bcap_adam")` preserves the earlier native `torch.optim.Adam`
preset with betas `(0, .999)`, G/D LR `.00425` and prior LR `.0085`.
Restore old checkpoints with `Recipe.from_dict(saved_fields)`; recipe labels also
belong to checkpoint identity. An omitted saved geometry mode resolves to
`none`; loading that checkpoint into the new active preset is a recipe mismatch,
rejected before live state mutation. Save the resolved recipe and complete
optimizer/trainer state, including geometry state and every consumed RNG stream.
Forge's `forge-api-v1` declarations retain their
historical `recipe_preset="bcap"` base through this explicit Adam preset, so
existing cards, configuration IDs and evidence keep their original meaning.
The measured dualnorm cards explicitly pin their optimizer settings.

### `ParticleRegularizer`

```python
ParticleRegularizer(target_std=1.0, eps=1e-4, weight=1.0)
```

Call with a floating tensor `[N, z_dim]`. Returns a weighted scalar combining
a hinge below `target_std` for each dimension's standard deviation and a
penalty on off-diagonal covariance. It does not force a Gaussian distribution.
Fewer than two rows produce a differentiable zero. It accepts arbitrary latent
rows and does not require a `ParticlePrior` object.

For selected sampled rows, use `spread(prior.z[indices.unique()])`. Deduplication
belongs to the caller; repeated row values are otherwise counted repeatedly.

## DDGAN

```text
DDGAN(alpha_bar=(1.0, 0.9, 0.5, 0.05, 0.0001), *,
      device=None, dtype=None, validate_args=True)
```

An `nn.Module` containing Gaussian diffusion coefficients as buffers.
`alpha_bar` starts at 1 and strictly decreases while remaining positive.
`process.steps == len(alpha_bar) - 1`.

| Method | Contract |
| --- | --- |
| `forward_pair(x0, t, rng=None, *, generator=None)` | Coupled real samples `(x_prev, xt)` from forward corruption; pass either RNG argument |
| `reverse(x0, xt, t, eta)` | Reverse transition from predicted clean `x0`, noisy `xt`, and caller-supplied noise `eta` |

Data has shape `[B, ...]`; `t` is a long tensor `[B]` with values `1..steps`.
Inputs and schedule share a device. Reverse inputs have identical shape and
dtype. `reverse` computes `A[t] * x0 + B[t] * xt + sqrt(posterior_var[t]) * eta`;
the final transition at `t=1` has zero noise variance. It preserves gradients
through its inputs and does not add a reconstruction or diffusion MSE loss.

For DDGAN training, G predicts clean data, then D judges the **reverse
transition**. Hold `xt`, labels, and time fixed for the candidate gradient penalty.
For instance, given caller-defined `G`, `critic`, `prior`, and a labeled batch:

```python
from particlegan import DDGAN

process = DDGAN().to(real.device)
t = torch.randint(1, process.steps + 1, (len(real),), device=real.device)
x_prev, xt = process.forward_pair(real, t)
z, indices = prior.sample(len(real))
fake_prev = process.reverse(G(z, labels, xt=xt, t=t), xt, t, torch.randn_like(xt))
d_penalty = penalty(lambda x: critic(x, labels, xt=xt, t=t)[0],
                    x_prev, fake_prev.detach())
```

Learned latent particles, forward-corruption noise, terminal `x_T`, and reverse
noise `eta` are separate sources of randomness. The selected denoising recipe
uses Gaussian corruption, Gaussian terminal state, and fresh Gaussian `eta`.

## UCD

```text
UCD(network, num_classes, *, target="class", num_steps=None, validate_args=True)
```

An `nn.Module` wrapping your logit network. Calling
`critic(x, labels, xt=None, t=None)` returns `(selected_score, logits)` with
shapes `[B]` and `[B, heads]`. Labels are long tensors `[B]` in `0..num_classes-1`.
Class labels select an output head; they are not inputs to `network`.

| Use | Network call | Logits | Selected head |
| --- | --- | --- | --- |
| One-shot, `target="class"` | `network(x)` | `[B, C]` | `labels` |
| DDGAN, `target="class"` | `network(x, xt=xt, t=t)` | `[B, C]` | `labels` |
| DDGAN, `target="time_class"` | `network(x, xt=xt)` | `[B, T*C]` | `(t-1)*C + labels` |

Joint time/class heads require `num_steps=T`. `critic.ucd_labels(labels, t=None)`
returns the head indices. The same operation is available without a wrapper:

```text
ucd_labels(labels, timestep=None, *, num_classes, target="class",
           num_steps=None, validate_args=True)
ucd_scores(logits, labels, timestep=None, *, num_classes, target="class",
           num_steps=None, validate_args=True)
ucd_loss(real_logits, fake_logits, targets, weight=0.02)
```

`ucd_loss` is `weight * (CE(real_logits, targets) + CE(fake_logits, targets))`.
The selected recipe adds it to D's loss. The function does not detach either
logit tensor, so detach fake samples before computing D's logits.

`ucd_scores` selects the adversarial scores directly from your model's existing
`[B, heads]` logits and returns `[B]`, preserving gradients. Use it when your
pipeline already computes logits or needs to keep its existing model and
checkpoint structure. `UCD` uses this same function internally. Joint time/class
selection requires `num_steps=T`.

DDGAN and UCD numeric bounds checks can synchronize CUDA. Set
`validate_args=False` only for already validated times/labels; shape, dtype,
and applicable device checks still run.

## Particle autoencoders

Set `prior_kind="mog"`, `sigma_rel=.025` and `encoder_mode="ae"` or `"hard"`
to add reconstruction encodings to caller-owned networks and loops. Set
`model="ddgan"` when supplying a diffusion generator. These component choices
share the winning optimizer/loss defaults. Hard VAE selects one particle with
prior-matching Gaussian noise and constant joint KL; reconstruction adds no KL.

| API | Contract |
| --- | --- |
| `recipe.encode(query, prior, offset=None, draws=2, generator=None)` | `ParticleEncoding`; AE requires offset and returns one draw; VAE rejects offset |
| `particle_ae(query, offset, prior, temperature=.25, distance_reduction="sum", offset_bound=3)` | Deterministic bounded-offset encoding |
| `particle_vae(query, prior, temperature=.25, distance_reduction="sum", draws=2, hard=True, generator=None)` | Default constant-KL posterior; `hard=False` opts into categorical sampling |
| `encoding.reconstruction_loss(prediction, target)` | MSE only, with score-gradient correction for categorical mode |
| `encoding.negative_elbo(prediction, target, observation_sigma=.03)` | Explicit Gaussian negative ELBO in nats, including joint KL; rejects AE |
| `recipe.make_optimizers(G, D, prior, encoder=E)` | Adds E at G's LR; deduplicates shared parameters |

Codes are `[B,S,latent_dim]`, predictions `[B,S,...]`, targets `[B,...]`.
`encoding.kl` is per-input joint KL; `log_probs` exposes the true posterior
(or None for AE). Optional categorical training needs at least two independent
draws. See the [full guide](particle-autoencoders.md) for defaults, runnable
examples, gradient caveats, DDGAN integration and measured evidence.

## Recipes and defaults

```python
get_recipe(**overrides)        # Current default: EMPF (DEFAULT_RECIPE = "empf").
get_recipe("gan", **overrides) # Named components, current shared hyperparameters.
get_recipe("empf", **overrides) # Selected EMPF/DualNorm/quintic recipe.
get_recipe("ka2", **overrides) # Previous public trainer, explicitly pinned.
get_recipe("ae_gan", trainer="ka2", **overrides) # Model with a pinned trainer.
get_recipe("bcap", **overrides) # Fixed BCAP, direction blend, constant DualNorm rates.
get_recipe("bcap_adam", **overrides) # Earlier fixed BCAP/native Adam control.
get_recipe("bispf_optimized", **overrides) # Explicit BISPF/DualNorm/quintic recipe.
get_recipe("bispf", **overrides) # Original BISPF/formulation optimizer preset.
get_recipe("halloween", **overrides) # Explicit historical optimizer/loss transfer.
get_recipe("e22", **overrides) # Schedule-free E22 policy, explicit task settings.
get_recipe("e22_routed", **overrides) # Conditional dense-bank paired adaptation.
recipe.replace(**overrides)   # A new immutable Recipe.
recipe.to_dict()              # Restorable fields; compatible defaults stay implicit.
get_recipe_fields(**overrides) # Plain complete fields; validate with Recipe before training.
Recipe.from_dict(resolved_dict)       # Restore explicit fields from a saved run.
```

`get_recipe(name="gan", *, trainer=None, **overrides)` selects components without constructing a
training loop. Explicit keyword fields override the selected configuration.

To promote a future registered trainer preset, change this single line in
[`particlegan/recipes.py`](../particlegan/recipes.py):

```python
DEFAULT_RECIPE = "empf"
```

The selector affects `get_recipe()`, `get_recipe("gan")` and all model-family
recipes. Explicit presets and `Recipe.from_dict(saved_fields)` remain independent of
it. Save resolved recipes, rather than resolving the current default when
resuming. The selected recipe matches the complete
[`empf-v2` candidate](../configs/forge/ideas/empf-v2.json)
(`82/115` counted view requirements and `22/61` fixed-suite passes in the
[technique inventory](../reports/forge/technique-inventory.md)); it is an
owner-directed research selection and adds no accepted calibration, Tier 3
confirmation, universal architecture or natural-image quality claim. Explicit
`get_recipe("bispf_optimized")` retains the complete
[`bispf-throughput-v1` candidate](../configs/forge/ideas/bispf-throughput-v1.json)
([19 passes and 11 failures](../reports/forge/bispf-throughput/README.md) in its
30-task readout; 1.02–2.15× observed matched DCGAN speedups over the 0.9.0 quintic
default across six tasks).

`empf` uses the online *e-means* prototype-chord Born-Infeld penalty (`ebi_beta=6.0`
with state-driven settling sharpness) and zero-mean logit gauge, non-saturating loss,
zero-momentum DualNorm, smoothing `.001`, direction blending, learned sampled-prior row
updates and DoF-aware EMPF prior regularization at `.01` (`ucd_weight=.02`). `bispf_optimized` uses the Born-Infeld chord penalty and zero-mean logit gauge,
non-saturating loss, zero-momentum DualNorm, smoothing `.001`, direction blending,
learned sampled-prior row updates and SBNP regularization at `.01`. Generator,
critic and prior step sizes are `.012`, `.018` and `.03`. Both schedule floors
are `1`, so rates stay constant. Batched SVD handles small CUDA convolution
kernels; larger quintic kernels use CUDA graphs. Supported wide convolutional
networks use BF16/channels-last training. CPU and dense transformer matrices
retain the native polar/SVD rule; transformer
throughput is a separate profiling question. `get_recipe("bispf")` and
`get_recipe("bcap")` retain their explicit presets; `Recipe.from_dict` preserves
historical snapshots without selecting a new default.

`get_recipe_fields(name="gan", trainer=None, **overrides)` returns all declarative
fields without constructing or validating runtime components. Configuration
tools may load `particlegan/recipes.py` directly without importing Torch;
ordinary `from particlegan import ...` still imports the runtime package.
Preset and field names are checked, while `Recipe.from_dict(fields)` validates values.
Persist the returned complete fields with the run so later defaults cannot
change its training method.

BISPF checkpoints include transport residual/data-scale moving averages and
the exact thermal scale alongside optimizer counters. Earlier normalized and
plain-Adam BISPF packets omitted those earned moments and are rejected with an
explicit compatibility error; use their original package source to inspect or
continue them. Historical formulation-optimizer packets with the moments but
without an exact thermal scale retain their previous reconstruction behavior.
New packets preserve that scale exactly. BCAP and KA2 observer packets keep their
existing formats.

Model families share the default optimizer, loss, penalty and schedule;
`trainer="ka2"` pins an explicit trainer while retaining the selected model
components. `trainer` applies only to model-family names. The
`bcap` preset supplies fixed caps, zero-momentum DualNorm, direction blending
and constant rates, with zero additive training noise. Its standard
`GANTrainer` loop binds the protected adversarial loss automatically; component
loops use the [protected-backward contract](#protected-backward-for-bcap-component-loops).
The `e22` preset selects DV12 stationarity control with per-row evidence,
critic-feature birth/death, learned output noise and served averaging; it
requires no research JSON or training horizon.
Unknown names
and fields are rejected. Restore a complete saved configuration with
`Recipe.from_dict(saved_fields)`; use `recipe.replace(name="my-run")` to label a run.

| Name | Components and dimensions |
| --- | --- |
| `gan` (default) | Scalar GAN, 20,000 particles, latent dimension 2, no sampling noise (inherits `DEFAULT_RECIPE = "empf"`) |
| `empf` | Selected EMPF research default; online *e-means* prototype bank, settling Born-Infeld penalty, DualNorm, optimized convolution filter, direction blending, EMPF prior `.01` and constant rates |
| `bispf_optimized` | Explicit optimized BISPF recipe; DualNorm, optimized convolution filter, direction blending, SBNP `.01` and constant rates |
| `bispf` | Original BISPF recipe with relativistic Sinkhorn loss and formulation optimizer |
| `bcap` | Scalar GAN, 20,000 particles, latent dimension 2; fixed BCAP, direction blending and DualNorm at constant rates |
| `bcap_adam` | Earlier pure fixed BCAP preset with native Adam and constant rates |
| `halloween` | Scalar GAN; distinct G/D rates, moments and epsilon, dense TensorFlow-v1 Adam, least-squares labels `(-1,1,1)`, constant rates and zero critic penalty |
| `e22` | Scalar GAN, 20,000 particles, latent dimension 2, batch 2,048; E22 controls with learned output noise initially .029 |
| `e22_routed` | Same E22 controls with `row_policy="routed_paired"`; requires explicit context/routing/feature callbacks and guard observations |
| `mog` | GAN, 400 MoG components, latent dimension 2, relative sigma .025 |
| `ddgan` | DDGAN, UCD with 4 classes, discrete particles |
| `ddgan_mog` | DDGAN, UCD with 4 classes, 400 MoG components, relative sigma .025 |
| `ae_gan` | GAN, AE encoding, 400 MoG components, latent dimension 2 |
| `vae_gan` | GAN, hard VAE encoding, 400 MoG components, latent dimension 2 |
| `ae_ddgan` | DDGAN, AE encoding, 1,024 MoG components, latent dimension 64, batch 64 |

The encoder families use relative sigma .025. `ae_ddgan` uses mean routing
distance and temperature .125; the others use sum distance and temperature .25.
These component choices retain the original API's model-family structure with
the new shared hyperparameters. The GAN development-suite evidence does not
establish convergence of those hyperparameters for every AE/VAE/DDGAN setup.

`halloween` transfers the optimizer/loss sections of an archived HyperGAN
configuration. Its architecture, auxiliary loss, original runtime and decay
clock are unbound. The preset makes its constant rates and learned-prior
settings explicit; it has no trained qualification. See the
[search-space and transfer guide](forge-search-spaces.md) for the exact scope.

E22's task inputs are `num_particles`, `z_dim`, `batch_size` and
`output_noise_std`; set them explicitly for a new task. Its default
`row_policy="independent"` retains the unconditional equal-mass row law.
Conditional densely blended banks declare `row_policy="routed_paired"` and
bind `RoutedRows`; routing weights alone do not provide row support evidence.
The [independent guide](e22.md) and [routed guide](e22_routed.md) define the
respective evidence, ownership, lifecycle and serving contracts. `GANTrainer`
supports the independent formulation; routed observations belong to the
caller-owned loop.

### Components that change a loop

The recipe publishes settings and small operations; the caller composes them.
UCD settings describe score selection and class-loss weight. AE/VAE settings
describe encoding and reconstruction. Neither creates a training strategy:

```python
ucd_recipe = get_recipe("gan", conditioning="ucd", num_classes=4)
ae_recipe = get_recipe("ae_gan")
vae_recipe = get_recipe("vae_gan")
```

| Component | Recipe supplies | Caller controls |
| --- | --- | --- |
| UCD | Class count, target, auxiliary loss weight | Labels, discriminator heads, score selection, adding `ucd_loss` to the D objective |
| AE/VAE | Prior, routing settings, `encode`, reconstruction weight, optional encoder optimizer group | Encoder forward pass, reconstruction/adversarial loss composition, backward and optimizer steps |
| DDGAN | Model selection and diffusion schedule | Corruption, timestep sampling, reverse transitions and update order |

See the [DDGAN/UCD loop](#a-minimal-ddgan--ucd-loop) and
[AE/VAE example](../examples/particle_autoencoder.py). `GANTrainer(recipe, G, D)`
is a separate, optional helper for unconditional particle GANs; it explicitly
rejects UCD, encoders and DDGAN. `Recipe` has no trainer factory or training step.

Discrete learned particles are the zero-noise limit of a mixture of Gaussians.
The current `ParticlePrior` reads raw centers; `MoGParticlePrior` standardizes
centers by default, even when sigma is zero. For matching center reads use
`sigma_rel=0, standardize=False`. The default GAN retains `ParticlePrior`;
unifying implementations or promoting a noisy MoG default requires separate
validation.

### Explicit composition

Choose components explicitly, while inheriting the common training defaults:

```python
recipe = get_recipe(model="ddgan", conditioning="ucd", num_classes=4,
                    prior_kind="mog", sigma_rel=.025, num_particles=400)
prior = recipe.make_prior().to(device)
process = DDGAN(recipe.alpha_bar).to(device)
opt_g, opt_d = recipe.make_optimizers(G, D, prior)
```

| Shared field | Current `get_recipe()` default |
| --- | --- |
| `model`, `conditioning`, `num_classes` | `gan`, `scalar`, `None` |
| `z_dim`, `num_particles` | `2`, `20_000` |
| `prior_kind`, `sigma_rel`, `standardize` | `particles`, `0`, `True` (standardize applies only to MoG) |
| `lr`, `d_lr_mult`, `prior_lr_mult` | `.012`, `1.5`, `2.5` |
| `optimizer_family`, `optimizer_momentum`, `optimizer_smoothing` | `dualnorm`, `0`, `.001` |
| `optimizer_convolution`, `optimizer_svd_backend` | `per_offset`, `native` |
| `optimizer_convolution_rule` | `quintic` (CUDA wide convolutions only; native fallback elsewhere) |
| `betas`, `d_betas`, `prior_betas` | `(0, .999)`, `None`, `None` (role overrides inherit shared betas) |
| `eps`, `d_eps`, `prior_eps` | `1e-8`, `None`, `None` (role overrides inherit shared epsilon) |
| `adam_variant`, `loss_labels` | `pytorch`, `(0,1,1)` |
| `lr_schedule` | `cosine` (also `constant`, `exponential`) |
| `lr_decay_rate`, `lr_decay_steps`, `lr_decay_staircase` | `.96`, `50000`, `False` (exponential schedule only) |
| `loss` | `non_saturating` (also `relativistic`, `hinge`, `wasserstein`, `least_squares`) |
| `constraint_geometry_mode` | `direction_blend` |
| `critic_formulation`, `reg_arm` | `bispf`, `None` |
| `reg_coeff`, `reg_kappa` | `1`, `1` (Born-Infeld penalty strength and chord scale) |
| `reg_every` | `1` (apply the penalty every k-th step at k× coefficient) |
| `prior_update`, `prior_regularizer`, `prior_reg`, `prior_l2`, `ema_decay` | `learned`, `sbnp`, `.01`, `0`, `0` |
| `lr_anneal_start`, `lr_floor` | `.6`, `1` (constant prior rate) |
| `network_lr_horizon_cap`, `network_lr_floor` | `None`, `1` (constant G/D rates) |
| `reg_anchor_min_decay`, `reg_anchor_weight`, `critic_r1_real` | `.9`, `0`, `True` (inactive KA2 settings) |
| `d_guard_ratio`, `d_guard_min_steps` | `0`, `200` (guard disabled) |
| `latent_damping_max_rate`, `direct_particle_gain` | `0`, `False` |
| `direct_particle_betas` | `(0, .9)` (`make_generator_optimizer(direct_particles=...)`) |
| `input_noise_std`, `input_noise_anneal_end` | `0`, `.1` (noise disabled) |
| `output_noise_std`, `output_noise_warmup` | `0`, `0` |
| `batch_size`, `total_steps` | `2048`, `7_000` |
| `ucd_target`, `ucd_weight` | `class`, `.02` |
| `alpha_bar` | `(1, .9, .5, .05, .0001)` |

Architectures, model/prior choices, data and resource budgets belong to the
caller.

| Optional factory | Result |
| --- | --- |
| `recipe.make_prior(**kwargs)` | Prior selected by `prior_kind`, tables drawn at random ([initialize](#initializing-priors) for R2) |
| `recipe.make_loss()` | `GANLoss(recipe.loss, labels=recipe.loss_labels)` (default non-saturating logistic) |
| `recipe.make_optimizers(G, D, prior=None, *, encoder=None, ema_critic=None, separate_encoder_groups=False, **adam_kwargs)` | Build `(opt_g, opt_d)` over the weights as given (see below) |
| `recipe.make_critic_optimizer(D, *, ema_critic=None, **adam_kwargs)` | Recipe-selected optimizer for one (additional) critic (see below) |
| `recipe.make_generator_optimizer(params, *, modules=None, latent_table=None, direct_particles=None, **adam_kwargs)` | Recipe-selected optimizer for generator-side params; `modules` binds kernel semantics alongside custom groups |
| `recipe.make_critic_penalty(opt_d, *, output=None, collect_stats=False, **penalty_kwargs)` | The critic penalty paired with a critic optimizer (see below) |
| `recipe.make_prior_regularizer(prior=None, reference=None, **kwargs)` | Selected prior penalty, including SBNP, with `weight=recipe.prior_reg` applied once. `prior` and `reference` are aliases for the same initialized prior or tuple of priors. |
| `recipe.prior_regularizer_checkpoint(module)` | Optional state for an active stochastic prior regularizer |
| `recipe.restore_prior_regularizer(module, state)` | Validate/restore regularizer state for exact caller-owned continuation |

### Regularization factories

The recipe, not the caller, chooses the regularization formulation, and your
loop stays plain PyTorch. The explicit `get_recipe("ka2")` recipe returns KA2 implementations
(`particlegan.ka2.KA2CriticAdam` and `CriticPenalty`, with
`particlegan.k3p.K3PGeneratorAdam`). Other recipes select their declared
optimizers and penalties; the named BCAP component loop uses the
[protected-backward call](#protected-backward-for-bcap-component-loops) below.
The previous K3P critic replays through
`benchmarks.legacy.recipe` ([K3P](k3p.md#replaying-k3p)).

The following loop and Adam controller descriptions use the explicit KA2 recipe.
The default EMPF recipe (`"empf"`) and explicit `"bispf_optimized"` recipe use the protected-backward contract below.

When binding G and E to `UpdatePolicy`, pass `separate_encoder_groups=True`.
Native Adam then uses distinct homogeneous generator and encoder groups with
identical network settings. Parameter updates and per-parameter Adam state are
unchanged; the default retains develop's unlabelled G/E checkpoint groups.
Normalized optimizer families already separate these roles.
For BISPF, declare a native continuous feature vector with
`recipe.make_critic_penalty(opt_d, input_domain="continuous")`, or pass
`penalty_options={"input_domain": "continuous"}` to `GANTrainer`. The declaration
retains the continuous RMS gradient norm, smooth Born–Infeld cap, gradient floor
and logit gauge at widths such as 1024 or 4096. It uses the existing continuous
law and keeps the native tensor shape. Explicit `"categorical"` selects the
categorical law for `[batch, categories]` matrices, including small vocabularies.
The default `"auto"` preserves the declared shape gates: with default gates,
matrices of width at least 170 use the categorical law; lower widths and image
tensors keep their existing behavior. Explicit declarations take precedence over
the automatic simplex gate, including when disabled. Other shape gates retain
their settings. This is a data-domain declaration, not an empirical quality
qualification. `GANTrainer` checkpoints bind `penalty_options` and refuse a
different declaration on restore. Component-loop callers must save and check
their selected penalty options along with the recipe and optimizer state.

See the [input-domain walkthrough and runnable categorical example](bispf-input-domains.md)
for the gradient-norm arithmetic, data representation, measured training outputs
and checkpoint continuation.

```python
opt_g, opt_d = recipe.make_optimizers(G, D, prior, ema_critic=copy.deepcopy(D))
penalty = recipe.make_critic_penalty(opt_d)
d_loss = adv_d + penalty(D, real, fake)                  # or penalty(D, x, fake, labels, t=t)
opt_d.zero_grad(); d_loss.backward(); opt_d.step()       # guard, Adam, KA2 controller + EMA
opt_g.zero_grad(); g_loss.backward(); opt_g.step()       # Adam with A2 latent damping

init.deterministic_orthogonal_(D2, seed=3)   # optional; 0/1/2 are the examples' G/D/E seeds
opt_d2 = recipe.make_critic_optimizer(D2, ema_critic=copy.deepcopy(D2))  # a second critic
penalty2 = recipe.make_critic_penalty(opt_d2)

torch.save({"G": G.state_dict(), "D": D.state_dict(), "D2": D2.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "opt_d2": opt_d2.state_dict()}, path)
```

- The KA2 formulation optimizers are `torch.optim.Adam` subclasses: `param_groups`, LR
  schedulers, closures and `state_dict()`/`load_state_dict()` work as usual.
  Their `state_dict()` adds a `"regularizer"` entry holding the EMA critic,
  KA2 controller, counters, guard count and A2/direct-particle histories, so the
  usual checkpoint above resumes bit-exactly.
- `ema_critic` is caller-allocated (e.g. `copy.deepcopy(D)`); the penalty
  requires it unless `reg_anchor_weight=0`. The optimizer freezes it and only writes it.
- `penalty(D, real, fake, *condition, **condition_kwargs)` returns a scalar.
  Conditioning is forwarded to the critic and its EMA. `D` may be the
  optimizer's critic, a submodule of it (one role of a shared module; the
  same-named EMA submodule is used) or a module wrapping one of those (e.g.
  `InputNoise(D, std, generator)`). A tuple/list output uses its first element
  unless `output=` selects the logits. The step for `reg_every` is the
  optimizer's completed step count + 1.
- `penalty.last_stats` holds the last call's stats when `collect_stats=True`;
  `penalty.diagnostics()` returns host scalars such as
  `{"blend_weight": ..., "clipped_tensors": ...}`; `penalty.ema_critic` is the
  paired EMA module.
- `make_optimizers` gives a learnable row-local `ParticlePrior` or
  `MoGParticlePrior(standardize=False)` table A2 damping (alone in its group
  with beta1 0) and the critic the spike guard. Standardized MoG reads couple
  rows and are not A2 eligible. `opt_g.prior_mechanisms` records this explicitly;
  `require_latent_damping=True` rejects an unavailable/disabled A2 hook before
  training. The component factory retains its historical optional behavior by
  default, while `GANTrainer` requires configured A2 on learned locations. Set
  `latent_damping_max_rate=0` and `d_guard_ratio=0` for plain Adam steps.
- `make_generator_optimizer(..., direct_particles=[...])` applies the
  direct-particle response to that param group.

Factory keyword arguments override constructor values for that call, without
changing the recipe. Optimizers exclude frozen parameters; G and prior have
separate groups at `lr` and `lr * prior_lr_mult`, while D uses `lr * d_lr_mult`.
`d_betas` / `d_eps` override D's shared settings; `prior_betas` / `prior_eps`
override the learned-prior group. G/E use shared `betas` / `eps`. Explicit
factory kwargs and parameter-group settings override those recipe values.
Set `prior_kind="mog"`, `sigma_rel` and `standardize` through the recipe, or
override them locally in `make_prior`. Nonzero sigma with the atoms kind is rejected.
If G contains the supplied prior, its parameters are included only once.
Additional Adam options such as `fused=True` or `eps=1e-8` are passed to both
optimizers; configure learning rates and betas through the recipe. You can
still construct optimizers yourself, including separate prior optimizers or
additional parameter groups for learned noise.

For architecture builders such as HNDL, pass the owning modules alongside
custom groups. The groups remain the authority for which parameters and rates
are updated; traversing `modules` supplies convolution layout without adding
other graph parameters:

```python
groups = [{"params": generator_parameters, "role": "generator", "lr": recipe.lr}]
opt_g = recipe.make_generator_optimizer(groups, modules=[generator, encoder])
base_rates = [group["lr"] for group in opt_g.param_groups]
# Conv kernels can split a caller's group. Apply schedules to every result.
for group_index, group in enumerate(opt_g.param_groups):
    scale = prior_scale if group["role"] in ("prior", "table") else network_scale
    group["lr"] = base_rates[group_index] * scale  # retain one base rate per group
```

Omit absent modules from this list. Passing a single module as `params` already
does this binding automatically. With explicit `modules`, frozen parameters
are removed and empty groups are omitted; bare historical parameter groups
retain their explicit ownership contract. Module-aware construction uses
effective per-parameter `requires_grad` and does not initialize loaded values.
For a frozen pretrained trunk and a trainable head, only the head enters the
optimizer. A trainable pretrained trunk keeps its loaded values and receives
the declared update law. Initialize only newly created heads/adapters before
constructing optimizers.

Freezing weights does not disable gradients to their inputs. Keep the forward
graph when a fixed feature backbone must send gradients to an upstream network;
manage train/eval mode separately for dropout and running buffers. Clear stale
gradients when freezing parameters after construction. Rebuild or
explicitly migrate optimizer groups and state when unfreezing parameters that
were omitted at construction. Save trainability, modes, buffers, the resolved
recipe and optimizer state in a caller-owned checkpoint. Custom high-rank
parameters, sparse gradients and quantized trainable weights need an explicitly
supported update law; this module binding does not infer one from tensor rank.

### Protected backward for BCAP component loops

`GANTrainer(get_recipe("bcap"), G, D, prior=prior)` records sampled prior rows
and binds its existing scalar adversarial objective automatically. No special
caller hook is needed for this standard unconditional GAN loop.

For a caller-owned component loop, build the joint generator/encoder/prior
optimizer through `recipe.make_optimizers(..., encoder=E)` or equivalent
role-named groups. Direction blending examines that optimizer's actual
normalized joint displacement against gradients of one or two existing scalar
host losses. It changes the update direction without adding a loss term,
changing a loss coefficient, or introducing new weights.

```python
from particlegan.optim.constraint_geometry import constraint_geometry_backward

# loss_g and loss_gan are your existing scalars from the same forward pass.
# Record actual sampled prior rows first when using a learned normalized table.
opt_g.zero_grad()
constraint_geometry_backward(loss_g, opt_g, (loss_gan,))
opt_g.step()
```

`loss_g` remains the complete original training objective. The protected tuple
contains one or two existing active scalar losses from that graph, covering the
joint G/E/prior parameter set. For a host that already owns both reconstruction
and adversarial objectives, pass `(loss_reconstruction, loss_gan)`; retain their
existing composition in `loss_g`. The helper binds their gradients before
backpropagating `loss_g`. Ordinary optimizers use their original backward through
the same helper. With active geometry, plain `.backward(); opt_g.step()` fails
before updating parameters because no protected losses were bound.

For learned normalized prior rows, call
`policy.observe_sampled_rows(indices)` or
`opt_g.set_sampled_rows(prior.z, indices)` with the actual generator-side sample
IDs before the protected-backward call. Reuse the existing forward pass, sampled
batch and objectives; protection adds no sampling calls or RNG draws. Keep all
G/E/prior groups in the same protected optimizer for the declared joint contract.

Direction blending addresses first-order conflicts. It performs no finite-loss
evaluation or line search, can stall on opposing objectives, and supplies no
finite-step loss-descent or GAN-convergence guarantee. To restore a custom loop's
earlier DualNorm behavior, explicitly select `constraint_geometry_mode="none"`.

### Other optimizer families

Fixed BCap or R1/R2 recipes also support optimizer experiments through
`optimizer_family`: `sgda`, `nsgda_global`, `nsgda_layer`, `ada_nsgda`,
`dualnorm`, `dualnorm_D_only`, and `particle_rownorm_only`. They use the same
public factories and `GANTrainer`, with the recipe's loss, penalty and
schedule unchanged. Disable the formulation's guard, anchor, latent damping
and direct-particle gain; `get_recipe("bcap")` already does this.
Changing its optimizer family also requires
`constraint_geometry_mode="none"`, `optimizer_smoothing=0.0`, and
`optimizer_convolution="none"`, or use `bcap_adam` as the base. For example:

```python
recipe = get_recipe("bcap", optimizer_family="adam",
                    constraint_geometry_mode="none",
                    optimizer_smoothing=0.0, optimizer_convolution="none")
```

This explicit override keeps the BCAP preset's rates and non-saturating loss;
`get_recipe("bcap_adam")` selects the earlier Adam control's complete settings.
The selected default itself uses:

```python
recipe = get_recipe("bcap", optimizer_family="dualnorm",
                    lr=.012, d_lr_mult=1.5, prior_lr_mult=2.5,
                    optimizer_momentum=0.)
trainer = GANTrainer(recipe, G, D, prior=prior)
```

These step sizes have different units from Adam learning rates and require
their own sweeps. `nsgda_global` normalizes the combined G/E gradient, D and
the prior separately; `nsgda_layer` normalizes each tensor. `ada_nsgda`
requires beta1 zero and grafts each tensor's unit-rate Adam update norm onto
its SGD direction, applying the scheduled rate once. `dualnorm` uses the
polar factor of every matrix gradient (including heads), scaled by
`sqrt(max(1, fan_out/fan_in))`, and L2-normalizes vectors. Its optional
`optimizer_momentum` is `0`, `.5`, or `.9`; matrices with gradient norm below
`eps` are skipped. Positive momentum requires `constraint_geometry_mode="none"`
when starting from the named BCAP preset. Every matrix size uses exact reduced SVD, with direction
`U diag(s > tau) Vh`, where `tau = max(rows, columns) * finfo(dtype).eps * s_max`.
Numerically null directions receive zero update instead of being amplified to
unit magnitude. The cutoff uses the computation dtype: float32 and float64 are
preserved; float16/bfloat16 inputs compute in float32 and cast the result back.
This replaces the former Newton--Schulz fast path for matrices larger than 1024,
so large full-rank matrices can cost more per update. Conv2d/ConvTranspose2d
weights require module bindings and `optimizer_convolution="per_offset"`,
enabled by the `bcap`, `bispf_optimized` and `empf` presets; unlabelled high-rank tensors
remain unsupported. The [convolution contract](dualnorm-convolution.md) specifies
channel-group layout, kernel scaling, smoothing and checkpoint compatibility.
Older checkpoints load their stored state, but continue
under this new rule; reproducing older trajectories requires their original
package source. No task gate or historical qualification is changed by this rule.

The `dualnorm` prior and `particle_rownorm_only` normalize each sampled prior
row, without momentum. Unsampled rows stay fixed even if a whole-table
regularizer creates gradients there. `GANTrainer` and
`UpdatePolicy.generate(..., rows=indices)` record actual generator-side draws.
A caller-owned loop that bypasses `generate` must call
`policy.observe_sampled_rows(indices)` in its generator phase, or
`opt_g.set_sampled_rows(prior.z, indices)` before stepping, and before the
protected-backward call when geometry is active. Repeated optimizer
setter calls union IDs; updates consume them. Checkpoints include pending IDs,
moments, momentum and the critic's observation-only step record.

The isolation arms retain native PyTorch Adam for other players. Pin
`optimizer_adam_lr` to the baseline's learning rate while sweeping `lr` for
the changed optimizer. `dualnorm_D_only` uses normalized `lr * d_lr_mult` on D,
baseline `optimizer_adam_lr` on G/E and baseline
`optimizer_adam_lr * prior_lr_mult` on the prior. `particle_rownorm_only` uses
normalized `lr * prior_lr_mult` on the prior and baseline Adam rates on G/E/D.
Direct generated-coordinate fixtures have no sampled prior table: their
generator coordinates follow the matrix/vector rule in `dualnorm` and retain
Adam in `particle_rownorm_only`. Use `get_recipe("bcap_adam")` as the
explicit control when constructing these isolation arms; native Adam retains
its existing update law.

`adam_variant="tensorflow_v1"` uses the named `TensorFlowV1Adam` dense update
law, adding epsilon before second-moment bias correction. Checkpoints include
optimizer application clocks, beta powers and moments. It rejects changing
betas, sparse/complex gradients, weight decay, AMSGrad and accelerated modes.
Recipe optimizers reject loading this variant's checkpoint into native Adam.
`lr_schedule="exponential"` counts completed whole training updates and uses
`lr_decay_rate ** (step / lr_decay_steps)`, or an integer exponent with
`lr_decay_staircase=True`; network and prior share that multiplier. `constant`
holds it at one. These schedules do not rescale with an execution budget.

`learning_rate_scale(step, total_steps, start=.6, floor=.05)` returns a Python
float: hold 1, then cosine decay to `floor`. `step` counts completed updates
(zero before the first update). It changes no optimizer state and clamps after
the horizon. EMA, update ratios, and scheduling remain caller-owned.

## TOML configuration

Constructors accept ordinary dictionaries from `tomllib.load`, `toml.load`, or
any other parser. The library does not read configuration files.

```toml
[particlegan]
name = "ddgan"
z_dim = 16
num_particles = 4096
num_classes = 8
betas = [0.0, 0.999]

# Alternative: configure components independently.
[prior]
z_dim = 16
num_particles = 4096
```

```python
import tomllib  # Python 3.10: install tomli and import it as tomllib.
from particlegan import get_recipe

with open("model.toml", "rb") as file:
    config = tomllib.load(file)
recipe = get_recipe(**config["particlegan"])
prior = recipe.make_prior(**config["prior"])
recipe = get_recipe(**{**config["particlegan"], "lr": 1e-4})
```

Choose recipe-owned values or independent component sections for your pipeline;
they are not merged automatically. Recipe arrays become immutable tuples.
The repository's experiment CLI accepts flat TOML/YAML files using its existing
experiment field names; see the [runner guide](experiment-runner.md).

## Loss augmentation and teacher/student pipelines

Each loss stands alone. For example, using latent features from your own model:

```python
from particlegan import ParticleRegularizer

spread = ParticleRegularizer(weight=0.1)
loss = existing_loss + spread(latent_features)  # [batch, latent_dim]
```

In a teacher/student pipeline, the teacher can supply the real targets. Freeze
teacher outputs when that is your intended gradient policy, and compose the
student objective yourself:

```python
with torch.no_grad():
    targets = teacher(inputs)
z, indices = prior.sample(len(inputs))
fake = student(z, inputs)
# After your D update, with D parameters frozen:
loss = supervised_loss(fake, targets)
loss = loss + adversarial_weight * gan.g_loss(D(fake), D(targets).detach())
loss = loss + spread(prior.z[indices.unique()])
```

Your pipeline controls teacher mode, conditioning, weights, gradient paths,
backward, and optimizer steps. None of these helpers require a dataset class,
teacher interface, or fixed training loop.

## Inference and checkpoints

For one-shot inference, save the matched EMA generator and prior from the
one-shot loop:

```python
torch.save({"generator": ema_g.state_dict(), "prior": ema_prior.state_dict(),
            "recipe": recipe.to_dict()}, "model.pt")
```

In the receiving application, construct the same G architecture and prior size,
then restore them. The recipe stores hyperparameters, not the generator class:

```python
from particlegan import Recipe

state = torch.load("model.pt", map_location=device, weights_only=True)
recipe = Recipe.from_dict(state["recipe"])
G = nn.Sequential(nn.Linear(recipe.z_dim, 64), nn.LeakyReLU(0.2),
                  nn.Linear(64, 2)).to(device)
prior = recipe.make_prior().to(device)
G.load_state_dict(state["generator"])
prior.load_state_dict(state["prior"])
G.eval()
prior.eval()
with torch.inference_mode():
    z, _ = prior.sample(64)
    samples = G(z)
```

For DDGAN inference, also restore the schedule (or reconstruct it from the saved
`alpha_bar`). Given your restored conditional G and labels, the reverse loop is:

```python
process = DDGAN(alpha_bar=recipe.alpha_bar).to(device)
with torch.inference_mode():
    x = torch.randn((len(labels), *data_shape), device=device)
    for step in range(process.steps, 0, -1):
        t = torch.full_like(labels, step)
        z, _ = prior.sample(len(labels))
        clean = G(z, labels, xt=x, t=t)
        x = process.reverse(clean, x, t, torch.randn_like(x))
```

Here `data_shape` excludes the batch dimension, for example `(3, 32, 32)`.
Use the model's dtype for `x`, and set G and prior to evaluation mode first.
Inference needs no discriminator or optimizer. Resuming training additionally
requires your D, optimizer, scheduler, EMA, update-counter, and RNG state.

The [implementation report](../reports/api.md) records the design, migration,
and validation. Lower-level names such as `VICRegLikeLoss`,
`DiffusionSchedule`, and `DrawSource` live in package submodules; the APIs
above are the public entry points.
