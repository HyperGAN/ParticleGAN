# R1/R2 implementation audit

The inspected public R1/R2 update matches the declared paired logistic loss,
squared input-gradient penalty and native Adam law. No supported production
correctness fix was found. This audit supplies software evidence, not a quality
qualification or a new comparison of recipes.

Inspected source is commit `a771841dd17bd8780c03f8ebb6e34b8dfe19bc94`.
[Exact source hashes and checks](implementation-audit.json) bind the audit to
the inspected files, including the newly added test.

The critic minimizes `mean(softplus(D(fake)-D(real)))` plus
`gamma/2 * (mean(||grad D(real)||²) + mean(||grad D(fake)||²))`.
The penalty detaches its input copies, sums input coordinates and averages
examples. There is no RMS divisor or cap. For the generator/prior phase, the
trainer resamples differentiable fakes, freezes critic parameters and evaluates
both scores after the critic update. Native Adam's observer counts updates;
it does not apply the historical K3P anchor, guard or damping. Schedule state
uses completed updates and the explicit recipe horizon; an execution cap does
not silently redefine that horizon.

The pinned [R3GAN loss implementation](https://github.com/brownvc/R3GAN/blob/19a7ddf463fbac2bd39b4c1c73d63f1c441c7403/R3GAN/Trainer.py)
supports these signs, detach paths and half-gamma factors. Its
[training loop](https://github.com/brownvc/R3GAN/blob/19a7ddf463fbac2bd39b4c1c73d63f1c441c7403/training/training_loop.py)
also mutates native Adam's beta2, so the current native bias-correction behavior
is shared with that reference. A different variable-beta correction would be
a separately declared trainer change, rather than a reference-parity fix.

The new [independent update oracle](../../../tests/test_r1r2_update_oracle.py)
writes out the scalar equations and uses native Adam directly, without the
repository's loss, penalty or schedule helpers. Seven consecutive one-update
comparisons span the beta2/gamma midpoint, endpoint and hold, and the start of
LR annealing. Seed 0 and public deterministic initialization bind the G, D and
learned MoG fixture. Separate real batches exercise the generator pairing;
matching checkpointable MoG streams exercise fresh fakes. Losses, penalties,
parameters, gradients and every native Adam state tensor agree within absolute
`2e-12` in CPU float64. Critic tensors remain exactly unchanged during the
generator/prior phase. Existing kernel parity, schedule, task-binding and loss
tests also pass: **110 passed** on Python 3.14.8 / Torch 2.14.0+cu130.

One deliberate reference difference remains: the shared penalty evaluates the
critic on detached copies instead of reusing its adversarial logits. This can
change running buffers or stochastic draws for BatchNorm, dropout or input-noisy
critics. The modern scalar toy cohort has no such layers or additive noise;
this difference does not explain its current failures. DCGAN performance
receipts separately declare batch-coupled BatchNorm derivatives and grant no
image-quality credit.

These tiny CPU software checks do not reproduce the paper's architecture,
fixed Gaussian prior, image budget or EMA serving, and do not test long-horizon
CUDA quality. Preserve the original failed scientific receipts and their gates;
this audit supports investigating measured optimizer/critic/prior dynamics
instead of changing signs, units or loss factors without evidence.
