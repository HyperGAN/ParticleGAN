# Native NS DDGAN failure: saved conditional field

The original control has a genuine conditional and final distribution failure.
Both step-1 panels fail throughout the 24-observation protocol, including the
central query. The later reverse steps supply enough Gaussian noise to conceal
much of the clean predictor's missing variance. This matches the mechanism
observed in the [earlier model-native diagnosis](../ddgan-failure-investigation/README.md),
whose different source, CPU runtime and BISPF recipe remain a separate cohort.
No matched successful model-native recipe was found.

The [compact receipt](ddgan-diagnosis.json) pins attempt
`32a62170283a41158410c13744709fb8`, source digest
`e481b63224fc2d8cf77b136ab54f06421f7ac0967e47509a63f95cfa42733464`,
original request/raw/API receipts, checkpoint, probe and raw analysis outputs.
The original native NS/AMSGrad recipe completed all 2,500 D/G/prior updates with
gamma .1, real/fake penalty weights 1, LR .0085, beta1 .5 for D/G, beta1 0 for
prior and beta2 .99. The task retains its learned 128-by-4 MoG prior, sigma .25,
batch 64, deterministic initializer, protocol seed 0, clean live sampling and
2,048 samples per panel. No evaluation or qualification result was changed.

| Original endpoint | Projection CDF error | Normalized mean error | Covariance eigenvalues |
| --- | ---: | ---: | ---: |
| Final ancestral samples | .32094 | .43367 | [.23183, 1.44115] |
| Step 1, q0 | .52395 | 1.00719 | [.14787, 1.91476] |
| Step 1, central q1 | .29191 | .39390 | [.32790, 2.06598] |
| Step 4, q0 | .04583 | .10458 | [.99132, 1.03331] |
| Step 4, q1 | .03182 | .04594 | [.95318, 1.01653] |

The unchanged bounds are CDF <=.12, normalized mean <=.25, covariance minimum
>=.65 and maximum <=1.4, jointly across all eight transitions and the final
chain, sustained for the final five observations. There are zero joint passing
observations. Final CDF spans .19923–.62285 across the original trace; its final
five values are .47608, .27704, .29215, .42486 and .32094. The last values are
not an isolated evaluation accident. Exact values remain in the receipt.

For the Gaussian law, the conditional reverse covariance is
`A_t² Var(x0 | xt) + posterior_var_t I`. Its learned-clean variance fraction is
100%/100% at step 1, 57.40%/39.78% at step 2, 15.95%/7.82% at step 3, and
1.05%/.47% at step 4. A high-noise transition pass therefore supplies weak
evidence for correct clean prediction. This is an exact variance decomposition,
not a claim that the later steps contain no learned information.

## Frozen critic controls

The [reproduction source](ddgan_critic_field_probe.py) uses all 128 learned prior
rows with deterministic Gaussian quadrature over latent and reverse noise. It
integrates the original eight fixed contexts with equal weight. The exact
Gaussian conditional moments and step-1 target are analysis-only oracles.
Orders 2 and 3 preserve every reported alignment and directional sign.

The mean-translation alignment is
`E[sigmoid(-D(fake)) * grad_fake D(fake)] dot (real_mean - fake_mean)`.
Positive assists conditional mean correction in output space. At order 3 it is
negative on seven of eight contexts, including central step-1 q1 (-.00423).
Step-1 q0 is positive (+.01522), despite its large variance and mean errors.
Thus a wrong output field coexists with the distribution mismatch, but cannot
explain every failing panel by itself. G's Jacobian and update geometry are not
included in this alignment.

The probe then differentiates these alignments along hypothetical D directions
using the saved, fixed AMS denominator. It neither updates moments nor takes a
training step. The old bias-corrected D momentum improves all eight alignments
locally. Fresh native R1+R2 gradients improve seven but worsen q0 at step 1;
fresh R1-only gradients have the same sign pattern. At q0 those derivatives are
+1.42e-4 for saved momentum, -1.05e-3 for fresh R1+R2 and -6.99e-4 for fresh
R1-only. At central q1 they are +2.63e-4, +3.99e-4 and +3.27e-4.

The respective adversarial/R1/R2 D-gradient norms are .01677/.00356/.00367
under this equal-context diagnostic law. These norms do not demonstrate
penalty starvation. Together, the controls do not support claiming that
zeroing D momentum or removing R2 repairs this task. They also cannot establish
critic lag, capacity failure or the temporal cause of the original error.
The already declared global R1-only comparison must determine its net trained
effect using the original full public protocol. No candidate is selected from
these endpoint controls.

## Reproduce without training

From the repository root, using PyTorch 2.14.0+cu130:

```sh
DDGAN_ARTIFACTS=/mnt/ml7tb/ParticleGAN-forge/r1r2-ddgan-diagnosis-20261022
PROBES=reports/forge/r1r2-local-repair-20261022
for order in 2 3; do
  python "$PROBES/ddgan_critic_field_probe.py" --order "$order" \
    > "$DDGAN_ARTIFACTS/final-critic-field-order$order.json" \
    2> "$DDGAN_ARTIFACTS/final-critic-field-order$order.log"
done
python "$PROBES/summarize_ddgan.py"
tail -F "$DDGAN_ARTIFACTS/final-critic-field-order3.log"
```

The probes verify the pinned original inputs and all 1,287 frozen source files.
They use CPU float64 and one thread, forbid random draws/model or optimizer
construction, and verify unchanged ambient Python/NumPy/Torch RNG states.
Total final quadrature compute was about 5.5 CPU seconds. There were zero new
training updates, continuations, random samples, paid training charges or seed
experiments. Quadrature sensitivity is not independent validation; these
results add no qualification credit and no new leaderboard.
