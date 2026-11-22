# DDGAN diagnostic round (seed 24002)

Nine new runs completed on both RTX A6000 GPUs in 8.7 minutes, with zero
failures. Every result passed the runner certificate check. Training code,
loss, bcap, optimizer, architecture, and particle regularization are unchanged
from the first screen. All learned latent tables contain 20,000 particles.
These are exploratory single-seed results, not confirmed rankings.

| Intervention | Joint HQ | HQ modes | Conditional mode TV |
|---|---:|---:|---:|
| Four-step concat, Gaussian latent, 28k updates | 72.1% | 100 | .135 |
| Four-step concat, learned latent, 28k updates | 71.2% | 100 | .170 |
| Four-step UCD, learned latent, 28k updates | 78.1% | 100 | .134 |
| One-step concat, learned latent, 7k updates | 99.2% | 100 | .116 |
| Two-step concat, learned latent, 14k updates | 43.0% | 77 | .316 |
| Class-free four-step concat, learned latent, 28k | 10.5% | 51 | .141 |
| Class-free one-shot learned-particle GAN, 7k | 99.2% | 100 | .112 |
| Four-class one-shot fixed-particle GAN, 7k | 19.3% | 46 | .506 |
| Four-class one-shot learned-particle GAN, 28k | 99.5% | 100 | .104 |

Longer training substantially rescues the multi-step objective. This changes
both the number of updates and the duration of the high learning-rate phase:
cosine decay still starts at 60% of the total run. It does not isolate a pure
per-timestep sample-count effect. The 1/2/4-step runs receive approximately
7000 updates per timestep on average, but use different schedules and amounts
of compute. Equal-update one-shot controls are also included.

The one-step DDGAN has an almost independent Gaussian observation and no
posterior step noise. Its success checks the single-transition implementation;
it is close to a one-shot GAN with extra random input and does not demonstrate
an advantage from iterative denoising. The two-step schedule uses alpha_bar
[1, .5, .0001], versus [1, .9, .5, .05, .0001] for four steps.

Removing class conditioning does not fix DDGAN. It also makes each conditional
distribution denser: the nearest same-class grid spacing changes from 2 to 1.
Thus the class-free control changes the denoising task as well as its inputs.

The fixed latent table does not reproduce the learned table's one-shot benefit
in this seed. A finite table alone is insufficient here; repeated seeds are
still needed before making a general claim about learning versus fixed support.

## Exact-posterior chain substitutions

These use analytic knowledge of the target and are diagnostics, never candidate
samplers. The all-oracle chain gives 98.8% HQ / .027 conditional TV, validating
the oracle sampler at the marginal level.

| Learned UCD checkpoint | All model HQ | Only final step learned HQ | Only final step oracle HQ |
|---|---:|---:|---:|
| 7k updates | .93% | 4.38% | 98.81% |
| 28k updates | 78.06% | 92.43% | 98.83% |

At 7k the learned last step fails even on exact earlier transitions. At 28k it
has improved dramatically; the gap between 92.4% with oracle earlier steps and
78.1% on its own chain also implicates accumulated errors / changed inputs.
The last-step oracle can correct a poor input distribution, so its high HQ is
not evidence that earlier learned transitions are exact. Mode TV after this
substitution is .064 at 7k and .075 at 28k, versus the .027 oracle-chain floor.

[Samples](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/diagnostics/samples.png), [full metrics](TABLE.md), [learning curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/diagnostics/curves.png),
[7k chain probe](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/diagnostics/old_chain/chain_probe.png),
[28k chain probe](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/diagnostics/ucd_28k_chain/chain_probe.png).

Configs: `configs/denoising/diagnostics/manifest.json`.
Shared progress log: `results/denoising/diagnostics.log`.
Next: the separately configured 56k budget/step-noise screen; no default chosen.
