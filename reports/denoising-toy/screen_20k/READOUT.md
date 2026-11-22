# First screening round: established bcap recipe

All **72 training runs completed without failures** on both RTX A6000 GPUs:
48 initial runs in 13.7 minutes, then 24 corrected learned-prior runs in 7.4
minutes. Each has 7000 updates. The final comparison contains 16 configurations
at three matched seeds (24001–24003), using 20,000 latent particles wherever
the latent prior is learned. It reuses the 24 unaffected Gaussian controls.
All 48 included results passed completion/provenance validation; all 32
factor contrasts have three matched seeds. Training sources were unchanged.

The [audit](../BCAP_AUDIT.md) verified the example's exact bcap formulation and
inherited training defaults. The initial 4096-particle table was the sole
material mismatch; corrected configs and outputs have separate paths.

| Configuration | Joint HQ, mean ± seed SD | HQ modes / 100 | Class accuracy |
|---|---:|---:|---:|
| One-shot GAN, concat, Gaussian latent | 25.2 ± 5.6% | 54.0 ± 4.0 | 62.1% |
| One-shot GAN, concat, learned latent | **99.2 ± 0.4%** | **100 ± 0** | **99.7%** |
| One-shot GAN, UCD, learned latent | 96.8 ± 1.3% | 98.3 ± 2.9 | 99.0% |
| Diffusion GAN, UCD, learned latent, Gaussian step noise | 1.0 ± 0.1% | 0.7 ± 1.2 | 33.5% |
| Diffusion GAN, UCD, learned latent, learned step noise | 0.9 ± 0.2% | 0 ± 0 | 32.5% |

Joint HQ means a sample belongs to the requested class and falls within three
standard deviations of a mode. True data scores about 98.9%; exceeding that
does not imply better calibration. HQ coverage also requires enough such
samples per mode, as specified in `grid_metrics`.

The answers for this training budget and target:

- **Denoising objective versus one-shot GAN:** diffusion GAN loses decisively
  on local quality and class fidelity. Across all 12 diffusion configurations,
  joint HQ is only 0.8–1.1%. Generated clouds fill the space between modes.
- **UCD:** no improvement for the strong one-shot particle baseline. It reduces
  mean joint HQ by 2.4 percentage points and worsens conditional mode TV from
  .100 to .174. In diffusion GAN it modestly improves coarse class/distance
  metrics, but does not recover the narrow modes.
- **Learned latent prior:** very helpful for the one-shot GAN under the
  established particle recipe. No comparable rescue for diffusion GAN.
  A fixed latent table control is still needed to separate finite support
  from learning the particles themselves.
- **Learned step-noise particles:** no convincing benefit over fresh Gaussian
  or fixed-table noise. With learned latent particles, replacing Gaussian
  step noise slightly worsens both HQ and conditional SW1 for both D variants.
  This is a negative screen result, not evidence of universal uselessness.

The 20,000-particle correction matters: the concat one-shot model now covers
all 100 modes on every seed, versus 96.7 on average with 4096 particles, and its
mean conditional SW1 improves from .393 to .221.

There is no fully calibrated winner yet. The strong particle GAN's core-width
ratio is .672 (target about 1), conditional mode TV is .100 (real-sample floor
.029), and covariance eigenvalue ratios average 1.48 / 14.26, reflecting tail
and shape errors. Its sharp plot alone is insufficient.

Likewise, conditional SW1 alone gives a misleading ranking: the blurred UCD
diffusion GAN scores .205, slightly below the sharp particle GAN's .221.
The real-sample floor is .070. The toy successfully exposes this disagreement.

Next round should diagnose the denoiser before expanding particle-noise
searches: isolate the final reverse transition, compare four-times-longer
training to account for random selection among four timesteps, and use a
class-free control. These are targeted structural/budget checks while keeping
the audited bcap/optimizer recipe fixed. The final transition's learned-noise
coefficient is zero; changing earlier noise cannot directly fix that last
denoising map. No cause for the failure is established yet, and no new model
has been baked into the example.

[Full table](TABLE.md) · [Samples](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/screen_20k/samples.png) · [Learning curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/screen_20k/curves.png) ·
[Exact configs, timings, paired effects](comparison.json)

The sample panel uses seed 24002, selected before corrected diffusion runs
finished, and displays 4000 samples per cell; reported metrics use 20,000.
All three seeds remain in the table. Raw logs: `results/denoising/screen.log`.
The corrected repeatable manifest is `configs/denoising/screen_20k/manifest.json`.
