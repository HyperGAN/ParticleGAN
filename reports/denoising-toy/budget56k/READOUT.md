# Round two: DDGAN recovers with more training

All **17 new training runs completed successfully on both RTX A6000 GPUs**.
Nine diagnostics took 8.7 minutes; six 56k-update runs took 17.8 minutes; two
additional 28k seed checks took 4.9 minutes, overlapping the latter grid.
There are no active experiments. Every run passed completion/provenance checks.
The exact training source fingerprint matches round one: no loss, bcap,
architecture, optimizer, or particle-regularization changes. All latent tables
use 20,000 particles. No model default has been changed.

## What changed our understanding

The original 7k screen was too short for this four-step model. The UCD DDGAN
with learned latent particles and Gaussian step noise rises from about 1% joint
HQ at 7k to **77.8 ± 1.6% at 28k across three matched training seeds**, covering
all 100 modes in every run. ± is sample standard deviation. The 56k comparison
below is exploratory at seed 24002 and still needs repeated training seeds.

Extending training also extends the high-learning-rate phase: decay always
starts at 60% of the total budget. This is a budget/schedule intervention, not
an isolated test of timestep exposure. The one-shot comparator receives the
same number of optimizer updates, not exactly matched FLOPs or elapsed time.

## Equal-update screen: 56,000 updates

| Model / D | Latent prior | Step noise | Joint HQ | Conditional mode TV ↓ | Conditional SW1 ↓ |
|---|---|---|---:|---:|---:|
| One-shot GAN / concat | Learned | N/A | **99.8%** | .103 | .220 |
| DDGAN / concat | Gaussian | Gaussian | 89.1% | .065 | .114 |
| DDGAN / concat | Learned | Gaussian | 88.9% | .072 | **.082** |
| DDGAN / UCD | Learned | Gaussian | 91.2% | **.059** | .104 |
| DDGAN / UCD | Learned | Fixed particles | 90.9% | .065 | .136 |
| DDGAN / UCD | Learned | Learned particles | **93.4%** | .070 | .140 |

All six recover all 100 HQ modes. Joint HQ requires correct class and distance
less than 3σ from a mode. The real-data reference is approximately 98.9% HQ,
.029 mode TV, and .070 conditional SW1. Higher HQ than real data is possible
through excessive concentration and is not evidence of superior calibration.

- **Diffusion objective:** now competitive on mode proportions and distance,
  but worse on local fidelity and class accuracy. It is not a clear overall
  winner. The one-shot model has almost no far-out tails but under-dispersed
  cores (width ratio .640). Gaussian-noise UCD DDGAN has core ratio .512 and
  3.4% of samples farther than 10σ from the nearest mode. Both need calibration.
- **UCD:** modestly better HQ and mode TV than concat with the same learned
  prior, but worse conditional SW1 in this seed. This is an exploratory effect.
- **Latent particles:** no clear DDGAN win over Gaussian latent noise. Under
  concat, HQ is essentially tied; learned particles improve SW1 but worsen TV.
  The separate one-shot fixed-table control gets only 19.3% HQ at 7k, so a
  finite table alone does not reproduce the learned-particle recipe's benefit.
- **Step-noise particles:** learning improves HQ by 2.2 percentage points over
  Gaussian noise, while worsening TV/SW1 and narrowing cores further to .439.
  Fixed particles do not help. Particle-specific benefits remain unproven.

## Is learned step noise mostly learning its covariance?

The learned 1024-row noise table has covariance eigenvalues .250 and .354,
versus 1 and 1 for fresh Gaussian noise. It is shared across reverse steps;
forward corruption and terminal noise remain Gaussian. The final reverse step
has zero posterior-noise multiplier.

We froze the learned-noise checkpoint and changed only its inference noise.
Three fresh evaluation draws of 20,000 samples each give:

| Inference substitution | Mean joint HQ | Mean mode TV | Mean conditional SW1 |
|---|---:|---:|---:|
| Learned table | 93.48% | .0683 | .1261 |
| Gaussian with learned mean/covariance | 93.22% | .0720 | .1324 |
| Initial fixed table, matched mean/covariance | 93.17% | .0703 | .1266 |
| Standard Gaussian | 90.33% | .1054 | .1966 |
| Zero step noise | 93.36% | .0727 | .1750 |

Covariance-matched replacements preserve most of the benefit for this frozen
model. This motivates a simpler learnable Gaussian covariance control; it does
not prove particles were unnecessary during training. These are three evaluation
seeds of ONE trained checkpoint, not three training replicates. Zero noise also
looks good on HQ while degrading SW1, reinforcing the need for multiple metrics.
See [raw substitutions](noise_probe_summary.json).

## Where the remaining denoising error sits

Exact-posterior chain substitutions diagnose the learned UCD / Gaussian-noise
model, without using oracle information during training:

| Training budget | Full learned chain HQ | Only final step learned HQ | Only final step oracle HQ |
|---|---:|---:|---:|
| 7k | .93% | 4.38% | 98.81% |
| 28k | 78.06% | 92.43% | 98.83% |
| 56k | 91.25% | 96.43% | 98.88% |

The last step improves substantially with training but still fails more often
on its own chain's inputs. Earlier-step error and final-step accuracy both
matter. Oracle substitutions are diagnostics, not deployable contenders. A final
oracle can repair bad inputs, so high HQ after that substitution does not establish
that earlier transitions are exact. See [56k chain probe](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/budget56k/ucd_56k_chain/chain_probe.png).

The one-step DDGAN reaches 99.2% HQ at 7k, but its almost independent Gaussian
observation makes it close to a one-shot GAN with extra randomness. The two-step
14k run reaches 43.0%; the class-free four-step 28k run only 10.5%. Neither
removing classes nor reducing the number of steps automatically fixes the problem.
The class-free task also has denser conditional modes, so it is not solely an
input-encoding intervention. See [diagnostics](../diagnostics/READOUT.md).

## Next round

1. Repeat the promising 56k comparisons on additional training seeds before
   treating UCD or learned noise as improvements.
2. Compare learned Gaussian noise scale/covariance against learned noise particles
   during training. This is a focused alternative suggested by the substitution
   results, not a broad hyperparameter search.
3. Target final-step accuracy / robustness to earlier-step errors while retaining
   the established bcap recipe. Prioritize a better quality/compute tradeoff over
   simply extending every run again. Test timestep allocation or denoiser
   parameterization explicitly; no structural cause has yet been isolated.
4. Before baking in a winner, confirm on fresh seeds and a nonuniform or anisotropic
   target. The present sharpness/mass tradeoff does not select a universal winner.

[Sample comparison](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/budget56k/samples.png) · [Full table](TABLE.md) ·
[Learning curves](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/denoising-toy/budget56k/curves.png) · [Three-seed 28k check](../confirm28k/TABLE.md).

Repeatable configs: `configs/denoising/budget56k/manifest.json`,
`configs/denoising/diagnostics/manifest.json`, and
`configs/denoising/confirm28k/manifest.json`. The latter has two new runs;
`comparison_manifest.json` includes the third run from diagnostics.
Progress log: `results/denoising/diagnostics.log`. Final EMA evaluation checkpoints,
exact training-source archives, metrics, and samples are saved per run. Checkpoints
do not include optimizer-resume state. Plots now correctly label one-shot step
noise N/A. Analysis/plot/diagnostic scripts compile; training sources were untouched.
