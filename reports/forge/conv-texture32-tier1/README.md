<a name="img_conv_texture32"></a>

# Tier 1 High-Dimensional Texture Acquisition Gate (`img_conv_texture32`)

`img_conv_texture32` is a fast **required Tier 1** task in [`discriminator_stability` revision 18](../../../configs/forge/views/discriminator_stability.json) (`288` updates, `300s` timeout ceiling; ~`45s` on GPU for feature-critic recipes) designed to expose high-dimensional ambient concentration-of-measure and feature-critic vs raw-pixel-OT disconnects before Tier 2–3 or external 32×32 image runs.

## Scientific Question & Concentration Geometry

In ambient dimension $d = 3 \times 32 \times 32 = 3072$ at minibatch size $B = 64$, real samples drawn from two equiprobable 12-mode random-phase oriented RGB texture bands ($\phi_1, \dots, \phi_{12} \sim \operatorname{Uniform}([0, 2\pi))$, `BAND_RMS_AMP = 0.68`, `NOISE_STD = 0.02`, clamped to $[-1, 1]$) concentrate on a high-dimensional shell:

- **Real-to-real 25th-percentile nearest-neighbor horizon:** $\ell_t = Q_{0.25}(d^{\text{NN}}_Y) \approx 16.06$
- **Interior zero-carrier DC mean-blob distance to the real shell:** $d^{\text{NN}}(\bar{x}_c, Y) \approx 15.59 < \ell_t$
- **Dimensionless raw-pixel Born-Infeld cost (`rhbi`):** median $\hat{c}_{ij} \approx 0.43 \ll \rho = 3.0$

Any recipe that measures transport geometry solely in raw ambient coordinates (`args[0]`), assigns zero critic gradient to active Sinkhorn pairs (`rhbi_relativistic_weight = 0.0`), and gates adversarial escape slack exponentially behind $-\frac{1}{2}\rho + \frac{1}{2}\tau \tilde{d}_{\text{fake}}$ when $\tilde{d}_{\text{fake}} < 0$ becomes trapped inside the interior pixel-OT ball, collapsing onto the two flat class DC means with near-zero spatial carrier energy.

## Protocol, Ownership, and Semantic Controls

- **Declaration:** [img_conv_texture32.json](../../../configs/forge/tasks/img_conv_texture32.json) (`adapter = "conv_texture32"`, [api_conv_texture32.py](../../../benchmarks/toy_audit/api_conv_texture32.py), [conv_texture32.py](../../../experiments/forge/conv_texture32.py)).
- **Prior & Initialization:** Recipe-owned `MoGParticlePrior(num_particles=64, z_dim=8, sigma=0.1, standardize=False)` (`prior_contract = "recipe_owned_v1"`), `particlegan.init.deterministic_orthogonal_`, protocol `seed = 0`, `batch_size = 64`, `steps = 288`.
- **Architecture:** `ResidualUpConvGenerator32(z_dim=8, width=16)` (`4->8->16->32` nearest-upsample residual conv blocks + linear DC shortcut + `tanh`) and `MiniResNetCritic32(width=16, mlp_hidden=64)` (strided residual feature pyramid + `[h_mean, h_rms, dc, ac_rms]` pooling + 3-layer non-inplace `LeakyReLU(0.2)` MLP head supporting second-order gradient penalties across all families).
- **Evaluation & Sustained 5-Check Gates (`N = 256` clean live prior draws at steps `12, 24, ..., 288`):**
  1. `finite_fraction == 1.0`
  2. `hf_band_ratio >= 0.55` *(oracle worst of 200 draws: `>= 0.95`; zero-carrier mean-blob trap: `0.000`)*
  3. `clean_texture_fraction >= 0.75` *(oracle worst of 200 draws: `>= 0.95`; 7×7 box-blur damage: `0.000`)*
  4. `orientations == 2` *(single-orientation collapse: `1`)*
  5. `mass_tv <= 0.18` *(oracle worst of 200 draws: `<= 0.11`; single-orientation collapse: `0.50`)*
  6. `pairwise_phase_diversity >= 0.40` *(oracle worst of 200 draws: `>= 0.85`; fixed-phase 2-template memorization: `0.074`)*

## Matched CUDA Readout (`seed = 0`, `NVIDIA RTX A6000`, Source `6899f3834`)

Standalone matched execution through Forge `run_task` + `grade_result` ([summary.json](summary.json), [publication.json](../../toy_audit/api_contract/conv_texture32/publication.json); `qualification_input = false`, preserving all archived family cohorts in [technique-inventory.md](../technique-inventory.md) without regrade):

| Family | Candidate | Verdict | Terminal Checks | `hf_band_ratio` (`>=0.55`) | `clean_texture_fraction` (`>=0.75`) | `orientations` (`==2`) | `mass_tv` (`<=0.18`) | `pairwise_phase_diversity` (`>=0.40`) | Loop Wall (s) | Actual Training GIF |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **EMPF** | `empf-v8` (`empf`) | **PASS** | **5 / 5** | `0.8177` | `1.0000` | `2` | `0.0781` | `0.7515` | `45.4s` | [empf-v8.gif](media/empf-v8.gif) |
| **BISPF** | `bispf-throughput-v1` (`bispf_optimized`) | **PASS** | **5 / 5** | `0.8256` | `0.9922` | `2` | `0.1250` | `0.6741` | `46.4s` | [bispf-throughput-v1.gif](media/bispf-throughput-v1.gif) |
| **RHBI** | `rhbi-v12` (`rhbi_optimized`) | **FAIL** | **0 / 5** | **`0.0086`** | **`0.0000`** | **`0`** | `0.0039` | **`0.0000`** | `106.0s` | [rhbi-v12.gif](media/rhbi-v12.gif) / [api-img-conv-texture32.gif](../../toy_audit/api_contract/conv_texture32/media/api-img-conv-texture32.gif) |

## Key Observations & Next Steps

1. **Exact reproduction of the 32×32 CIFAR-10 failure in 288 updates:** `rhbi-v12` matches the two flat DC class color means (`mass_tv = 0.0039`, `dc_mean_err = 0.300`) while producing **zero** high-frequency texture (`hf_band_ratio = 0.0086`, `clean_texture_fraction = 0.0000`, `orientations = 0`, `pairwise_phase_diversity = 0.0000`), failing every one of the 24 observations (`0/5` terminal checks).
2. **Feature-critic baselines pass cleanly in ~46s:** Under the identical `batch_sequence_sha256` (`eac470a9c7d6...`) and `initial_parameters` digests, both `empf-v8` (`clean_texture_fraction = 1.000`, `5/5` terminal checks in `45.4s`) and `bispf-throughput-v1` (`clean_texture_fraction = 0.992`, `5/5` terminal checks in `46.4s`) acquire crisp, phase-diverse horizontal and vertical RGB textures well within Tier 1 speed budgets.
3. **Next Step (Separate Bounded Study):** Evaluate generic, modality-agnostic RHBI repairs—critic penultimate representation geometry vs raw input coordinates, sign-faithful adversarial slack escape when $\tilde{d}_{\text{fake}} < 0$, and nonzero active-pair critic coupling—in a dedicated bounded Forge study without task-shape conditionals.
