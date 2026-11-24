<a name="bytegan-stretch"></a>

# Zero-Sigma 256-Byte Discrete Codebook & BiGAN Dictionary Stretch Benchmarks

Two frontier stretch tasks ask whether an unchanged global family recipe trained with an exact-capacity **zero-sigma discrete `ParticlePrior`** (`num_particles = 256`, `z_dim = 8`, `kind = "particle_cloud"`, `sigma = 0.0`, `standardize = false`)—with **no MoG Gaussian kernel smoothing**—can learn an exact 1-to-1 bijection over all 256 `uint8` bytes encoded on the 8-bit binary hypercube `{-1, +1}^8`, both unconditionally and as a bidirectional joint BiGAN dictionary `(G(z), E(x), D(x, z))`.

They belong to `discriminator_stability` revision 20 (`bytegan_hypercube8` at Tier 2; `bytegan_bigan_dictionary` at Tier 4), never block required qualification, and appear in the single [Forge technique inventory](../technique-inventory.md) stretch and disposition tables.

## Protocol, Ownership, and Controls

Both tasks execute from `particlegan.init.deterministic_orthogonal_` under protocol `seed = 0`, isolated named RNG streams, `recipe_owned_v1` prior ownership, 24 scheduled clean/live observations (`scoring_weights = "live"`), and a **5-check terminal sustain gate** (`minimum_stable_checks = 5`). No host-side cluster fitting, particle assignment, weight resets, post-optimizer repairs, per-task recipe overrides, or seed searches are permitted.

---

<a name="bytegan_hypercube8"></a>

## 1. `bytegan_hypercube8` (Tier 2 Stretch, 4,000 Steps)

- **Target & Capacity:** Uniform i.i.d. random `uint8` bytes (`0..255`) encoded on the 8-bit binary hypercube `{-1, +1}^8`, with `M = 256` zero-sigma `ParticlePrior` codes in `z_dim = 8`, unbounded `SimpleMLPGenerator(8, 128, 2, 8)`, and `SimpleMLPDiscriminator(8, 128, 2, fourier=2)` (`batch_size = 128`, `timeout = 600s`).
- **Sustained 5-Check Gates:** `finite_fraction == 1.0`, `valid_particle_fraction == 1.0` (each particle within `bit_tol = 0.25` of its nearest hypercube corner), `valid_unique_bytes == 256` (exact 1-to-1 bijection by pigeonhole), and `rms_bit_error <= 0.12`.

---

<a name="bytegan_bigan_dictionary"></a>

## 2. `bytegan_bigan_dictionary` (Tier 4 Stretch, 6,000 Steps)

- **Target & Capacity:** Uniform i.i.d. random `uint8` bytes (`0..255`) on `{-1, +1}^8` trained as a joint BiGAN `(G(z), E(x), D(x, z))` over `public_components` (`task_cohort = "component_output_transport_v1"`, `transport_consumer = "output_marginal_v1"`) with `M = 256` zero-sigma `ParticlePrior` codes in `z_dim = 8`, `SimpleMLPGenerator` for `G` and `E`, and `SimpleMLPDiscriminator(16, 128, 2, fourier=2)` (`batch_size = 128`, `timeout = 900s`).
- **Sustained 5-Check Gates:** Forward gates `finite_fraction == 1.0`, `valid_particle_fraction == 1.0`, `valid_unique_bytes == 256`, `rms_bit_error <= 0.12`, plus inverse dictionary gates `recon_valid_exact_bytes == 256`, `recon_rms_bit_error <= 0.12`, and `encoder_particle_bijection_bytes == 256` (exact 1-to-1 nearest-particle indexing preventing encoder bypass of the particle cloud).

---

## Diagnostic Status

*Pending matched GPU diagnostic runs on `bytegan_stretch` (`evidence_scope = "research_diagnostic"`).*
