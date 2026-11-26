<a name="bytegan-stretch"></a>

# Zero-Sigma 256-Byte Discrete Codebook & BiGAN Dictionary Stretch Benchmarks

Two frontier stretch tasks ask whether an unchanged global family recipe trained with an exact-capacity **zero-sigma discrete `ParticlePrior`** (`num_particles = 256`, `z_dim = 8`, `kind = "particle_cloud"`, `sigma = 0.0`, `standardize = false`)—with **no MoG Gaussian kernel smoothing**—can learn an exact 1-to-1 bijection over all 256 `uint8` bytes encoded on the 8-bit binary hypercube `{-1, +1}^8`, both unconditionally and as a bidirectional joint BiGAN dictionary `(G(z), E(x), D(x, z))`.

They belong to `discriminator_stability` revision 20 (`bytegan_hypercube8` at Tier 2; `bytegan_bigan_dictionary` at Tier 4), never block required qualification, and appear in the single [Forge technique inventory](../technique-inventory.md) stretch and disposition tables. Full matched CUDA diagnostics across five unchanged global family recipes (`seed = 0`, `NVIDIA RTX A6000`, `evidence_scope = "research_diagnostic"`) are recorded in [summary.json](summary.json) and the supplemental [public-API publication](../../toy_audit/api_contract/bytegan/publication.json).

## Protocol, Ownership, and Controls

Both tasks execute from `particlegan.init.deterministic_orthogonal_` under protocol `seed = 0`, isolated named RNG streams (`init:*`, `data:target:training`, `prior:latent`, `noise:*`, `eval:*`), `recipe_owned_v1` prior ownership, 24 scheduled clean/live observations (`scoring_weights = "live"`), and a **5-check terminal sustain gate** (`minimum_stable_checks = 5`). No host-side cluster fitting, particle assignment, weight resets, post-optimizer repairs, per-task recipe overrides, or seed searches are permitted.

Twelve analytic positive and destructive negative oracle controls (`tests/test_forge_bytegan.py`) verify scorer selectivity against exact bijections, single-byte collisions, faint barycentric ghosts (`0.05 * corner`), exploded amplitudes (`5.0 * corner`), half-cube collapse, swapped canonical reconstructions, encoder cloud bypass (`E(x)` collapsed onto a single particle while `G(E(x)) == x`), blurry reconstructions, and nonfinite outputs.

---

<a name="bytegan_hypercube8"></a>

## 1. `bytegan_hypercube8` (Tier 2 Stretch, 4,000 Steps)

- **Target & Capacity:** Uniform i.i.d. random `uint8` bytes (`0..255`) encoded on the 8-bit binary hypercube `{-1, +1}^8`, with `M = 256` zero-sigma `ParticlePrior` codes in `z_dim = 8`, unbounded `SimpleMLPGenerator(8, 128, 2, 8)`, and `SimpleMLPDiscriminator(8, 128, 2, fourier=2)` (`batch_size = 128`, `timeout = 600s`).
- **Sustained 5-Check Gates:** `finite_fraction == 1.0`, `valid_particle_fraction == 1.0` (each particle within `bit_tol = 0.25` of its nearest hypercube corner), `valid_unique_bytes == 256` (exact 1-to-1 bijection by pigeonhole), and `rms_bit_error <= 0.12`.
- **Actual Training Media (`unified-v12`):** [api-bytegan-hypercube8.gif](../../toy_audit/api_contract/bytegan/media/api-bytegan-hypercube8.gif).

---

<a name="bytegan_bigan_dictionary"></a>

## 2. `bytegan_bigan_dictionary` (Tier 4 Stretch, 6,000 Steps)

- **Target & Capacity:** Uniform i.i.d. random `uint8` bytes (`0..255`) on `{-1, +1}^8` trained as a joint BiGAN `(G(z), E(x), D(x, z))` over `public_components` (`task_cohort = "component_output_transport_v1"`, `transport_consumer = "output_marginal_v1"`) with `M = 256` zero-sigma `ParticlePrior` codes in `z_dim = 8`, `SimpleMLPGenerator` for `G` and `E`, and `SimpleMLPDiscriminator(16, 128, 2, fourier=2)` (`batch_size = 128`, `timeout = 900s`).
- **Sustained 5-Check Gates:** Forward gates `finite_fraction == 1.0`, `valid_particle_fraction == 1.0`, `valid_unique_bytes == 256`, `rms_bit_error <= 0.12`, plus inverse dictionary gates `recon_valid_exact_bytes == 256`, `recon_rms_bit_error <= 0.12`, and `encoder_particle_bijection_bytes == 256` (exact 1-to-1 nearest-particle indexing preventing encoder bypass of the particle cloud).
- **Actual Training Media (`unified-v12`):** [api-bytegan-bigan-dictionary.gif](../../toy_audit/api_contract/bytegan/media/api-bytegan-bigan-dictionary.gif).

---

## Multi-Family CUDA Diagnostic Results (`seed = 0`, `NVIDIA RTX A6000`, Source `ab0da6c53`)

Diagnostic campaign `bytegan-stretch-diag-v1` (`view = bytegan_stretch`, `evidence_scope = "research_diagnostic"`, `qualification_input = false`). All ten runs completed the full declared update budget (`4,000` / `6,000` steps) with `all_finite = true`, `hooks_exercised = true`, `unintended_rng_deviations = 0`, and exact optimizer update counts across every role. Every recorded ordinary qualification verdict remains unchanged. (`unified-v12` is the current default-recipe frontier candidate; `rhbi-v12`, `bispf-optimized-quintic-v1`, `empf-v8`, and `recipe-prior-refactor-bcap-pure-v1` are their families' current selected candidates in `configs/forge/selections/family-current-v1.json`.)

### Task 1: `bytegan_hypercube8` (4,000 Steps, Unconditional 256-Byte Codebook)

| Family | Candidate | Verdict (Term) | `valid_unique_bytes` Final (Peak @ Step) | `unique_bytes` Final (Peak @ Step) | `collision_count` Final (Min) | `valid_particle_fraction` Final (Peak) | `rms_bit_error` Final (Min) | `mass_tv` Final (Min) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| **Unified** | `unified-v12` | **FAIL** *(0/5)* | **139 / 256** *(139 @ 334)* | **154 / 256** *(154 @ 334)* | **102** *(102)* | `0.8555` *(0.8555)* | `0.0875` *(0.0875)* | **0.3848** *(0.3691)* | 186.2 |
| **RHBI** | `rhbi-v12` | **FAIL** *(0/5)* | 123 / 256 *(123 @ 3500)* | 125 / 256 *(126 @ 1667)* | 131 *(130)* | `0.9727` *(0.9727)* | `0.1041` *(0.0939)* | 0.5059 *(0.4785)* | 222.3 |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** *(0/5)* | 55 / 256 *(55 @ 2834)* | 61 / 256 *(61 @ 4000)* | 195 *(195)* | `0.9531` *(0.9648)* | `0.1233` *(0.1233)* | 0.7617 *(0.5469)* | 198.6 |
| **BCAP-Pure** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** *(0/5)* | 43 / 256 *(43 @ 2167)* | 43 / 256 *(43 @ 2000)* | 213 *(213)* | **`1.0000`** *(1.0000)* | **`0.0341`** *(0.0324)* | 0.8320 *(0.7461)* | 168.7 |
| **EMPF** | `empf-v8` | **FAIL** *(0/5)* | 38 / 256 *(38 @ 3500)* | 40 / 256 *(40 @ 2834)* | 216 *(216)* | `0.9844` *(0.9844)* | `0.1116` *(0.1116)* | 0.8438 *(0.8438)* | 154.6 |

### Task 2: `bytegan_bigan_dictionary` (6,000 Steps, Bidirectional Joint BiGAN Dictionary)

| Family | Candidate | Verdict (Term) | `valid_unique_bytes` Final (Peak @ Step) | `unique_bytes` Final (Peak @ Step) | `collision_count` Final (Min) | `valid_particle_fraction` Final (Peak) | `rms_bit_error` Final (Min) | `mass_tv` Final (Min) | `recon_valid_exact_bytes` Final (Peak @ Step) | `recon_rms_bit_error` Final (Min) | `encoder_particle_bijection_bytes` Final (Peak @ Step) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| **BCAP-Pure** | `recipe-prior-refactor-bcap-pure-v1` | **FAIL** *(0/5)* | **175 / 256** *(180 @ 4250)* | 197 / 256 *(198 @ 5500)* | 59 *(58)* | `0.8398` *(0.8984)* | `0.1942` *(0.1485)* | **0.2363** *(0.2363)* | 1 / 256 *(2 @ 2750)* | `0.7370` *(0.6685)* | **183 / 256** *(186 @ 4750)* | 197.4 |
| **Unified** | `unified-v12` | **FAIL** *(0/5)* | 170 / 256 *(181 @ 5750)* | 189 / 256 *(197 @ 5750)* | 67 *(59)* | `0.8516` *(0.9180)* | **`0.1505`** *(0.1357)* | 0.2617 *(0.2344)* | 0 / 256 *(2 @ 2250)* | `0.8635` *(0.7003)* | 154 / 256 *(169 @ 5500)* | 133.5 |
| **RHBI** | `rhbi-v12` | **FAIL** *(0/5)* | 166 / 256 *(**183** @ 5000)* | **203 / 256** *(**209** @ 5250)* | **53** *(**47**)* | `0.7734` *(0.8789)* | `0.1814` *(**0.1327**)* | 0.2383 *(**0.1973**)* | 1 / 256 *(4 @ 3500)* | **`0.6967`** *(**0.5860**)* | 170 / 256 *(186 @ 5250)* | 171.8 |
| **EMPF** | `empf-v8` | **FAIL** *(0/5)* | 106 / 256 *(107 @ 5750)* | 134 / 256 *(134 @ 6000)* | 122 *(122)* | `0.7422` *(0.8398)* | `0.1994` *(0.1777)* | 0.4570 *(0.4492)* | 0 / 256 *(1 @ 750)* | `1.2643` *(1.1250)* | 160 / 256 *(160 @ 6000)* | 271.1 |
| **BISPF** | `bispf-optimized-quintic-v1` | **FAIL** *(0/5)* | 59 / 256 *(60 @ 5000)* | 59 / 256 *(60 @ 2750)* | 197 *(196)* | **`0.9141`** *(**0.9219**)* | `0.1697` *(0.1684)* | 0.7266 *(0.4961)* | **2 / 256** *(**7** @ 4000)* | `1.6147` *(1.3457)* | 147 / 256 *(150 @ 4000)* | 352.3 |

*Additional BiGAN inverse diagnostics at step 6,000 (peak):* `recipe-prior-refactor-bcap-pure-v1` reached `encoder_cycle_match_bytes = 160/256` (peak `163`), `recon_sign_exact_bytes = 135/256` (peak `150`), `encoder_nearest_dist_mean = 2.130`; `rhbi-v12` reached `encoder_cycle_match_bytes = 119/256` (peak `154`), `recon_sign_exact_bytes = 104/256` (peak `164`), `encoder_nearest_dist_mean = 1.969`; `unified-v12` reached `encoder_cycle_match_bytes = 91/256` (peak `104`), `recon_sign_exact_bytes = 80/256` (peak `110`), `encoder_nearest_dist_mean = 2.072`.

---

## Scientific Interpretation & Next-Step Recommendations

1. **ByteGAN is a genuine 0/10 frontier stretch benchmark with clean family separation.** Every tested global family passes all numerical sanity, finiteness, and bit-amplitude bounds cleanly yet fails exact 256/256 1-to-1 bijection (`0/5` terminal passes across all 10 runs). Bit crispness itself is easy—`recipe-prior-refactor-bcap-pure-v1` reaches `valid_particle_fraction = 1.000`, `rms_bit_error = 0.0341`, and `max_bit_error = 0.1035` on `bytegan_hypercube8`, and `unified-v12`, `rhbi-v12`, and `empf-v8` all pass `rms_bit_error <= 0.12`—proving the bottleneck is **exact combinatorial collision-breaking across 256 equidistant discrete corners**, not continuous bit regression.
2. **Unconditional `bytegan_hypercube8` hits a sub-coupon-collector lock-in ceiling (`38–154 / 256` raw bytes).** Independent random placement of 256 particles onto 256 corners yields an expected `256 * (1 - (255/256)^256) ≈ 162.0` unique bytes (`94` collisions). Every unconditional run stalls strictly below this ceiling:
   - *Premature thermal freezing (`unified-v12`, `rhbi-v12`):* `unified-v12` attains the highest unconditional coverage (`139/256` valid, `154/256` raw, `mass_tv = 0.3848`) by step `334`, then locks completely for the remaining `3,666` steps (`valid_unique_bytes = 139`, `rms_bit_error = 0.0875` unchanged from step 334 to 4000). `rhbi-v12` similarly freezes on `125/256` raw bytes by step `500` and spends steps `500–4000` sharpening those same `125` corners (`valid_particle_fraction = 0.9727`, `131` collisions). Once particles sit cleanly on valid corners, local continuous critic/transport residuals vanish even though half the hypercube is orphaned.
   - *Heavy many-to-one collapse (`bispf`, `bcap-pure`, `empf-v8`):* Without strong discrete self-repulsion in 8D, `bispf` (`55/256`), `bcap-pure` (`43/256`), and `empf-v8` (`38/256`) pile `4–7` crisp particles onto each occupied corner (`195–216` collisions).
3. **Joint BiGAN pairing breaks the 162-byte unconditional ceiling on forward generation across three families—yet fails exact inverse reconstruction (`0–2 / 256` valid).** Every single family improves forward unique byte coverage on `bytegan_bigan_dictionary` relative to `bytegan_hypercube8` (`bcap-pure`: `43 -> 175` valid / `197` raw; `unified-v12`: `139 -> 170` valid, peak `181` / `197` raw; `rhbi-v12`: `123 -> 166` valid, peak `183` / `209` raw; `empf-v8`: `38 -> 106` valid / `134` raw; `bispf`: `55 -> 59`). Because the joint critic `D(x, E(x))` vs `D(G(z), z)` receives real uncovered bytes `x` paired with `E(x)`, it sustains pressure against orphaned corners throughout training. However, two gaps block BiGAN PASS:
   - *Residual collisions (`47–67` minimum collisions):* Even the strongest runs (`rhbi-v12` peak `209` raw / `183` valid; `bcap-pure` `197` raw / `175` valid; `unified-v12` peak `197` raw / `181` valid) leave `~50–80` orphaned corners.
   - *Encoder-to-particle landing gap (`encoder_nearest_dist_mean = 1.97–3.79`, `recon_rms_bit_error = 0.70–1.61`):* `bcap-pure` achieves `183/256` distinct nearest particles and `160/256` nearest-particle cycle matches (with `135/256` sign-exact `G(E(x))`), yet only `1/256` bytes land within `bit_tol = 0.25`. Because all five unchanged global recipes leave `joint_reconstruction_weight = 0.0`, pure adversarial joint matching aligns `E(x)` to the *Voronoi neighborhood* of `z_i` without contracting `||E(x) - z_i||` or `||G(E(x)) - x||` onto the discrete corner.
4. **Recommended recipe-owned hypotheses (without task-side overrides or hacks):**
   - **Collision-aware thermal reopen & discrete self-repulsion:** Keep the recipe's thermal cooling / prototype-lock controller open whenever colliding particles map within a tight output-space neighborhood (`||G(z_i) - G(z_j)|| << median NN distance`) and apply a recipe-owned output/critic-space duplicate-particle repulsion or balanced 1-to-1 minibatch assignment term to break the 2.0-Euclidean hypercube corner barrier.
   - **Recipe-owned BiGAN cycle & latent-codebook contraction:** Evaluate enabling a nonzero global `joint_reconstruction_weight` (plus recipe-owned latent prototype-pullback contraction on `E(x)` toward its matched particle code) in a declared formulation study so `G(E(x))` and `E(x)` sharpen from `~135–164` sign-exact Voronoi matches to exact `256/256` crisp dictionary lookup without harming ordinary Tier 1–4 tasks.
