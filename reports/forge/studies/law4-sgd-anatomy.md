# Law-4 Causal SGD & Asymmetric G/D Anatomy Study (`research/law4-sgd-anatomy`)

**Branch:** `research/law4-sgd-anatomy` (branched off `develop` `87d72c39b`)  
**Package:** [`particlegan/_law4_sgd/`](../../../particlegan/_law4_sgd/) (`__init__.py`, `optimizer.py`), registered in `particlegan/optim/dualnorm.py`, `particlegan/recipes.py`, `experiments/forge/techniques.py`, and `experiments/forge/clockfree.py` — **zero edits** to `particlegan/_unified_v11/` or any pre-existing `reports/forge/records/readout-*.json`.  
**Protocol & Controls:** Protocol seed `0`, `deterministic_orthogonal` initializer, clean live sampling, `NVIDIA RTX A6000`, `lr = 0.014, d_lr_mult = 1.5, prior_lr_mult = 2.9` **held 100% identical to `unified-v11`** across all 6 arms (empirically verified as the literal optimum for pure SGD on `ring16` across `[0.002, 0.1]`), holding Laws 1–3 (`critic_formulation="unified_v11"`, `prior_regularizer="unified_v11"`, `UnifiedV11StepRecord`, `UnifiedV11CondFiberBank`, zero-`pinv` `direction_blend`) strictly identical.  
**Unit & Semantic Controls ([`tests/test_law4_sgd.py`](../../../tests/test_law4_sgd.py)):** `16/16` PASS — exact unnormalized SGD semantic control (`w_new == w_0 - lr * g`), sparse sampled-row prior masking, asymmetric G/D controls, exact Frobenius-MP formula & `1e-36` underflow safety, `0` tensor states/param on pure/secant/Frob-SGD, monkeypatch-enforced `0` SVD/`eigh`/`pinv`/`autograd.grad` (and `0` Quintic NS on pure/secant/Frob-SGD), tamper-rejecting checkpoint round-trips, and bit-exact fresh-process continuation.  
**Governance Scope:** All 6 Forge readout receipts carry `evidence_scope: "research_diagnostic"` on `r4-occam-ablation-v1` (no ordinary qualification or default-adoption credit; `DEFAULT_RECIPE` remains `"unified_v11"`).

---

## Executive Answers

### 1. "Why can't pure SGD work?"
Pure unnormalized SGD (`law4-pure-sgd`, `readout-62543241e3cb64f098759ad7.json`) scores **`2/13` executed PASS** (`9` numerical FAIL, `2` NaN-explosion `INCOMPLETE` via `FloatingPointError`, `3` prereq-gated) vs `unified-v11` **`16/16` PASS** — even at an optimal global LR (`lr = 0.014`) where pure SGD **does** pass easy 2D vectors (`vector_unequal_width` `hq=0.9990`, `vector_unequal_mass` `hq=0.9963`). Our 5-rung ladder causally decomposes pure SGD's failure into **four distinct, separable mathematical obstructions**:

1. **Player-Level Impedance Explosion (`>500×` Nash imbalance):** Pure SGD explodes to `NaN` (`FloatingPointError: nonfinite public training loss`) within `< 2s` on **both** `nash_overpowered_critic_ring16` (`620×` $D \gg G$) and `nash_overpowered_generator_ring16` (`548×` $G \gg D$). Wrapping pure SGD with **ONLY Law 4's global player-level secant trust band** (`law4-sgd-secant`, `readout-d5c8d860e8107c9cc66729c6.json`, 0 per-layer norm, 0 MP, 0 NS, 0 states) rescues **both** Nash tasks to full 5/5 sustained PASS (`hq = 0.9768` / `0.9751`) plus `ring16` (`0.9841`) — proving player-level secant impedance matching is **necessary and sufficient** for `>500×` player capacity imbalance, yet leaves `9/13` tasks failing.
2. **Critic Starvation Under Law-2 Born–Infeld Potentials:** Because Law 2 bounds $\|\nabla_x D\|_2 \le 1$, raw critic parameter gradients are `~30×` smaller than generator gradients. In our 80-step telemetry ([`law4-sgd-anatomy-telemetry.json`](law4-sgd-anatomy-telemetry.json)), pure SGD starves critic function-space displacement by **`31×–62×`** (`dD = 0.0043` vs `unified-v11` `0.135` on `ring16`; `D/G = 0.07–0.24` vs `1.98–4.31`).
3. **Intra-Player Inter-Layer Scale Skew & `sech^2` Saturation Deadlock:** A single player-wide scalar cannot equalize `7,500×` inter-layer gradient skew on 16-layer MLPs, Conv2d channel/kernel skew (`img_bars4` collapses to `0/4` modes under both Rung 1 & Rung 2), multi-module `(G, E)` BiGAN skew (`five_word_joint_smoke` `3/5` modes), or $\operatorname{sech}^2(8) = 4.5 \times 10^{-7}$ saturation rails (pure SGD and SGD+secant stay 100% frozen at `a = 8.000`, `dG = 0.0`). Moving to **zero-state layerwise degree-0 Frobenius-MP SGD** (`law4-frob-sgd-mp`, `readout-12f988a4ce689b5994a4ea61.json`, `0` tensor states/param, `0` NS, `0` Adam) collapses MLP16 skew from `7,500× -> 19×`, escapes `a = 8.0 -> 1.837`, and jumps to **`9/16` PASS** — unlocking `grid100` (`prec=0.9724, ctr=0.0660`), `five_word_joint_smoke` (`5/5, tv=0.0189, recon=1`), `gaussian1d_stability` (`ks=0.0301`), `img_bars4` (`4/4`), and both `>500×` Nash tasks (`0.9583 / 0.9709`).
4. **Rank-1 Spectral Collapse Beyond Frobenius Normalization:** Because Frobenius normalization (`G / ||G||_F`) is rank-preserving, `law4-frob-sgd-mp` still collapses on 16-layer skip-free `grid100_mlp16` (`prec = 0.1565`) and under-steps minor singular directions on `model_ddgan_gaussian_chain` (`t1_q0 = 0.8386`) and anisotropic mixtures — isolating the exact mathematical leap completed by 5-step Quintic Newton–Schulz polar orthogonalization (`unified-v11`, `16/16`).

### 2. "Or SGD on G if impedance mismatch isn't a problem?"
**Partial YES on shallow/balanced/4-mode Conv2d/Nash regimes — definitive NO on multi-scale, deep, multi-module, and rail regimes — AND our causal reverse control proves a striking G-vs-D Asymmetry Law reversing conventional intuition!**

- **SGD on G without G secant (`law4-sgd-g-equil-d`, Rung 3a, `readout-f19a0f5cc8ffd529c09444c0.json`):** **`5/13` PASS** (`img_bars4` jumps `0/4 -> 4/4 hq=1.0`, `nash_overpowered_critic` passes `16/16 hq=0.9783`, `t3_iso_horizon_saddle_shift` passes `0.9609`), yet `nash_overpowered_generator` collapses to `8/16` (`0.3672`) and `ring16` drops to `14/16` (`0.6897`).
- **SGD on G WITH Law-4 G secant band (`law4-sgd-g-secant-equil-d`, Rung 3b, `readout-fd14ca2051e98f707ab51fe9.json`):** **`9/15` PASS** — adding ONLY the G secant trust band flips `nash_overpowered_generator` (`8/16 -> 16/16 hq=0.9751`), `ring16` (`14/16 -> 16/16 hq=0.9814`), `gaussian1d_smoke`, and `zero_tanh_smoke` to PASS. Yet pure SGD on G **still fails 6 hard generator regimes** (`grid100` `prec=0.4655` gating `mlp16`, `five_word_joint_smoke` `3/5`, `model_ddgan_gaussian_chain` `t1_q0=1.4281`, `zero_tanh_rail_escape` `mae=0.0031`, `gaussian1d_stability`, `t3_iso_scale_mass_8x` `eig=0.4035`).
- **Causal Reverse Control — Equilibrated G & Prior, Pure SGD on D (`law4-equil-g-sgd-d`, Rung 4, `readout-294525ad76a2970a72a7caac.json`):** **`12/16` PASS** (**+7** over Rung 3a and **+3** over Rung 3b!). Pure unnormalized SGD on D passes `grid100` (`0.9919, ctr=0.1204`), `five_word_joint_smoke` (`5/5, tv=0.0189`), `model_ddgan_gaussian_chain` (`t1_q0=0.2596`), `zero_tanh_rail_escape` (`mae=0.0006`!), `t3_iso_scale_mass_8x` (`eig=0.6869`), `img_bars4` (`1.0`), `ring16` (`0.9873`), and `nash_overpowered_generator` (`0.9719`) — failing catastrophically ONLY on `nash_overpowered_critic` (`620×` $D \gg G$, instant `5.4s` NaN explosion), completing an **exact 2×2 Nash causal double dissociation**:
  - **Rung 3a (`SGD-G + Equil-D`):** `overpowered_critic` **PASS (`0.9783`)** · `overpowered_generator` **FAIL (`8/16, 0.3672`)**
  - **Rung 4 (`Equil-G + SGD-D`):** `overpowered_critic` **NaN EXPLOSION (`5.4s`)** · `overpowered_generator` **PASS (`16/16, 0.9719`)**
- **Why?** Law 2 (Born–Infeld 1-Lipschitz critic) already smooths D's potential on balanced architectures, whereas **G carries the nonlinear pushforward geometry** (100-mode lattices, 16-layer skip-free compositions, `tanh` rails, BiGAN `(G, E)`, reverse-diffusion fibers) and therefore strictly requires Law-4 per-layer equilibration far more than D!

---

## Verbatim 16-Task × 7-Arm Causal Matrix (`r4-occam-ablation-v1`, Seed `0`, RTX A6000)

| Probe Task | `unified-v11` (Ref) | R1 `pure_sgd` | R2 `sgd_secant` | R3a `sgd_g_equil_d` | R3b `sgd_g_sec+eq_d` | R4 `equil_g_sgd_d` | R5 `frob_sgd_mp` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gaussian1d_smoke` *(prereq)* | **PASS** | FAIL (`0.0820`) | FAIL (`0.0880`) | FAIL (`0.0718`) | **PASS** (`0.0475`) | **PASS** (`0.0242`) | **PASS** (`0.0240`) |
| `gaussian1d_stability` | **PASS** | GATED | GATED | GATED | FAIL (`0.0452`) | FAIL (`0.0201`) | **PASS** (`0.0301`) |
| `zero_tanh_smoke` *(prereq)* | **PASS** | FAIL | FAIL | FAIL | **PASS** (`0.0020`) | **PASS** (`0.0012`) | **PASS** (`0.0031`) |
| `zero_tanh_rail_escape` | **PASS** | GATED | GATED | GATED | FAIL (`0.0031`) | **PASS** (`0.0006`) | FAIL (`0.0063`) |
| `ring16_acquisition` | **PASS** | FAIL (`0.8071`) | **PASS** (`0.9841`) | FAIL (`14/16`) | **PASS** (`0.9814`) | **PASS** (`0.9873`) | **PASS** (`0.9573`) |
| `vector_unequal_width` | **PASS** | **PASS** (`0.9990`) | FAIL (`tv=0.101`) | **PASS** (`0.9973`) | **PASS** (`0.9998`) | **PASS** (`0.9973`) | FAIL (`0.9836`) |
| `vector_unequal_mass` | **PASS** | **PASS** (`0.9963`) | **PASS** (`0.9993`) | **PASS** (`0.9983`) | **PASS** (`0.9951`) | **PASS** (`0.9973`) | FAIL (`0.9998`) |
| `img_bars4` | **PASS** | FAIL (`0/4`) | FAIL (`0/4`) | **PASS** (`1.0000`) | **PASS** (`1.0000`) | **PASS** (`1.0000`) | **PASS** (`0.9688`) |
| `grid100` | **PASS** | FAIL (`0.1466`) | FAIL (`0.4545`) | FAIL (`0.1358`) | FAIL (`0.4655`) | **PASS** (`0.9919`) | **PASS** (`0.9724`) |
| `grid100_mlp16` | **PASS** | GATED | GATED | GATED | GATED | FAIL (`0.9722`) | FAIL (`0.1565`) |
| `model_ddgan_gaussian_chain` | **PASS** | FAIL (`1.328`) | FAIL (`1.328`) | FAIL (`1.414`) | FAIL (`1.428`) | **PASS** (`0.2596`) | FAIL (`0.8386`) |
| `five_word_joint_smoke` | **PASS** | FAIL (`3/5`) | FAIL (`3/5`) | FAIL (`4/5`) | FAIL (`3/5`) | **PASS** (`5/5`) | **PASS** (`5/5`) |
| `nash_overpowered_critic_ring16` | **PASS** | **NaN** *(INC)* | **PASS** (`0.9768`) | **PASS** (`0.9783`) | **PASS** (`0.9648`) | **NaN** *(INC)* | **PASS** (`0.9583`) |
| `nash_overpowered_generator_ring16` | **PASS** | **NaN** *(INC)* | **PASS** (`0.9751`) | FAIL (`8/16`) | **PASS** (`0.9751`) | **PASS** (`0.9719`) | **PASS** (`0.9709`) |
| `t3_iso_scale_mass_8x` | **PASS** | FAIL (`1/4`) | FAIL (`1/4`) | FAIL (`eig=0.404`) | FAIL (`eig=0.404`) | **PASS** (`0.6869`) | FAIL (`0.9414`) |
| `t3_iso_horizon_saddle_shift` | **PASS** | FAIL (`0/4`) | FAIL (`0/4`) | **PASS** (`0.9609`) | **PASS** (`0.9609`) | FAIL (`0.9219`) | FAIL (`0.8984`) |
| **Verbatim PASS / Executed** | **`16 / 16`** | **`2 / 13`** *(2 NaN)* | **`4 / 13`** | **`5 / 13`** | **`9 / 15`** | **`12 / 16`** *(1 NaN)* | **`9 / 16`** |

### Committed Official Forge Readout Receipts
- **Rung 1 (`law4-pure-sgd`):** [`readout-62543241e3cb64f098759ad7.json`](../records/readout-62543241e3cb64f098759ad7.json)
- **Rung 2 (`law4-sgd-secant`):** [`readout-d5c8d860e8107c9cc66729c6.json`](../records/readout-d5c8d860e8107c9cc66729c6.json)
- **Rung 3a (`law4-sgd-g-equil-d`):** [`readout-f19a0f5cc8ffd529c09444c0.json`](../records/readout-f19a0f5cc8ffd529c09444c0.json)
- **Rung 3b (`law4-sgd-g-secant-equil-d`):** [`readout-fd14ca2051e98f707ab51fe9.json`](../records/readout-fd14ca2051e98f707ab51fe9.json)
- **Rung 4 (`law4-equil-g-sgd-d`):** [`readout-294525ad76a2970a72a7caac.json`](../records/readout-294525ad76a2970a72a7caac.json)
- **Rung 5 (`law4-frob-sgd-mp`):** [`readout-12f988a4ce689b5994a4ea61.json`](../records/readout-12f988a4ce689b5994a4ea61.json)
- **Mechanistic Telemetry (80-step seed-0 anatomy across 5 regimes):** [`law4-sgd-anatomy-telemetry.json`](law4-sgd-anatomy-telemetry.json)
