# Research glossary

| Term | Meaning |
|---|---|
| **Technique / formulation** | A training method, such as BCAP (`particlegan.Recipe` mechanism class / family). |
| **Recipe** | A concrete configuration of a technique: losses, weights, optimizer, prior learning or freezing policy, and prior regularizer (`configs/forge/configurations/<id>.json`). |
| **Candidate / Idea** | A versioned declaration (`configs/forge/configurations/<id>.json` or legacy `configs/forge/ideas/<id>.json`) binding a `Recipe` preset/overrides, prior policy, hypothesis, and lineage. |
| **Task** | The problem to solve (`configs/forge/tasks/<id>.json`), including architecture, target data, initial prior distribution and capacity, sampling, and evaluation criteria. |
| **View / Goal** | A versioned evaluation policy (`configs/forge/views/<id>.json`) that assigns tasks to qualification tiers with roles (`required`, `stretch` / `held`, `diagnostic`) and declares a view-level `ranking` rule. CLI commands accept `--view` and `--goal` interchangeably. |
| **Tier** | A progressive qualification stage (`qualification_tier >= 1`) within a view. Tier 1 screens fast smoke/stability gates; higher tiers run fuller accuracy, transfer, or continuation suites. |
| **Protocol** | The fixed comparison rules (`configs/forge/protocols/screening.json`): initialization, data sequence, update budget, evaluation cadence, and scoring (fixed seed `0` for screening). |
| **Study** | A bounded scientific comparison (`configs/forge/studies/<id>.json`) pairing a candidate against an explicit control on a declared view/tier scope with an immutable budget and stop rule. |
| **Search** | A finite, deterministic hyperparameter grid (`configs/forge/searches/<id>.json`) over `TUNABLE_FIELDS` that preserves the base candidate's technique signature. |
| **Campaign** | A resource budget envelope (`budget_seconds`, `candidate_budget_seconds`) governing queue reservations. Studies and searches embed their own campaign definitions. |
| **Attempt / Receipt** | One worker execution of a resolved job (`runs/forge/<campaign>/<attempt>/` and `reports/forge/attempts/<attempt>/`), bound by `result.json` and `evidence.json` SHA-256 certificates. |
| **Cohort** | A comparability partition defined by `(candidate_revision, source_digest, runtime_cohort)` (CPU vs CUDA and GPU model). Results from different cohorts are never silently merged. |
| **Evidence scope** | Whether an observation is `current` (regradable qualification input), `pinned` (archived source/task/runtime), `historical` (imported summary), `research_diagnostic` (bounded study/search), or `calibration_diagnostic`. |
| **Readout** | A concluded scientific record (`reports/forge/records/readout-<hash>.json`) binding a candidate's finished attempts to a `conclusion`, `comparison`, and `next_action`. Required before reusing or modifying a stopped candidate. |
| **Promotion** | A preregistered multi-seed robustness stage (`reports/forge/promotions/<id>/registration.json`) for a candidate that has already qualified and concluded its screening readout. |
| **Experiment** | A comparison of declared recipe changes under a fixed protocol. |
| **Run** | One execution of a recipe on a task. |
| **Metric** | A numerical measurement of performance. |
| **Gate** | A declared pass/fail requirement. |
| **Baseline** | The reference recipe used for comparison. |
| **Representative recipe** | A technique’s best observed complete recipe under the declared tasks, gates, and budgets. |
| **Counted requirement** | A test with the role `required` in a view. It counts toward the primary leaderboard denominator and toward tier qualification; a test in several views is one requirement per view. |
| **Stretch test** | The single canonical term for extra-credit, optional, frontier, and held benchmark experiments (role `stretch`, or legacy `held`). Stretch tests are declared, run, graded, and shown in their own **`## Stretch experiments`** leaderboard table by tier (`Tier 1 stretch`–`Tier 4 stretch`, `Total stretch`) as well as in the full-suite results table. They never block tier qualification or counted `PASS` status. |
| **Held test** | A previously required test moved to the stretch cohort with role `held` (a backwards-compatible alias of `stretch` that carries a disposition note: `reason`, `summary`, `explanation`, and `reentry` rule). It is tallied in the **`## Stretch experiments`** table and full-suite comparison alongside new `stretch` tasks. |
| **Removed from the count** | A test that stays assigned as a diagnostic with a note naming why it is not counted: a phase or duplicate of a counted test, a superseded definition, or a test that cannot run as declared. |
| **Ranking (`view.ranking`)** | A view-level ordering policy (`policy: "qualified_tier_only_with_raw_metrics"`, `compare_compatible_cohorts: true`, `cost_separate: true` in `configs/forge/views/*.json`; `order_by: ["qualified_tier", "candidate_id"]`, `metric_aggregation_allowed: false` in `particlegan_forge`) that orders candidates by highest qualified tier rather than cross-task scalar metric sums. Distinct from task assignment roles (`required`, `stretch`, `diagnostic`; legacy task role `ranking` is retained in `IMPORTANCES` only for schema compatibility and is superseded by `stretch`). |
| **STRETCH / HELD status** | Reported for a tier with no counted requirements. A tier with `stretch` tasks (and no `held` tasks) reports `STRETCH` and never blocks view-level `PASS` when all counted tiers pass; a tier with legacy `held` tasks (and no `required` tasks) reports `HELD` (`counted_eligible` records whether all counted tiers passed). |
| **Archived view revision** | The exact bytes of an earlier view revision, copied to `configs/forge/view-history/<view>-v<N>.json` before the view changes. Recorded results keep pointing to it. |

**Ownership:** The recipe owns prior updates and regularization (`experiments/forge/boundaries.py`). The task defines the initial prior distribution and capacity, and the sampling contract.

**Selection:** Compare one global recipe across the task suite. Keep failures visible; avoid selecting a different recipe for each task.

**Stretch & holding:** Use `importance: "stretch"` as the single canonical role for optional, extra-credit, frontier, and held experiments. Stretch tasks are visible in the dedicated **`## Stretch experiments`** table on [reports/forge/technique-inventory.md](reports/forge/technique-inventory.md) and in the full-suite comparison without gating required qualification. When a previously required test is moved out of the counted denominator because no single global recipe passes it together with every counted test (`no_global_recipe`), because it is unsolved (`unsolved`), or because it continues a stretch/held parent (`held_prerequisite` / `stretch_prerequisite`), its view stores the disposition summary, explanation link, and re-entry rule. See the [2026-10-15 test dispositions](reports/forge/test-dispositions-20261015/README.md) and the [problem list](reports/forge/problems.md) for each question's intent, failures and known passing configurations.

## Acronyms and short names

Expansions are quoted only where the repository records one; the source column says where.
"No recorded expansion" means no code, docstring, report or commit message spells the name out,
so the row gives a functional description instead.

| Name | Expansion | What it is | Source |
|---|---|---|---|
| **EMPF** | Exponential Moving Prototype Flow | The current default family (`DEFAULT_RECIPE = "empf"`): clock-free online *e-means* sufficient-statistics prototype bank, mass-weighted prototype-to-fake Sinkhorn OT, state-driven settling Born-Infeld critic penalty, and DoF-aware EMPF prior regularizer. | `particlegan/empf.py` module docstring; `particlegan/recipes.py`; [`reports/forge/empf/README.md`](reports/forge/empf/README.md) |
| **BISPF** | Born-Infeld Sinkhorn Particle Flow | The previous 0.9.x default family (`get_recipe("bispf_optimized")` / `get_recipe("bispf")`): Sinkhorn relativistic pairing, a Born-Infeld critic penalty, the SBNP prior and residual-driven noise. | `particlegan/bispf.py` module docstring; `particlegan/recipes.py` |
| **SBNP** | Stratified Blue-Noise Particle Prior | BISPF's prior regularizer / particle table (`StratifiedBlueNoisePrior`). | `particlegan/bispf.py` module docstring |
| **SRpGAN** | Scale-Free Sinkhorn Relativistic Pairing | BISPF's pairing of real and fake samples in the relativistic loss. | `particlegan/bispf.py` module docstring |
| **EBI-TCR** | Elastic Born-Infeld Transport-Chord Regularization | BISPF's critic penalty, with a zero-mean logit gauge. | `particlegan/bispf.py` module docstring |
| **RDTN** | Residual-Driven Thermal Noise | BISPF's noise term. | `particlegan/bispf.py` module docstring |
| **BCAP** (BCap, `b_cap`) | No recorded expansion | Named after penalty arm `b_cap`, arm *b* of the August 2026 penalty study (`a_r1r2`, `b_cap`, `c_eikonal`, …). It is a fixed one-sided L2 cap on both real and fake critic input gradients, `coeff/2 * (E relu(‖g_r‖-κ)² + E relu(‖g_f‖-κ)²)`. In Forge, the BCAP family is a set of recipes built on this penalty (for example BCAP-pure and BCAP dualnorm). | `particlegan/grad_regularizers.py` (`GradientPenalty`); `experiments/analyze.py` (`ARM_ORDER`); [technique inventory](reports/forge/technique-inventory.md) |
| **R1 / R2** | No recorded expansion | Critic input-gradient penalties on real (R1) and fake (R2) samples. `a_r1r2` is the fixed zero-centred squared-L2 penalty on both. | `particlegan/grad_regularizers.py` (`GradientPenalty`); [`docs/api.md`](docs/api.md) |
| **K3** | No recorded expansion | A K3P predecessor. Its critic penalty is the learning-rate-driven blend `s·a_r1r2 + (1−s)·b_cap`, where `s` depends only on the critic's last applied learning rate: exactly `a_r1r2` while that rate is at least half its maximum, exactly `b_cap` at the .01 floor. A guard clips critic updates more than 5× their usual size after 200 updates per tensor. | [`reports/toy100/overnight-20260925/k3g-final.md`](reports/toy100/overnight-20260925/k3g-final.md) |
| **K3P** | No recorded expansion; recorded as "K3 + gradient anchoring to an EMA critic" | The 0.8.0 default. Its critic penalty hands over from R1 plus a fake-gradient cap to one-sided caps plus an EMA-critic anchor as the critic learning rate falls. | [`reports/toy100/continuous-practical-leaderboard.md`](reports/toy100/continuous-practical-leaderboard.md) results table; [`docs/k3p.md`](docs/k3p.md); `particlegan/k3p.py` |
| **K3G** | No recorded expansion; recorded as "K3 + generator guard" | A K3 variant that applies the critic guard to the generator too. | [`reports/toy100/continuous-practical-leaderboard.md`](reports/toy100/continuous-practical-leaderboard.md) results table |
| **KA2** | No recorded expansion | The default before BISPF. It is K3P with the critic-penalty blend and EMA-critic tracking driven by the critic's own Adam moment surprise rather than by its learning rate. | [`docs/ka2.md`](docs/ka2.md); `particlegan/ka2.py` module docstring |
| **A2** | No recorded expansion | Bounded sparse latent-row damping, used by K3P and KA2 (`latent_damping_max_rate`). | [`docs/ka2.md`](docs/ka2.md); `particlegan/k3p.py` (`LatentRowDamping`) |
| **E22** | No recorded expansion | A schedule-free policy preset (`get_recipe("e22")`). Nothing in its training control path computes statistics on raw data or raw generator outputs. It came from the "no-outputs pass" (`gan-attempts/noout-20260928`). | [`docs/e22.md`](docs/e22.md); commit `0657de783` message |
| **DV12** | No recorded expansion | The value `dv12` of `Recipe.continuous_policy`, the last of the `DataDriftController` variants `dv1`–`dv12`. It is E22's horizon-free data-innovation and game-stability controller (paired with `lr_control="stationarity"`) and also applies a latent perturbation with its own RNG stream. | `particlegan/continuous.py` (`DataDriftController`); `particlegan/recipes.py` (`continuous_policy`); [`docs/e22.md`](docs/e22.md) |
| **UCD** | No recorded expansion | A conditional critic with one output per class (or per time/class pair) and no condition input. The adversarial logit is `out[:, c]`, plus class cross-entropy. The repository cites arXiv:2510.00624 for it. | `lib/sparse_models.py` module docstring; `particlegan/conditioning.py` |
| **hq** | "high quality" | The fraction of generated samples within 3σ of their nearest true mode centre. The 100-Gaussian gate requires all 100 modes and `hq ≥ 0.95`. | `lib/toy_models.py` (`mode_coverage`); [`FINDINGS.md`](FINDINGS.md); `reports/readme-100gaussians/summary-current.json` |
| **RpGAN** / Rp logistic | Relativistic-paired (logistic) GAN | The relativistic paired loss used by K3P, KA2 and the earlier toy studies. | [`docs/ka2.md`](docs/ka2.md) |
| **DDGAN** | Denoising-diffusion GAN | The `ddgan` model family. | [`README.md`](README.md) model families table |
| **MoG** | Mixture of Gaussians | A Gaussian-mixture particle prior (`mog` recipe). | [`README.md`](README.md) model families table |
| **AE-GAN / VAE-GAN** | Autoencoder GAN / variational autoencoder GAN | An encoder routes data to particles. | [`README.md`](README.md) model families table |
| **BH** | Benjamini-Hochberg | Multiple-testing control at level q, used by row evidence and birth/death. | `particlegan/birth_death.py` module docstring; [`docs/e22.md`](docs/e22.md) |
| **mode TV / mass TV** | Mode-mass total variation | Total variation distance between generated and target mode masses. | [`docs/toy100.md`](docs/toy100.md) |
| **FID50k / FID5k** | No recorded expansion beyond the sample count | FID computed from 50k (final) or 5k (diagnostic) generated samples. | [`reports/cifar-ddgan/READOUT.md`](reports/cifar-ddgan/READOUT.md) |
| **GAN v3** | No recorded expansion | The recipe released in 0.7, kept as a Forge family. | [technique inventory](reports/forge/technique-inventory.md) |

## Repository layout for Forge

- `configs/forge/` — **Immutable & versioned declarations**: `configurations/` (v3 recipe cards), `ideas/` (legacy v1/v2 cards), `studies/` (bounded questions), `searches/` (finite hyperparameter grids), `tasks/` (problem contracts), `views/` (tier/role policies), `protocols/` (screening rules), `calibration/` (gate calibration profiles).
- `reports/forge/` — **Committed durable evidence**: `technique-inventory.md` (canonical family leaderboard), `problems.md` (per-task failure diagnoses), `records/` (concluded readouts and dispositions), `configuration-search/` & `cadence-studies/` (published study reports), `EXPERIMENT_MEMORY.md` & `compilation.json` (deterministic recall index).
- `runs/forge/` — **Repo-local operational queue** (untracked): `queue/state.json`, `events.jsonl`, source snapshots, and per-attempt `run.log` / `terminal.json` files.

## Agent quickstart

The one-page command sequence (recall → new → plan → run → board → readout) lives in
[EXPERIMENTATION.md](EXPERIMENTATION.md#quickstart).
