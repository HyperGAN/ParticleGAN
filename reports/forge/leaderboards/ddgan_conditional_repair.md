# DDGAN conditional repair: current research leaderboard

GPU 0 shared-load diagnostic only; no ordinary qualification or speed ranking. All runs use protocol seed 0, clean live four-step sampling and unchanged scientific gates.

One global recipe per row; current tasks are Gaussian and the full-budget bimodal wall-allowance variant. A missing full result remains unknown. Passing marginal subgates do not count as task passes.

| Global recipe | Current PASS / 2 | Current FAIL | Unknown | Retained incomplete attempts |
| --- | ---: | ---: | ---: | ---: |
| `bispf-ddgan-energy-annealed-v2` | 0/2 | 2 | 0 | 0 |
| `bispf-ddgan-energy-unconstrained-v3` | 0/2 | 2 | 0 | 0 |
| `bispf-optimized-quintic-v1` | 0/2 | 2 | 0 | 1 |
| `bispf-ddgan-energy-v1` | 0/2 | 1 | 1 | 1 |

All retained attempts and exact endpoint metrics:

| Recipe | Task | Verdict | Updates | Last 5 passes | Final CDF | Worst conditional CDF | GIF |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `bispf-optimized-quintic-v1` | `ddgan_repair_bimodal_chain_v1` | **INCOMPLETE** | >=3,542/5,000 | — | — | — | — |
| `bispf-ddgan-energy-v1` | `ddgan_repair_bimodal_chain_v1` | **INCOMPLETE** | >=3,334/5,000 | — | — | — | — |
| `bispf-optimized-quintic-v1` | `ddgan_repair_gaussian_chain_v1` | **FAIL** | 2,500/2,500 | 0/5 | 0.5905 | 0.6144 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-optimized-quintic-v1-ddgan_repair_gaussian_chain_v1.gif) |
| `bispf-ddgan-energy-v1` | `ddgan_repair_gaussian_chain_v1` | **FAIL** | 2,500/2,500 | 0/5 | 0.0937 | 0.1289 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-ddgan-energy-v1-ddgan_repair_gaussian_chain_v1.gif) |
| `bispf-optimized-quintic-v1` | `ddgan_repair_bimodal_chain_wall_v2` | **FAIL** | 5,000/5,000 | 0/5 | 0.3738 | 0.4728 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-optimized-quintic-v1-ddgan_repair_bimodal_chain_wall_v2.gif) |
| `bispf-ddgan-energy-annealed-v2` | `ddgan_repair_bimodal_chain_wall_v2` | **FAIL** | 5,000/5,000 | 0/5 | 0.0450 | 0.1272 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-ddgan-energy-annealed-v2-ddgan_repair_bimodal_chain_wall_v2.gif) |
| `bispf-ddgan-energy-annealed-v2` | `ddgan_repair_gaussian_chain_v1` | **FAIL** | 2,500/2,500 | 0/5 | 0.0617 | 0.1911 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-ddgan-energy-annealed-v2-ddgan_repair_gaussian_chain_v1.gif) |
| `bispf-ddgan-energy-unconstrained-v3` | `ddgan_repair_gaussian_chain_v1` | **FAIL** | 2,500/2,500 | 0/5 | 0.0316 | 0.3707 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-ddgan-energy-unconstrained-v3-ddgan_repair_gaussian_chain_v1.gif) |
| `bispf-ddgan-energy-unconstrained-v3` | `ddgan_repair_bimodal_chain_wall_v2` | **FAIL** | 5,000/5,000 | 0/5 | 0.0436 | 0.1024 | [actual training](../../forge/ddgan-conditional-repair/media/bispf-ddgan-energy-unconstrained-v3-ddgan_repair_bimodal_chain_wall_v2.gif) |

CDF bounds: each final and conditional panel <=0.12. A PASS also requires all mean, covariance and bimodal mode/local-spread bounds at every final five scheduled checks. [Full failed bounds, provenance and exact final metrics](../ddgan-conditional-repair/results.json).
