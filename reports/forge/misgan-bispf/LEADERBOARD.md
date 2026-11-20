# Four-mode MisGAN status

Corrected independent-game **v2 is unmeasured** and inherits no prior PASS. The table below preserves the original coupled-state v1 grades and numbers under [source 2025ca28](https://github.com/255BITS/ParticleGAN/tree/2025ca2875517cd3cdf8a91bba1ecfeebf3ba7ca). Its single shared critic history is distinct from v2. No new scientific run or regrading occurred.


One fixed four-mode task and protocol; all failures retained. No ordinary or historical 100-mode qualification.

| Recipe | Forge gate | Last 5 checks | Generator HQ | Mass TV | Mask TV | Imputer accuracy | Ambiguous HQ | Branch TV | Branch coverage | GIF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bispf-misgan-transport-settle-v1` | PASS | 5/5 | 0.9890 | 0.0940 | 0.0106 | 0.9727 | 0.9373 | 0.0781 | 1.0000 | [actual training](../../toy_audit/api_contract/misgan_bispf/media/bispf-misgan-transport-settle-v1.gif) |
| `bispf-optimized-quintic-v1` | FAIL | 0/5 | 0.9966 | 0.2441 | 0.0212 | 0.9739 | 0.9932 | 0.0447 | 1.0000 | [actual training](../../toy_audit/api_contract/misgan_bispf/media/bispf-optimized-quintic-v1.gif) |
| `bispf-misgan-observed-v1` | FAIL | 0/5 | 0.9985 | 0.1697 | 0.0213 | 0.9695 | 0.9861 | 0.0879 | 1.0000 | [actual training](../../toy_audit/api_contract/misgan_bispf/media/bispf-misgan-observed-v1.gif) |
| `bispf-misgan-amortized-v1` | FAIL | 0/5 | 0.9946 | 0.1697 | 0.0248 | 0.9717 | 0.9827 | 0.1433 | 1.0000 | [actual training](../../toy_audit/api_contract/misgan_bispf/media/bispf-misgan-amortized-v1.gif) |

Charged elapsed time: 2307.489 seconds, including 3 preserved infrastructure errors. GPU contention prevents throughput comparison.
Matched initial model hashes: True; matched observed-batch digests: True.

Exact task, source, recipe, budget, RNG and sample-law identities are in [results.json](results.json). The gates do not certify within-mode covariance calibration.
