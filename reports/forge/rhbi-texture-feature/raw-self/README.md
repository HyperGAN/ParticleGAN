# RHBI texture recovery with raw self-repulsion

## Frozen test

Start from the texture-positive full feature recipe. In the marginal generator objective, set `rhbi_feature_transport="raw_self"`. Use generated samples in raw coordinates for both the fake–fake Sinkhorn plan and the physical Born–Infeld self-cost. Keep the full-feature real–fake assignment and physical cost, including the generator derivative through normalized critic features. The same cross-plan sets the slack weights and scales the self-repulsion by each row's active mass. Retain the parent's support terms, critic objective, horizons, learned prior update policy and SBNP regularizer. This recipe has no new coefficient. [Previous assignment-only experiment](../assignment/README.md)

Run the candidate, raw RHBI-v12 and full feature transport on the complete 288-update texture test. All arms use protocol seed 0 and the public deterministic initializer. Keep architecture, prior capacity and initial law, batches, sampling, budget and all 24 evaluations fixed within each task. The clean live generator must meet every task threshold at all five terminal observations. The positive control must pass and the raw control must fail. Reserve 300 worker-seconds per arm, 900 per stage.

If the candidate and controls meet those conditions, run the three arms at the same source on the 600-update sparse arc. The predicted outcomes are a parent PASS, full-feature FAIL and candidate PASS on all five terminal observations. Report texture and arc separately. If changing the self geometry loses texture, check the failing bounds before another mechanism and preserve the full-feature recipe as the texture reference. Scope ends before CIFAR, additional seeds or default adoption.
