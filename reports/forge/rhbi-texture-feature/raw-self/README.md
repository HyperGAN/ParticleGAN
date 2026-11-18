# RHBI texture recovery with raw self-repulsion

**Full feature attraction with raw self-repulsion recovered the 32×32 textures.** At update 288, 241/256 generated samples qualified as clean textures. Both orientations appeared, and all six texture bounds passed at every required terminal check. The same recipe failed sparse-arc interpolation. [Texture measurements](texture.json) · [Arc measurements](arc.json)

| Recipe | Texture gate | High-frequency energy ratio | Clean textures | Class mass TV | Phase diversity | Sparse-arc gate | Final good chord interiors |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Feature attraction, raw self-repulsion | [PASS 5/5](../../records/readout-da17f462507164779b6b47ac.json) | 0.6700 | 94.14% | 0.1250 | 0.8049 | [FAIL 0/5](../../records/readout-c12c9b140cd30d69b788d46d.json) | 51/69 |
| Full feature transport | [PASS 5/5](../../records/readout-9900c47fe4b1edd1a04a8f96.json) | 0.6959 | 90.63% | 0.1328 | 0.7547 | [FAIL 0/5](../../records/readout-dac256adfce23a647398b368.json) | 51/69 |
| Raw RHBI-v12 | [FAIL 0/5](../../records/readout-1bb7f1ca1295d6b4243338f9.json) | 0.0086 | 0.00% | 0.0039 | 0.0000 | [PASS 5/5](../../records/readout-93dcfd4e0ea1ccf289dda8a7.json) | 53/66 |

The texture gate requires energy ≥0.55, clean fraction ≥75%, both orientations, class mass TV ≤0.18, phase diversity ≥0.40 and finite fraction 1. In the candidate's five terminal observations, clean fraction ranged from 93.75% to 97.27%. Each matched control reproduced all 24 previous observations on each task. Six runs consumed 2,664 updates and 338.16 worker-seconds. [Texture training GIF](texture-training.gif) · [Sparse-arc training GIF](arc-training.gif)

## What the arc result isolates

Both feature recipes learned all 12 endpoints. Their final learned graph had 23 edges and 51/69 high-quality interior samples, below the 80% floor. The only missed terminal bound was interior quality. On the raw parent's own 22 edges, the parent passed 53/66 and both feature recipes passed 49/66. The feature recipes had identical generator, critic and prior tensors at all 101 saved states and identical executed cross-plans on all 600 updates. Changing self geometry had no measurable effect on this arc trajectory. The cross-graph comparison is exploratory. Each official gate uses its own learned graph. [Checkpoint and graph comparison](arc-edges.json)

## Research decision

Keep the raw-self variant as a texture-positive reference. Select it with `get_recipe("rhbi_optimized", rhbi_feature_transport="raw_self")` for a scalar marginal GAN with a supported critic feature head. This was one seed on a selected synthetic image task. Recovery on CIFAR and superiority to the full-feature control remain open.

To pursue interpolation, the next comparison would hold raw self-repulsion and feature-based physical attraction fixed, then switch the real–fake assignment and slack back to raw coordinates. The earlier assignment-only experiment tested the opposite combination. Together those cases can distinguish the roles of choosing a target and moving toward it.

## Frozen test

Start from the texture-positive full feature recipe. In the marginal generator objective, set `rhbi_feature_transport="raw_self"`. Use generated samples in raw coordinates for both the fake–fake Sinkhorn plan and the physical Born–Infeld self-cost. Keep the full-feature real–fake assignment and physical cost, including the generator derivative through normalized critic features. The same cross-plan sets the slack weights and scales the self-repulsion by each row's active mass. Retain the parent's support terms, critic objective, horizons, learned prior update policy and SBNP regularizer. This recipe has no new coefficient. [Previous assignment-only experiment](../assignment/README.md)

Run the candidate, raw RHBI-v12 and full feature transport on the complete 288-update texture test. All arms use protocol seed 0 and the public deterministic initializer. Keep architecture, prior capacity and initial law, batches, sampling, budget and all 24 evaluations fixed within each task. The clean live generator must meet every task threshold at all five terminal observations. The positive control must pass and the raw control must fail. Reserve 300 worker-seconds per arm, 900 per stage.

If the candidate and controls meet those conditions, run the three arms at the same source on the 600-update sparse arc. The predicted outcomes are a parent PASS, full-feature FAIL and candidate PASS on all five terminal observations. Report texture and arc separately. If changing the self geometry loses texture, check the failing bounds before another mechanism and preserve the full-feature recipe as the texture reference. Scope ends before CIFAR, additional seeds or default adoption.

## Reproduction

From the repository root, supply the Forge run directory containing the raw-self campaigns and the archived sparse-arc control campaign. The reducer verifies the common source and recipe delta, the public initializer and prior tensors, seen batches, named initial and terminal RNG streams, optimizer counts, published file hashes, and independent Forge grades. The arc diagnostic restores each endpoint and checks that rescoring its native graph reproduces the official chord metric.

```sh
PYTHONPATH=. python reports/forge/rhbi-texture-feature/raw-self/compare.py /path/to/runs/forge texture
PYTHONPATH=. python reports/forge/rhbi-texture-feature/raw-self/compare.py /path/to/runs/forge arc
PYTHONPATH=. python reports/forge/rhbi-texture-feature/raw-self/arc_edges.py /path/to/runs/forge
```

Tail either campaign's `progress.jsonl` or the attempt's `run.log` under `runs/forge/rhbi-raw-self-{texture,arc}-v1/`. The full checkpoints and transport matrices are archived there.
