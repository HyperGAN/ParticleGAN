# Can feature assignment alone recover texture?

**Feature assignment alone failed the texture gate.** It raised the final high-frequency energy ratio from 0.009 to 0.199, below the 0.55 floor. None of its 256 clean samples qualified at any of the 24 evaluations. Full feature transport reached 0.696 energy and 90.6% clean textures. Both controls reproduced all 24 observations from the earlier experiment exactly. [Measured results](results.json) · [Actual assignment-only training GIF](training.gif)

| Recipe | Full gate | Terminal checks passed | High-frequency energy ratio | Clean texture fraction | Orientations recovered | Class mass TV | Phase diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Full feature transport](../../records/readout-7140c16ce14700569784ef15.json) | PASS | 5/5 | 0.6959 | 0.9063 | 2 | 0.1328 | 0.7547 |
| [Feature assignment, raw motion](../../records/readout-c20e0ab213205669efe1045e.json) | FAIL | 0/5 | 0.1992 | 0.0000 | 0 | 0.0742 | 0.0000 |
| [Raw RHBI-v12](../../records/readout-270008de8ad43a04e4813b2b.json) | FAIL | 0/5 | 0.0086 | 0.0000 | 0 | 0.0039 | 0.0000 |
| Required at each terminal check | | 5/5 | ≥0.55 | ≥0.75 | 2 | ≤0.18 | ≥0.40 |

Table metrics are from update 288. The split's energy ratio ranged from 0.1875 to 0.2025 over the last five evaluations. It recovered zero orientations and had zero phase diversity throughout that window. Class mass TV also missed its bound at update 240 (0.1992). Stronger high-frequency output alone did not recover either oriented texture class.

## Research decision

Stop the assignment-only recipe at the texture gate. Sparse arc, unequal mass and CIFAR remain unmeasured for this variant. All three texture arms used one source, seed and budget. The matched batches, named stream initialization, initial model and prior hashes, runtime, evaluation law and recipe fields outside the single mode agree. All 14 named RNG streams also finished in identical states. Three runs consumed 864 updates and 143.41 worker-seconds.

Full feature transport still supplies the positive texture control and carries its previously measured sparse-arc regression. The split removed feature-space movement toward real examples and feature-space self-repulsion together. The next experiment I would test changes only the fake–fake plan and cost to raw coordinates in the full-feature recipe. Keep its cross-plan, physical attraction, auxiliary terms and critic updates intact. A texture PASS would justify a matched arc test to learn whether removing feature self-repulsion preserves the interiors. A texture FAIL would implicate the choice of self geometry under that recipe and budget.

## Frozen protocol

Compare raw RHBI-v12, full feature transport and one assignment-only variant on the existing 288-update, 32×32 texture gate. Use one frozen source, the public deterministic initializer, protocol seed 0, and the same architecture, batches, starting prior, sampling, budgets and 24 scheduled observations. Reserve 300 worker-seconds per arm, 900 total. The parent and full-feature runs check that the earlier negative and positive controls reproduce under the new source. This is a selected research diagnostic.

The single recipe change is `rhbi_feature_transport="assignment_only"`. Solve the marginal generator cross-plan, including real matches and slack, using the original concatenation of raw coordinates and normalized critic features. Each generator update treats that plan as fixed. Compute its physical Born–Infeld movement costs and the fake–fake plan and costs in raw coordinates. Preserve the recipe's active-mass weighting of self-repulsion, raw-coordinate support-completion distances, raw-real transport horizons, critic/logit and slack terms, prior updates and SBNP regularizer. Gradients through the ordinary critic logits still contribute. There are no learned or tuned constants in the split.

Require the complete sustained texture gate on the split plus reproduction of the raw FAIL and full-feature PASS. Only that outcome opens a same-source comparison on sparse arc and unequal mass with all three recipes, at 600 and 1,200 updates per arm respectively (6,300 worker-seconds maximum). If a control changes gate, investigate comparability before interpreting the split. A numerical texture failure stops this candidate. No seed search or CIFAR run is part of the study.

## Reproduction

Run the reducer against the original and current texture campaigns. It checks the matched contracts and actual-training GIF hashes, compares the control observations and prints the compact measurements.

```sh
PYTHONPATH=. python reports/forge/rhbi-texture-feature/assignment/compare.py /path/to/runs/forge
```

Each attempt's `run.log` and the campaign's `progress.jsonl` are in `runs/forge/rhbi-assignment-texture-v1`.
