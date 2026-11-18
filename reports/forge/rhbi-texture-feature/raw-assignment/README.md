# Can raw matching preserve texture and recover arc interpolation?

**Raw matching with feature-based attraction passed both the 32×32 texture and sparse-arc gates.** Each gate held at all five terminal observations. At the final image evaluation, 256/256 generated samples passed the clean-texture criterion. The arc model learned all 12 endpoints and passed 56/69 chord interiors. [Texture measurements](texture.json) · [Arc measurements](arc.json)

| Recipe | Texture gate | Clean textures at update 288 | High-frequency energy ratio | Arc gate | Good interiors at update 600 |
| --- | --- | ---: | ---: | --- | ---: |
| Raw assignment, feature attraction | [PASS 5/5](../../records/readout-2a638eb3fa03819e6015bfcd.json) | 256/256 | 0.7055 | [PASS 5/5](../../records/readout-68041f47a43e69a562505633.json) | 56/69 |
| Feature assignment and attraction (parent) | [PASS 5/5](../../records/readout-69e232cfff5c56be8649104f.json) | 241/256 | 0.6700 | [FAIL 0/5](../../records/readout-c083425e72ca647a8b8613cf.json) | 51/69 |
| Raw RHBI-v12 | [FAIL 0/5](../../records/readout-b95be1e653ca6b69470c481b.json) | 0/256 | 0.0086 | [PASS 5/5](../../records/readout-3faa4a64e574f0506bfa33d4.json) | 53/66 |

At the last five image observations, 93.75% to 100% of the candidate's samples were clean and both orientations appeared each time. Final class mass TV was 0.1563 (ceiling 0.18), and phase diversity was 0.7712 (floor 0.40). Across both tasks each control reproduced all 24 of its earlier observations exactly. The six runs consumed 2,664 updates and 297.40 worker-seconds. [Texture training GIF](texture-training.gif) · [Arc training GIF](arc-training.gif)

## Did the learned graph explain the arc gain?

The candidate and its immediate parent selected exactly the same 23 pairs of particle rows. On those pairs the candidate passed 56 interior samples and the parent passed 51. On the raw RHBI control's 22 pairs, the candidate and raw control both passed 53/66, while the parent passed 49/66. All three native-graph rescoring results matched their official endpoint measurements. The 80% bound requires 56 passing samples out of 69, so a single additional missed interior would change the candidate's endpoint verdict. [Restored model and graph comparison](arc-edges.json)

On the parent's own inputs, replacing its assignment with raw matching moved at most 0.1372% of its transport mass on any arc update. Once the arms trained separately, only 12 of their 600 executed cross-plans were identical. The candidate matched its own same-state raw replay at all 600 updates. These traces distinguish the small immediate change from the cumulative difference in the trained models.

## Research decision

Advance this fixed recipe to the remaining matched retention tasks covering unequal-mass rare-mode width, moving image patch and saddle interpolation. The prior rare-width comparison failed in the raw baseline as well, so include the same-source controls. The result here establishes recovery on two selected synthetic tasks at seed 0. CIFAR quality and independent confirmation remain open. [Prior retention results](../README.md)

The public setting for scalar marginal GANs with a supported critic head is `get_recipe("rhbi_optimized", rhbi_feature_transport="raw_assignment")`.

## Frozen comparison

The parent recovered texture with feature-based attraction and raw self-repulsion, then failed the arc gate. Set `rhbi_feature_transport="raw_assignment"` and change only the detached real–fake cost that enters the marginal generator's Sinkhorn solve. Compute this cost in raw coordinates using the same local horizons, centered critic scores, masks, and slack convention. The resulting plan supplies the real matches, slack weights and active fractions that weight self-repulsion. Keep the physical cross-cost and its derivative in augmented raw-plus-critic-feature coordinates, the support-completion geometry from the parent, the raw fake–fake plan and cost, the learned prior policy and SBNP regularizer. [Parent experiment](../raw-self/README.md)

Compare the candidate with its immediate parent and raw RHBI-v12. Within each task use a single frozen source, protocol seed 0, the public deterministic initializer, and matched prior capacity, starting tensors, seen batches, sampler, budgets and evaluation cadence. Reserve 300 worker-seconds per arm per stage. First require the complete texture gate across the last five of 24 observations after 288 updates. Both controls must reproduce their published 24 observations, with a parent PASS and a raw RHBI FAIL. Only that result admits the three matched sparse-arc runs of 600 updates. The arc prediction is a candidate PASS alongside a raw RHBI PASS and a parent FAIL. Grade the clean live model on its own learned graph.

A numerical texture failure stops this candidate. A disagreement between controls or execution contracts triggers an evidence review. There is no seed search, retuning, CIFAR run or default promotion in this study. Source-bound receipts determine eligibility.

## Reproduction

From the repository root, point the shared reducer at the directory holding the two raw-assignment Forge campaigns. It checks source and recipe identity, initialization, batches, named RNG streams, update budgets, independent grades, control observation hashes and actual-training media. The arc diagnostic reloads the terminal checkpoints and verifies their native-graph results.

```sh
PYTHONPATH=. python reports/forge/rhbi-texture-feature/raw-self/compare.py /path/to/runs/forge texture --study raw-assignment
PYTHONPATH=. python reports/forge/rhbi-texture-feature/raw-self/compare.py /path/to/runs/forge arc --study raw-assignment
PYTHONPATH=. python reports/forge/rhbi-texture-feature/raw-self/arc_edges.py /path/to/runs/forge --study raw-assignment
```

Each campaign's `progress.jsonl` and its attempts' `run.log` can be tailed in `runs/forge/rhbi-raw-assignment-{texture,arc}-v1/`. The per-update traces and full checkpoints stay in the local archive.
