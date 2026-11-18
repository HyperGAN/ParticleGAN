# Does raw assignment preserve texture across the RHBI retention tasks?

**Raw assignment with feature attraction passed texture, sparse arc and unequal-mass width. It failed moving-patch sharpness and saddle interpolation.** Raw RHBI passed both of those interpolation tasks, while its texture collapsed and its rare-mode width fell below the threshold. All five tasks used the same global recipe per arm and protocol seed 0. [Texture measurements](texture.json) · [Arc measurements](arc.json) · [Remaining retention measurements](retention/results.json)

| Recipe | Texture | Sparse arc | Unequal mass | Moving patch | Saddle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw assignment, feature attraction | [5/5](../../records/readout-2a638eb3fa03819e6015bfcd.json) | [5/5](../../records/readout-68041f47a43e69a562505633.json) | 5/5 | 4/5 | [1/5](../../records/readout-47ae4e9c8c2e782e4747f222.json) |
| Feature assignment and attraction (parent) | [5/5](../../records/readout-69e232cfff5c56be8649104f.json) | [0/5](../../records/readout-c083425e72ca647a8b8613cf.json) | 5/5 | 4/5 | [0/5](../../records/readout-e77a1c6bb3e4b9bf185d0dc1.json) |
| Raw RHBI-v12 | [0/5](../../records/readout-b95be1e653ca6b69470c481b.json) | [5/5](../../records/readout-3faa4a64e574f0506bfa33d4.json) | 3/5 | 5/5 | [5/5](../../records/readout-4eed35ca04c58e48dcea5195.json) |

Each cell counts the last five observations at which every bound passed. All five are required. The texture and arc columns come from their original six runs. The other columns come from nine same-source runs. At the final image evaluation, the candidate, parent and raw RHBI produced 256, 241 and 0 clean textures out of 256 respectively. Their high-frequency energy ratios were 0.7055, 0.6700 and 0.0086. At the final arc evaluation, they passed 56/69, 51/69 and 53/66 interiors respectively. [Retention report and actual-training GIFs](retention/README.md)

At the last five image observations, 93.75% to 100% of the candidate's samples were clean and both orientations appeared each time. Final class mass TV was 0.1563 (ceiling 0.18), and phase diversity was 0.7712 (floor 0.40). Across both tasks each control reproduced all 24 of its earlier observations exactly. The six runs consumed 2,664 updates and 297.40 worker-seconds. [Texture training GIF](texture-training.gif) · [Arc training GIF](arc-training.gif)

## Did the learned graph explain the arc gain?

The candidate and its immediate parent selected exactly the same 23 pairs of particle rows. On those pairs the candidate passed 56 interior samples and the parent passed 51. On the raw RHBI control's 22 pairs, the candidate and raw control both passed 53/66, while the parent passed 49/66. All three native-graph rescoring results matched their official endpoint measurements. The 80% bound requires 56 passing samples out of 69, so a single additional missed interior would change the candidate's endpoint verdict. [Restored model and graph comparison](arc-edges.json)

On the parent's own inputs, replacing its assignment with raw matching moved at most 0.1372% of its transport mass on any arc update. Once the arms trained separately, only 12 of their 600 executed cross-plans were identical. The candidate matched its own same-state raw replay at all 600 updates. These traces distinguish the small immediate change from the cumulative difference in the trained models.

## Research decision

Keep CIFAR on hold. The candidate preserved rare-mode width for 19 consecutive checks. Its final moving-patch peak was 0.7737 against a floor of 0.78, and only 53/71 saddle interiors met the quality criterion at the final check. Raw RHBI passed 59/71 on the same triangle row IDs. Next, use the saved states to compare how raw and feature attraction move the generated interiors. These five synthetic tasks are a selected diagnostic suite, and CIFAR quality and independent confirmation remain open. [Retention analysis](retention/README.md)

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
