# RHBI marginal transport in critic feature coordinates

The 288-update texture gate tests the prediction that adding the critic's learned representation to RHBI's distance restores diverse high-frequency output. The selected baseline is rhbi-v12. The image baseline EMPF and the BISPF reference have frozen positive [texture controls](../conv-texture32-tier1/README.md).

For the marginal generator loss, concatenate each raw sample with the input to the critic's single scalar linear readout. Center features on the real-batch mean, divide by its RMS Euclidean radius (floor 1e-6), and multiply by the existing detached RHBI horizon. Real feature statistics are detached. The same metric supplies real–fake cost and fake self-coupling. The critic objective, slack term, raw-sample horizon and thermal statistics, optimizer and learned SBNP prior follow v12. This opt-in recipe requires a scalar marginal critic with a single linear score head. Capture uses the actual critic forward and the generator derivative through its features.

The [matched training trajectories](trajectory/README.md) show when the parent and feature arm diverged on rare width and arc interpolation.

## Frozen decision

One candidate, protocol seed 0. Compare with the parent on the complete 288-update texture gate. Only a full pass advances the candidate to the frozen rare-mass, sparse arc, patch-shift and saddle retention tasks. A failure on any of these stops external CIFAR evaluation. No retuning or seed search in this study. Results in this research view provide no ordinary qualification or default adoption credit.

The [probe](probe.py) restores the hash-bound baseline endpoint and measures the RHBI loss, critic gradients and nearest-neighbor distances for real textures and two paired damages against an independent real batch. [Probe results](probe.json).

## Results and decision

**RHBI acquired the textures and failed all four retention tasks. Stop this revision before CIFAR.** The texture arms used seed 0 on two RTX A6000 GPUs, one source commit, identical model/prior initialization hashes and identical real batches. Their resolved recipes differ in exactly one field. The source-bound probe measured a median normalized feature nearest-neighbor distance of 0.405 for real textures, 13.35 for blur and 15.39 for flat means. In raw pixels the corresponding distances were 16.69, 14.11 and 14.81. The feature coordinates reverse the ordering of damaged and real examples in this probe. [Matched identities](comparison.json) · [Actual candidate training GIF](texture-feature-training.gif)

| Matched texture arm | Full gate | Last five passed | High-frequency energy ratio | Clean texture fraction | Orientations | Mass TV | Phase diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RHBI-v12 with feature transport | PASS | 5 | 0.6959 | 0.9063 | 2 | 0.1328 | 0.7547 |
| RHBI-v12 | FAIL | 0 | 0.0086 | 0.0000 | 0 | 0.0039 | 0.0000 |
| Gate | | 5 | ≥0.55 | ≥0.75 | 2 | ≤0.18 | ≥0.40 |

[Candidate readout](../records/readout-293fdd1fd459e8a46fbd285b.json) · [Parent readout](../records/readout-4b999d9239756b51afedd9fd.json)

| Frozen retention test | Candidate gate | What failed |
| --- | --- | --- |
| Unequal mass | FAIL | All endpoint bounds passed. The final passing suffix was 1/5. At update 1,150 the smallest within-mode covariance eigenvalue ratio was 0.142 against a floor of 0.15. |
| Sparse arc interpolation | FAIL | Final endpoints were all high quality, but only 73.9% of chord samples met the quality bound of 80%. |
| Moving image patch | FAIL | Two of five terminal observations passed. At step 767 the measured interpolation peak was 0.768 against a floor of 0.78. |
| Saddle interpolation | FAIL | Zero of five terminal observations passed. At step 767 the high-quality barycenter fraction was 0.716 against a floor of 0.78. |

The published rhbi-v12 selection passes the four retention tasks under its original source. The fresh parent control in this study covers texture only. The retention readout's endpoint mass-TV prediction was satisfied; the preregistered decision requires all four complete gates. These six attempts used 276.3 seconds of worker wall time. [Retention readout](../records/readout-b8643c936f96a75fb059eb10.json) · [Original parent cohort](../families/rhbi.md)

The frozen interpolation scorer uses a float32 mean of per-row finite flags. On this CUDA runtime, even 82 true flags give 0.99999994. That trips the scorer's early return on the patch task at steps 667 and 700. The saddle endpoint also took the early return. Its reported error of 9.9 is a sentinel. Both tasks independently miss a measured bound at step 767, as shown above. Preserve these receipts. A future scorer revision should use an integer count for the all-finite decision and be tested with the observed row counts before comparison.

## Where the feature distance changes the match

The feature distance strongly redirected the blurry texture control. Its effect on the learned rare-mode and arc endpoints was small. Each row below compares both distances on identical clean generator outputs, real examples, critic scores and frozen RHBI state. The texture checkpoint comes from the raw-space parent. The other two checkpoints come from the feature-trained candidate. [Probe source](retention_probe.py) · [Compact measurements and checkpoint hashes](retention-probe.json)

| Frozen endpoint | Raw distance horizon ℓ | Active mass, raw → with features | Fraction of the transport plan moved¹ | Change in output gradient² |
| --- | ---: | ---: | ---: | ---: |
| Blurry texture, 288 updates | 15.625 | 100.00% → 3.16% | 97.56% | 10.12× |
| Unequal mass, 1,200 updates | 0.0275 | 82.46% → 82.45% | 0.035% | 0.00107× |
| Sparse arc, 600 updates | 0.0287 | 29.30% → 29.25% | 0.055% | 0.00181× |

¹ Total variation across the real-target and slack columns, pooled over generated rows. Active mass is the fraction assigned to a real example. The frozen solver can send the rest to slack. ² L2 norm of the gradient difference divided by the raw-gradient norm for the marginal generator loss. The cosine between the two gradients was −0.009 on texture and above 0.99999 on each retention task. This comparison measures gradients at the generated outputs, before the generator Jacobian, separate prior and simplicial terms, and optimizer.

**Unequal mass.** Both distances assigned zero active mass across the four mode boundaries on these batches. Among the 99 generated examples nearest the 2% component, active mass was 65.52% with either distance. Two batches had no rare real observation, and both distances routed all six rare generated rows in those batches to slack. Re-scoring the original 4,096 stored samples at update 1,150 found 119 rare examples (2.91%). Their smallest normalized covariance eigenvalue was 0.142 against the 0.15 floor. The sample count and thin direction distinguish a width failure from disappearance of the rare component at that observation. Only the endpoint has a saved critic for the coupling probe.

**Sparse arc.** All 12 endpoints met the quality bound. All 18 failures among the 69 chord interiors lay inside the circle. The median angle between endpoints was 48.2° for failed interiors and 25.9° for passing interiors. When the 69 interiors were presented to the marginal loss as a diagnostic, only 8.38% → 8.32% of the failed interiors' mass matched real points, versus 72.33% → 72.27% for passing interiors. The ordinary endpoint-sample batches did change their top real example on 19.2% of common active mass. On those changes the old and new winners differed by just 1.1×10⁻⁷ of the raw row's active probability on average. Mean active angular mismatch was 1.323° → 1.321°.

**Decision.** Keep the texture mechanism as an image result. These endpoint measurements give little reason to tune feature strength or redesign matching to address retention. The most informative next training comparison would capture the first divergence between the parent and candidate on one frozen source. Record the rare component's narrow-axis variance and the arc errors on the same latent edges, alongside the transport plans, before the final window. An early coupling difference preceding a quality difference would support a training-path explanation. The current measurements leave that timing open. CIFAR remains unmeasured.

The probe used 32 consecutive real batches per task, protocol seed 0, separate diagnostic RNGs and zero optimizer updates. It verified each checkpoint hash, exact public-context restoration, and identical serialized trainer and named RNG state after evaluation. A derivative control detached only the fake feature path and checked the forward loss and cross-plan bit for bit. Two fresh processes reproduced the compact diagnostic byte for byte. The extra arc panel reused the same 69 distinct interiors across reference batches. These are endpoint diagnostics, with no additional qualification runs.

## Reproduction

For the matched study, use execution commit `27803af6ae036e91b7c7a627ddd03771a51a13eb`. The read-only probe accepts the original provenance checkpoint specified in [probe.json](probe.json). To reproduce the coupling probe, use the script from this report revision with the three original provenance checkpoints and the unequal-mass saved observations in their original campaign directories. Run from the repository root on CUDA with `PYTHONPATH=. python reports/forge/rhbi-texture-feature/retention_probe.py /path/to/runs/forge`. Each Forge worker writes `run.log` in its campaign/attempt directory. From the repository root:

```sh
python -m experiments.forge enqueue rhbi-v12-feature-transport --study rhbi-texture-feature-study
python -m experiments.forge enqueue rhbi-v12 --study rhbi-texture-feature-control
python -m experiments.forge drain --gpus 0,1 --campaign rhbi-texture-feature-v1
python -m experiments.forge logs --follow --task img_conv_texture32
```

After recording the texture readout, its full PASS admits the frozen retention study:

```sh
python -m experiments.forge run rhbi-v12-feature-transport --study rhbi-texture-retention-study --gpus 0,1
```
