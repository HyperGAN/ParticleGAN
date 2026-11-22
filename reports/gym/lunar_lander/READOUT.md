# Lunar Lander: first bounded baseline round

The experiment is implemented and benchmarked. **The three-generator GAN helps
relative to the reconstruction-only graph, but does not yet beat persistence.**
The direct supervised predictor is the conditional-prediction baseline to beat.
These results do not establish a useful model-based landing controller.

## Conditional leaderboard

All selected checkpoints are at update 1,000, chosen by the same validation
continuous MSE criterion. The test set contains 8,192 transitions from 19 unseen
episodes/terrain instances. Lower errors are better.

| Model, validation-selected checkpoint | Test next-state MSE | Contact Brier | Matched-action effect MSE | 20-step MSE |
| --- | ---: | ---: | ---: | ---: |
| Direct supervised | **0.007048** | **0.004498** | **0.003010** | **0.330456** |
| Persistence | 0.011567 | 0.005249 | 0.012995 | 0.552613 |
| Three Gs + E + joint/marginal Ds | 0.063822 | 0.007322 | 0.039382 | 0.520222 |
| Three Gs + E, reconstruction only | 0.082694 | 0.004892 | 0.039649 | 0.640620 |

The direct model improves next-state MSE by 39.1% over persistence. The GAN
improves 22.8% over reconstruction only, but has 5.52 times persistence's error.
Its matched-action response is also worse than predicting no response.
At twenty steps the GAN narrowly beats persistence on the fixed scenes; that
does not overturn its poor one-step and action-response results.

The [complete board](baseline/README.md) also reports all final checkpoints.
The direct final checkpoint scores **0.004820** test MSE, 58.3% below persistence,
even though its validation MSE worsened from 0.010901 to 0.016149. This is a
reported endpoint, not a replacement selected using test performance. The
GAN final and reconstruction final deteriorate to 0.204280 and 0.487572 test MSE.
Validation/test disagreement deserves attention in later comparisons; keep the
frozen selection rule and episode splits.

Continuous errors average six standardized fields. Contacts are scored separately;
fewer than one sixth of test transitions are contact transitions. For example,
the selected direct model's MSE is 0.002592 in flight, 0.002228 on approach, and
0.032916 at contact. The selected GAN is 0.074728 / 0.043478 / 0.074706 respectively.
Contact alone therefore does not explain the GAN's deficit.

Recursive evaluation uses 24 fixed behavior-phase anchors, with 24 references
surviving through 20 steps and 22 through 50. These deliberately selected scenes
are not the same sampling distribution as the primary one-step test set.
The predictor never receives refreshed real states; contacts feed back at
`p >= .5`; evaluation stops at the reference episode end.

## Joint generation does benefit from the discriminators

At the validation-selected checkpoints, equally weighted held-out terrain contexts:

| Model | Prior joint SW1 ↓ | Prior coverage ↑ | Composed SW1 ↓ | Composed coverage ↑ |
| --- | ---: | ---: | ---: | ---: |
| Three-G GAN | **0.390665** | **34.49%** | **0.390128** | **33.39%** |
| Reconstruction only | 0.438917 | 2.74% | 0.424576 | 3.72% |
| Reference halves | 0.084006 | 94.77% | — | — |

The GAN improves prior SW1 by 11.0%, and its contact-pattern total variation is
0.135 versus 0.303 for reconstruction only. Both remain far from the empirical
reference comparison. The reference halves have fewer samples and may share
anchor siblings; they characterize reference variability rather than a proven
independent-sample floor. Coverage is a reference-neighbor-radius metric, not
the percentage of all physical lander states learned.

By 10,000 updates, reconstruction-only prior SW1 explodes to 40.59 with zero
coverage. The GAN final stays at 0.4324, but also worsens compared with its
selected checkpoint. The adversarial objectives constrain generated observations;
that benefit has not produced competitive conditional predictions.

## A concrete failure mechanism to investigate

The [training/validation diagnostics](diagnostics.json) separate generation
instability from generalization. Reconstruction-only generated normalized state
RMS grows from approximately 0.8 at update 1,000 to 177 at 10,000. Its training
next-state error also worsens, so this is more than ordinary held-out overfitting.
On test inputs, 89.7% of final encoder offset coordinates sit within 5% of their
bound, versus 0.43% at the selected checkpoint. Only 12 components are then used,
versus 91 earlier. These are correlated symptoms, not independent proofs of cause.

The inherited synthetic loss has **detached targets and live encoder inputs**.
Writing the input reconstruction as `F(x)`, its source gradient is
`2 J_F(x)^T (F(x) - x)` because the target is detached. For the simple scalar
case `F(x) = c*x`, `0 < c < 1`, gradient descent pushes `x` outward, even while
the actual reconstruction error grows. This shows the objective can permit
positive feedback; it does not prove that this is the sole mechanism here.
The GAN's distribution constraints are consistent with restraining that drift.

**Recommended next comparison:** detach the synthetic encoder input for the
synthetic reconstruction loss only. Keep its targets detached, keep real triple
reconstruction unchanged, and keep the original/composed adversarial gradients
live. Run the reconstruction and GAN variants on the same finite dataset and
budget. This preserves `G1 -> st`, `G2 -> at`, `G3 -> st+1` and isolates the
suspected feedback path. It requires a distinct reconstruction forward pass so
the adversarial composition does not accidentally lose its input gradients.
No such variant was trained in this baseline round.

After stability improves, use matched-action and held-out conditional errors to
guide further routing or loss changes. Increasing training duration is not
supported by this round. A landing planner should wait for substantially better
command response and recursive prediction.

## Protocol, cost, and artifacts

- Finite data: 32,768 / 4,096 / 8,192 train/validation/test triples; four commands
  per replayed anchor; 75 / 16 / 19 source episodes, with disjoint terrain.
- Collection: 31.0 seconds, 3,397,947 explicit simulator step calls; 3,431,849
  including the simulator's internal reset steps. Source/replay details are in
  [dataset metadata](dataset_metadata.json).
- Each learned arm: 10,000 updates, batch 256, same training-record stream for
  generator/predictor updates. MoG1024, z32, G/E width128, bcap, Rp logistic,
  EMA, shared role-conditioned state D, and action/joint Ds for the full model.
- Training on GPU 1: direct **9.7s**, reconstruction **89.9s**, GAN **301.9s**.
  These exclude checkpoint evaluation/save intervals. The GAN has extra D draws
  and computation; this is not a matched-compute comparison.
- Inference dependency parameters: direct 99,940, E+G3+prior 100,040. Full
  G/E/prior 145,618, plus 210,307 discriminator parameters for the GAN.
  Current inference executes all G branches; measured throughput includes that
  overhead. Parameters and optimizer draw counts are preserved in the board.
- Simulator: Gymnasium 1.2.3, Box2D 2.3.10, continuous Lunar Lander, wind off.
  Eleven terrain heights are privileged context. Random engine dispersion and
  hidden articulated/contact state remain.
- [Simulator probes](simulator_probe.json): exact complete replay and engine
  threshold checks pass. Engine-only outcome variance rises near contact;
  the measured variance is not a Bayes-error floor.
- Verification: 312 CPU tests passed, four opt-in CUDA tests skipped, plus 27
  passing subtests; the GPU 1 training smoke passed. Independent checkpoint
  replay matches validation metrics within 1.2e-7 across CPU/GPU.

Open the [offline demo](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/gym/lunar_lander/demo/index.html) or download the
[demo bundle](demo/lunar_lander_demo.zip). It shows fixed recorded engine
counterfactuals, one-step/recursive comparisons, and prior/composed transitions,
including failures. These are checkpoint predictions and reference-controller
actions, not a learned landing policy. Numerical data, checkpoint/source hashes,
and definitions are preserved in [leaderboard.json](baseline/leaderboard.json)
and [protocol.json](baseline/protocol.json).
