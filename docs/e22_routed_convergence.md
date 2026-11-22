# A learned-game convergence diagnostic for routed adapters

This toy asks whether the remaining Supra convergence gap appears when ordinary
LoRA and gated particle LoRA learn the same exactly reachable target through a
learned conditional critic. It is based on merged `develop` at
`6ec7e5788e14ea15ddc3e16ac71110458108b6a6`, including #155 and #223.
The fixed protocol is [e22_routed_convergence_v1.json](e22_routed_convergence_v1.json).

The separate [PR #224](https://github.com/HyperGAN/ParticleGAN/pull/224) isolates
an unsafe generator learning-rate release using a fixed quadratic critic. That
counterexample does not reproduce the whole learned-game convergence problem.
This fixture uses a trained critic, two sequential token-routing sites, a shared
128-by-4 bank/controller, source/time conditioning, a frozen BF16 host and FP32
adapter branches. It preserves the native paired RpGAN, KA2 token penalty,
per-site DV12, full-model row proposals and zero learned-feature-harm guards.
Output-error guards are disabled.

Both adapter families can exactly realize the teacher. The teacher is an
ordinary rank-two adapter whose down basis matches the students' initial down
basis. Copying its down/up factors and zeroing the particle bridge supplies an
explicit particle witness. Complete outputs must agree on all fit, guard and
test contexts before training. This is a controlled LoRA-generated target
family; it does not establish general particle inferiority or reproduce the
full Supra backbone.

The three fixed comparisons are ordinary LoRA through the native game, full
gated particle LoRA through the native game, and historical ordinary LoRA through
MSE plus AdamW. Only the separately labeled historical reference uses MSE as a
training objective. No particle optimizer, structural decision, checkpoint
selection or stopping rule uses output error.

The ordinary native arm uses the supported public policy with a frozen prior
scaffold, no table/router optimizer groups and both row controls disabled.
Owner-dependent critic floors and learned-noise eligibility can therefore
differ from the particle arm. The comparison measures complete configurations;
it does not attribute a difference solely to the particle formula. Receipts
must show actual owner counts, applied rates and noise observations.

Network initialization uses the public `particlegan.init.initialize_` sampling
method and distinct named CPU generators. Shared down/up factors and the
critic start bitwise equal across native arms. Each up factor is then zeroed,
giving identical fresh base outputs. The bank separately uses the public R2
initializer. This named sampled network cohort differs from the earlier Supra
deterministic-orthogonal network cohort.

Each arm has exactly 6,400 editing-only updates of four contexts, with fixed
checkpoints every 200 updates. Both native critics at updates 800 and 6,400
are mandatory common judges: every saved generator is scored under all four,
with private paired draws. Report the zero-residual teacher and fresh-model
anchors under each judge. Changing critics during training makes each arm's
own loss unsuitable as a shared quality ranking.

The particle-versus-ordinary gap is reproduced only if both endpoint critics
rank ordinary ahead by more than `1e-4` in mean generator game loss at the fixed
6,400 endpoint. Output RMSE remains a
diagnostic. A reproduced gap still needs a separately declared intervention
before claiming a cause or changing defaults.

This is a standalone policy-aware diagnostic with a declared CPU budget of
900 seconds per arm and 2,700 seconds total. It earns no Forge screen,
calibration, default-adoption or Supra quality credit. Timeouts are incomplete
executions, not scientific outcomes. Bulk logs, per-update JSONL and native
checkpoints remain outside Git; reproduction sources and compact receipts are
tracked.

Run the fixed comparison and the separately declared initialization diagnostic:

```sh
PYTHONPATH=. python -u examples/run_e22_routed_convergence.py --out runs/routed-convergence-v1
PYTHONPATH=. python -u examples/e22_routed_convergence_neutral.py --parent runs/routed-convergence-v1 --out runs/routed-convergence-neutral-v1
```

The native package must match the frozen base; a changed native formulation needs
its own diagnostic revision. Source and card hashes are checked during execution.
To tail a concise view of the baseline's full control log:

```sh
tail -F runs/routed-convergence-v1/run.log | jq -c '{event,arm,step,loss_g,loss_d_game,test_game}'
```

The [compact baseline receipt](e22_routed_convergence_results.json) reproduces
the convergence gap. At the fixed 6,400 endpoint, held-out paired game scores
are:

| Training configuration | Ordinary endpoint critic | Particle endpoint critic |
| --- | ---: | ---: |
| Ordinary LoRA, native game | 0.923587 | 0.930792 |
| Gated particles, native game | 1.504398 | 2.304944 |
| Ordinary LoRA, historical MSE/AdamW reference | 0.819877 | 0.831208 |
| Gated particles, initial H/b zero | 0.802713 | 0.800708 |

Lower is better. Both other mandatory critics agree on this endpoint ranking.
The zero-residual teacher anchor is `log(2)`. These are shared learned-game
scores, not output RMSE. Zeroing the trained particle codes worsens the particle
endpoint by `0.130728` and `0.298169` under the two endpoint critics, confirming
that the retained bank contributes on this fixture.

Independent CPU review qualified all 102 saved checkpoints, all 99 four-judge
curve rows and six actual recovery updates. Noise stayed at `0.125`, generator
rate scales stayed at one, and there were no surprise fires or accepted moves.
The particle critic's raw rate stayed at one, so its table-support floor never
bound. That floor and a change in learned noise were inactive in this run;
the full configurations and their feedback can still differ.
The router temporarily contracted to `0.5` and returned to one.

The new intervention has its own frozen
[card](e22_routed_convergence_neutral_v1.json). It zeros only the additive hidden
bridge `H` and bias `b` before policy/EMA construction. The sampled particle
matrix `C`, shared bank, routing, all trainable owners and all native controls
remain. It uses the exact parent teacher, batches, Gaussian panels and four
learned judges, without altering the held parent sources.

The optional zero-update basis diagnostic is reproducible separately:

```sh
PYTHONPATH=. python -u examples/diagnose_e22_routed_convergence_basis.py --parent runs/routed-convergence-v1 --out runs/routed-basis-audit-v1
```

It measures feature-basis recovery and autograd output tangents without optimizer
updates. Every tested tangent descends its own learned game. The measured basis
distortion supports an acquisition hypothesis, not a wrong-sign autograd claim;
BF16's actual forward is a staircase while these probes measure its declared
autograd tangent.
The [compact basis receipt](e22_routed_convergence_basis_results.json) distinguishes
the original executed scratch tools from the packaged standalone reproducer;
the latter received source, syntax and CLI checks without an unchanged rerun.

The intervention closes the reproduced gap. Its fixed 6,400 endpoint improves
the particle score under every original learned judge:

| Configuration | Ordinary D800 | Ordinary D6400 | Particle D800 | Particle D6400 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary native game | 0.719085 | 0.923587 | 0.725256 | 0.930792 |
| Original particles | 1.022885 | 1.504398 | 1.185411 | 2.304944 |
| Initial H/b zero, particles retained | 0.702292 | 0.802713 | 0.704504 | 0.800708 |
| Historical ordinary MSE reference | 0.709138 | 0.819877 | 0.709846 | 0.831208 |

Zeroing initial H/b before policy construction preserves the common fresh
outputs because both up factors start at zero, but changes the features those
up factors first learn from. The original hidden modulation has RMS 0.777 and
0.871 times the hidden features at the two sites. Recovering the ordinary hidden
basis from the up-factor inputs has relative residuals 0.534 and 0.468; the
neutral initialization reduces these to 0.067 and 0.112. The function-space
gradient tangent also aligns more closely with ordinary LoRA. Combined with
the separately declared training intervention, this isolates the initial H/b
choice as a cause of the gap in this teacher-aligned family. The basis evidence
supports feature distortion as the mechanism; it does not uniquely establish
mediation or separate H from b.

H, b and C remain trainable. Bank and router gradients are live on 6,399 updates,
the final C norms are 0.639 and 0.708, and zeroing particle codes worsens held-out
game scores under all four judges by 0.756, 0.585, 1.185 and 0.930. The improved
adapter continues to use the shared particle bank. Neither particle run accepted
a structural move. This PR supplies a native learned-game reproducer and an
initialization control; it changes no ParticleGAN optimizer or default recipe.

Output RMSE is diagnostic: neutral particles reach 0.034320, ordinary native
LoRA 0.047493, original particles 0.135631 and the historical MSE reference
0.031274. The historical reference still has the lowest output RMSE. The toy
does not establish a full-Supra improvement or performance across target
families, initialization cohorts or seeds.

The original neutral training completed all 6,400 updates and exact 800-to-802
recovery, then its offline scorer rejected an unresolved step-zero checkpoint
loaded into an already-resolved policy. The preserved original source is the
first commit's neutral helper. The final helper creates a fresh policy before
scoring; this is its sole source change. An evaluation-only recovery rescored
the original saved states without retraining, preserved the error receipt and
bound the original training source separately from the scoring repair. Its
total training plus scoring time was 204.095 seconds, within the declared
900-second intervention budget. See the
[compact intervention receipt](e22_routed_convergence_neutral_results.json)
for source identities, fixed endpoint scores and particle checks.

Independent review verified all 34 intervention states, all 33 four-judge
curves, unchanged original artifacts and two further exact recovery updates:
61,606 checks. Training, recovered evaluation and independent review together
took 227.09 seconds. The default software tests cover exact teacher reachability,
public initialization, retained trainable particle owners, stream isolation,
checkpoint identity and the fresh-policy scoring repair.

The full learned convergence regression is opt-in:

```sh
PARTICLEGAN_RUN_ROUTED_CONVERGENCE_LONG=1 PYTHONPATH=. python -m pytest -q tests/test_e22_routed_convergence_long.py
```

It trains fresh fixed-horizon comparisons in a temporary directory, restores
actual generators and all four saved critic weights, and independently checks
the declared learned-game improvement and particle-contribution gates. It
cannot pass from stored receipt verdicts. Default test runs skip this longer
experiment. The assertion was checked against the qualified artifacts without
repeating unchanged training; the complete final opt-in subprocess path was
not rerun after the one-line scoring repair.
