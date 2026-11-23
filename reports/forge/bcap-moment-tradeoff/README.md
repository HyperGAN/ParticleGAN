# Covariance tether: sustained modes versus word retention

The `.01` initial mean/covariance tether improves sustained mode quality, while
word hold exposes **two different transient errors**: lost prior-generated
coverage, then weak paired reconstruction. Saved states show that the same
scalar penalty has a much larger relative gradient contribution in the word
task. This supports investigating the tether's force relative to the current
adversarial gradient; it does not establish the historical cause of either
word failure.

This is a **saved-state diagnostic**, with zero optimizer updates, zero new
sampling draws, and no new gate credit. It uses the original covariance
execution snapshot, checks checkpoint/receipt hashes and initializer/prior
bindings, and preserves every published result. The BCAP leader remains the
baseline and the selected global recipe. [Original comparison](../bcap-prior-search/findings.md),
[full original witnesses](../bcap-prior-search/covariance/repair-witness.json),
and [diagnostic measurements/provenance](results.json) remain separate.

## What actually changed

| Original measurement | Leader score / target | Tether `.01` score / target |
| --- | ---: | ---: |
| Mode hold: final passing suffix | 3 / ≥5 | 7 / ≥5 |
| Mode hold: modes at update 1,050 | 8 / ≥8 | 8 / ≥8 |
| Mode hold: quality at update 1,050 | 0.86938477 / ≥0.9 | 0.99804688 / ≥0.9 |
| Mode hold: minimum quality over final five observations | 0.86938477 / ≥0.9 | 0.98486328 / ≥0.9 |
| Word hold: mandatory primary checks | 25 / ==25 | 23 / ==25 |
| Word hold: mandatory confirmation checks | 25 / ==25 | 23 / ==25 |

The repaired mode gate concerns sample quality, with all eight modes present
at the discriminating observation. Both recipes pass at the endpoint. The
tether also has an earlier quality dip at update 850, so this evidence supports
the declared terminal suffix rather than never leaving the goal.

For words, both holds restore their own earliest certified acquisition state:
leader update 834 and tether update 4,167, followed by the same 4,000 new
updates. They have different trained starting weights and consumed prefixes.

| Original tether observation | Primary score / target | Confirmation score / target |
| --- | ---: | ---: |
| Update 5,167: generated quality | 1 / ≥0.95 | 1 / ≥0.95 |
| Update 5,167: generated modes | 4 / ==5 | 4 / ==5 |
| Update 5,167: mass TV | 0.203515625 / ≤0.1 | 0.2046875 / ≤0.1 |
| Update 5,167: paired minimum token probability | 0.99998224 / ≥0.9 | 0.99998224 / ≥0.9 |
| Update 6,167: generated modes | 5 / ==5 | 5 / ==5 |
| Update 6,167: mass TV | 0.02578125 / ≤0.1 | 0.0251953125 / ≤0.1 |
| Update 6,167: paired minimum token probability | 0.79676759 / ≥0.9 | 0.79676759 / ≥0.9 |

At update 5,167, generated `melon_` mass is zero and `lemon_` mass is
0.388671875 in the primary draw. Every generated sample is an accepted word,
and `G(E(melon_))` still reconstructs correctly with high confidence. The
missing word therefore remains representable through the encoder, while
generation from the current prior misses it. This identifies a discrepancy
between the two paths without attributing it to a particular latent row.

At update 6,167, all five generated words return, but the fourth character
(`l`, zero-based position 3) of reconstructed `apple_` has probability
0.7967675924301147 in both cohorts. Its argmax is still correct. Coverage and
argmax inspection would conceal this numerical inverse failure. Both hold
endpoints recover.

## Relative force differs across tasks and states

The six analyzed snapshots are the two final mode states, plus each word
recipe's own acquisition and final state. Word diagnostics enumerate all
five zero-width prior rows. Mode diagnostics enumerate all twelve rows and
fixed symmetric axis offsets at the saved intrinsic sigma `.025`; these
offsets are **not Gaussian samples**. Models run on CPU with deterministic
settings, frozen weights, and isolated/restored constructor RNG. These are
diagnostic panels, not historical training batches or comparable gate cohorts.

Every measurement below has **no qualification target**. The force ratio is
`||.01 ∇R|| / ||∇L_adversarial||` over raw prior locations. The displacement
change uses the hypothetical zero-momentum smoothed DualNorm prior step
before direction blending; no optimizer step is executed.

| Saved state | Step | Relative tether gradient | Relative prior displacement change | Target |
| --- | ---: | ---: | ---: | --- |
| Tether mode endpoint | 1,200 | 0.0108734 | 0.00566339 | Diagnostic only |
| Tether word acquisition | 4,167 | 0.0747697 | 0.0795912 | Diagnostic only |
| Tether word endpoint | 8,167 | 0.341457 | 0.340552 | Diagnostic only |
| Leader mode endpoint, counterfactual tether | 1,200 | 0.0111496 | 0.0356341 | Diagnostic only |
| Leader word acquisition, counterfactual tether | 834 | 0.728262 | 0.787946 | Diagnostic only |
| Leader word endpoint, counterfactual tether | 4,834 | 8.97585 | 4.37085 | Diagnostic only |

Within the trained tether word trajectory, adversarial prior gradient norm
falls from 0.00614982 to 0.000234143, whereas weighted tether gradient norm
falls from 0.000459820 to 0.0000799500. The tether gets smaller in absolute
terms but larger relative to the adversarial signal. Small moment discrepancy
does not imply negligible influence on the update direction.

At the tether word endpoint, the direct latent-input path through the joint
critic has gradient norm 0.000234294, versus 0.000000581441 through `G(z)`.
The joint critic evaluates `(G(z), z)`, so it can still push prior locations
when the decoder's categorical outputs are saturated. The two differentiated
paths add back to the complete prior gradient. This observation is consistent
with the earlier [saved word-dynamics investigation](../fivewords-dynamics/README.md),
which independently found inverse instability and joint-surrogate/output-goal
tension under its own source and protocol.

All four current final states have **zero** cumulative direction-blend
conflicts, blends and stalls. The two word acquisition states also have zero.
Thus direction blending did not intervene in these original runs. At the
actual tether snapshots, this fixed-panel prior-only calculation has zero
rows whose combined direction is uphill for the adversarial prior objective;
a simple observed sign reversal is not established either.

## Next investigation

The supported first hypothesis is a **globally configured relative-force limit
on the covariance tether**, rather than another unqualified increase in its
scalar coefficient. Test whether retaining the mode repair while bounding
tether influence as the adversarial prior signal shrinks recovers word hold.
Record actual adversarial/tether gradients and applied row displacements around
the discriminating windows, including the generator and encoder alignment.
The force limit itself remains an unexecuted hypothesis.

Keep the six Tier 1 prerequisites, original word hold, trajectory and residual
passes as required controls, and ultimately preserve every leader Tier 2 pass.
Use one global recipe, seed 0, and the same task conditions/budgets. A common
trained starting state or common-clock word test would require an explicitly
separate cohort and could not replace the original own-prefix grades.

There are no checkpoints at mode update 1,050 or word updates 5,167/6,167.
Word scored output tensors survive; mode sample tensors do not. Consequently,
this investigation cannot measure historical force at those exact failures or
separate prior movement from generator/encoder movement there. A future bounded
comparison needs those measurements to test causation. It may also find that
inverse protection is needed, as suggested by the earlier word-dynamics work.

The sole current [leaderboard](../technique-inventory.md) and all previous
qualification counts remain unchanged. Actual-training illustrations from the
original runs are available for [mode hold](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-prior-search/covariance/media/configuration-ec49f810ceed/mode_hold.gif)
and [word hold](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-prior-search/covariance/media/configuration-ec49f810ceed/five_word_joint_hold.gif).

## Reproduce

Run from the repository root with the original local archive present:

```sh
python -u reports/forge/bcap-moment-tradeoff/diagnose.py \
  --source /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/covariance/queue/snapshots/c63305181b75bbfae337143b6836397aa998cadbf172581859031a01c90892c2 \
  --evidence reports/forge/bcap-prior-search/covariance/repair-witness.json \
  --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/covariance \
  --output reports/forge/bcap-moment-tradeoff/results.json
```

Raw execution logs stay outside Git at
`/mnt/ml7tb/ParticleGAN-forge/bcap-moment-tradeoff-20261011/diagnostic-verified.log`.
Use `tail -F` on that file. Hash/state checks, gradient-path additivity, and
unchanged diagnostic model/checkpoint state assertions run in the reproducer.
