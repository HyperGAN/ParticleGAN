# Proximal prior software verification

The spring's public optimizer and checkpoint contracts pass **315 focused
checks**. The short [actual-training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/prior-proximal-software/goal.gif) and [compact measurements](final-metrics.json)
illustrate scalar output CDFs and learned prior positions for the BCAP leader
and strength1 during genuine public `GANTrainer` updates.

This is an explicit software variant of the
[scalar stability question](../gaussian1d-diagnosis/README.md), with 64 updates,
32 initial MoG locations in two latent dimensions, fixed sigma .1, batch32,
width8/depth1 tanh networks and 512 evaluation samples every eight updates.
The target is `N(2, .5²)` throughout. There is no target shift or sustained
Tier 2 claim. Both arms use protocol seed0 and the public deterministic
initializer, isolated constructor/data/training/evaluation streams and the
complete BCAP leader recipe. Only strength changes.

The numerical software gate requires 64 restoring updates, exactly the
captured number of unique update-owned rows, positive measured restoration,
finite outputs and an exact checkpoint-resumed next update. Oracle quantiles
pass the scalar fidelity scorer; a shifted destructive control fails it. Final
CDF/moment fidelity is reported separately against the ordinary scalar bounds;
the shortened cohort supplies no qualification.

| Measurement | Leader strength0 score / target | Strength1 score / target |
| --- | --- | --- |
| Exact resumed next update | true / true | true / true |
| Finite sample fraction | 1 / 1 | 1 / 1 |
| Restoring updates | 0 / 0 | 64 / 64 |
| Restored unique rows | 0 / 0 | 1295 / 1295 captured owned rows |
| Maximum restoring displacement | 0 / 0 | .021318 / >0 |
| Target CDF KS | .419968 / ≤.05 **FAIL** | .211615 / ≤.05 **FAIL** |
| Mean error in target sigma | .734022 / ≤.2 **FAIL** | .033189 / ≤.2 |
| Output std / target std | 1.368524 / [.8, 1.2] **FAIL** | 1.534323 / [.8, 1.2] **FAIL** |

The software gate passes for both arms; the separate short-cohort fidelity
gate fails for both. The spring improves centering in this illustration and
widens the output law, which is a useful counterexample to treating reduced
prior motion as sufficient distribution fidelity.

Reproduce with an exclusive local output path:

```sh
PYTHONPATH=. python -m benchmarks.toy_audit.prior_proximal_smoke \
  --output runs/prior-proximal/software-reproduction
```

Logs, intermediate state, checkpoints and failed development probes remain
ignored under `runs/prior-proximal/`. Only this compact receipt and final GIF
are committed. The reproduction source is
[prior_proximal_smoke.py](https://github.com/255BITS/ParticleGAN/blob/a28c9631d59e2977616c0375e2a07c73171a9930/benchmarks/toy_audit/prior_proximal_smoke.py).

Disabled-feature parity was checked against base
`0337bbbc2300fd351e77e9a222b92401137fd421`: after the same 64-update software
protocol, every model tensor, optimizer packet, trainer RNG and named consumed
stream has the same content digest. Pending sampled ownership, reference
storage independence, magnitude-sensitive restoration and atomic rejection of
malformed checkpoints are covered by
[test_prior_proximal.py](https://github.com/255BITS/ParticleGAN/blob/a28c9631d59e2977616c0375e2a07c73171a9930/tests/test_prior_proximal.py).

See [mechanism and proposed bracket](https://github.com/255BITS/ParticleGAN/blob/a28c9631d59e2977616c0375e2a07c73171a9930/docs/prior-proximal.md).

Reproduction requires the [proximal feature's executed source](https://github.com/255BITS/ParticleGAN/tree/a28c9631d59e2977616c0375e2a07c73171a9930); this aggregate PR contains reports and saved evidence. [The completed search](../bcap-prior-search/proximal/README.md) reports the ordinary task gates.
