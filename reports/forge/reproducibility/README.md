# Public-API reproducibility audit

New comparisons use seed `0`, the repository initializer, fixed task conditions and isolated data streams. This bounded software control compares the same vector task under K3P and KA2, and repeats K3P with the same seed.

Budget: three CPU runs, 8 updates each, one Torch thread, 20 seconds per run. Each numerical bound was zero mismatches before execution. No full distribution-quality qualification is claimed.

| Check | Mismatches | Bound | Result |
| --- | ---: | ---: | --- |
| batch sequence mismatches | 0 | 0 | PASS |
| cross trainer initial model mismatches | 0 | 0 | PASS |
| repeat final state mismatches | 0 | 0 | PASS |
| repeat initial state mismatches | 0 | 0 | PASS |

Reproducibility verdict: **PASS**. The short runs remain FAIL for their unchanged full training-quality gates.

Actual target/output training GIFs: [K3P](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/reproducibility/k3p.gif), [KA2](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/reproducibility/ka2.gif). [Final metrics and source/runtime/artifact hashes](final-metrics.json) bind the measured execution. Raw states, arrays and progress receipts remain in the local run directory.

Use one complete global trainer configuration across the task ladder; hold each task's architecture, target law, batch size, initialization, prior, sampling and budget fixed across candidates. Fixed identity/zero or stored-weight controls and named initialization diagnostics remain explicit separate cohorts. Archived results retain their original sources and are not regraded. The [current solution leaderboard](../technique-inventory.md) retains its recorded qualifications; this control selects no winning trainer.

Regenerate this readout without training:

```sh
python -m benchmarks.toy_audit.reproducibility_audit
```

Reproduce the bounded public-API proof in a fresh local directory:

```sh
python -u -m benchmarks.toy_audit.reproducibility_audit --run --output runs/reproducibility-proof
```
