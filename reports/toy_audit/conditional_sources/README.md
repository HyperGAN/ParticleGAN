# Conditional source-family coverage

Four formerly source-only entries now have one bounded exact-source attempt each. Scientific sources/configurations come from develop `6ec7e5788e14ea15ddc3e16ac71110458108b6a6`; CPU uses one thread and each entry has a hard 120-second cap including its software parity probes. Original catalog source-only receipts and quality ratings remain unchanged. No public library, recipe/configuration, batch, seed, initializer or original update budget was repaired or reduced.

| Entry | Original budget | Execution / completed pairs | Original source gate | Added live / EMA | Training GIF |
| --- | --- | --- | --- | --- | --- |
| Sparse mixed identity symbols | 5,000 | TIMEOUT / ≥3,236 | INCOMPLETE | INCOMPLETE / INCOMPLETE | [actual checkpoints](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/conditional_sources/media/source-family-00.gif) |
| Sparse mixed split symbols | 5,000 | TIMEOUT / ≥2,099 | INCOMPLETE | INCOMPLETE / INCOMPLETE | [actual checkpoints](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/conditional_sources/media/source-family-01.gif) |
| Analytic denoising grid, one class | 28,000 | BLOCKED / ≥0 | NO_FROZEN_GATE | BLOCKED / BLOCKED | none: source prerequisite blocks training |
| Analytic denoising grid, four classes | 7,000 | BLOCKED / ≥0 | NO_FROZEN_GATE | BLOCKED / BLOCKED | none: source prerequisite blocks training |

## What these definitions verify

Sparse identity asks for eight class-conditional distributions over 64 real modes in 24 coordinates, three active coordinates per mode with Gaussian noise, exact inactive zeros, and the class's deterministic symbol. Split replaces that symbol law with a balanced two-symbol law in each class. Continuous mode and symbol must describe the same draw. The source's convergence bar checks coverage/HQ/class/symbol and near-zero sparsity; it does not require exact zeros, active Gaussian shape or the full split-symbol mass. The new separately frozen gate adds those checks, including a reject bin in each class's joint quality mass and fixed radial/projected CDF checks on active standardized residuals. The residual-shape checks pool assigned modes; they do not independently certify every mode's covariance.

Denoising one/four classes asks for repeated clean draws from the analytic multimodal `q(x0 | xt,c)` at fixed observations and diffusion times. Four classes select a checkerboard subset of 25 modes. Recovering just a posterior mean, ignoring the noisy observation, or sampling a different class is insufficient. The original source trains Gaussian reverse transitions and reports posterior sliced-W1 diagnostics, but declares no single scientific acceptance gate. Analytic controls support the added definition; they are not trained model evidence or a substitute for the unavailable GPU execution.

The versioned evaluator has 17 exact-law/control results: exact oracle samples pass, while tiny inactive smear, zero-width centres, wrong requested class, single-symbol collapse, incoherent balanced symbols, posterior-mean collapse and ignored observations are rejected where applicable. Tiny smear and zero-width centres both pass the original sparse bar on the same cloud, demonstrating its specific blind spots.

## Source-bound outcomes

### Sparse mixed identity symbols

Original source: `experiments/train_sparse.py`; config: `source DEFAULTS`. Seed 1, batch 256, full budget 5,000.

Preserved terminal error: `WallTimeout: The fixed 120-second conditional-source allowance expired`.

Last source frame: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-00/source/experiments/train_sparse.py:377` in `train`.

Full external log: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-00/execution.log` (SHA-256 `ff7ca6bcc2cdd8e966ef74e75ae14cdbc25cfebf84f2222c8a9be3f1d2d5de44`).

Last scored update 3200 live: 0/64 HQ modes, HQ 0.0000, worst conditional joint TV 1.0000, exact inactive zeros 0.0000, maximum symbol TV 1.0000. This partial checkpoint cannot certify the original full-budget gate.
Last scored update 3200 ema: 0/64 HQ modes, HQ 0.0000, worst conditional joint TV 1.0000, exact inactive zeros 0.0000, maximum symbol TV 1.0000. This partial checkpoint cannot certify the original full-budget gate.

The effective recipe, actual prior type/buffers, original initializer and optimizer group overrides are recorded alongside source/config hashes. Live and EMA reads use the original `sample_z`/`draw_fakes` closures with separate evaluation RNGs. Exact short-prefix parity compares model/optimizer/gradient/input tensors, module modes, requires-grad flags and owned/global RNG states before the full attempt.

### Sparse mixed split symbols

Original source: `experiments/train_sparse.py`; config: `configs/sparse/discrete/gst_split_s1.yaml`. Seed 1, batch 256, full budget 5,000.

Preserved terminal error: `WallTimeout: The fixed 120-second conditional-source allowance expired`.

Last source frame: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-01/source/lib/toy_metrics.py:100` in `sliced_w1`.

Full external log: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-01/execution.log` (SHA-256 `d385265a7333bc3e9f42b4ef670ce44bdad706ccc5e39c4e183ac95c8710cdb3`).

Last scored update 2000 live: 25/64 HQ modes, HQ 0.2815, worst conditional joint TV 0.8203, exact inactive zeros 0.0000, maximum symbol TV 0.1523. This partial checkpoint cannot certify the original full-budget gate.
Last scored update 2000 ema: 22/64 HQ modes, HQ 0.2424, worst conditional joint TV 0.8594, exact inactive zeros 0.0000, maximum symbol TV 0.1523. This partial checkpoint cannot certify the original full-budget gate.

The effective recipe, actual prior type/buffers, original initializer and optimizer group overrides are recorded alongside source/config hashes. Live and EMA reads use the original `sample_z`/`draw_fakes` closures with separate evaluation RNGs. Exact short-prefix parity compares model/optimizer/gradient/input tensors, module modes, requires-grad flags and owned/global RNG states before the full attempt.

### Analytic denoising grid, one class

Original source: `experiments/train_denoising.py`; config: `configs/denoising/diagnostics/ddgan_class_free_28k_s24002.yaml`. Seed 24002, batch 256, full budget 28,000.

Preserved terminal error: `RuntimeError: This experiment requires a GPU; CUDA is unavailable`.

Last source frame: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-02/source/experiments/train_denoising.py:178` in `train`.

Full external log: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-02/execution.log` (SHA-256 `e47b8235b71653343d404bb6cf87fe5e57b667b7cc830695a59d8c7ec83699ef`).

### Analytic denoising grid, four classes

Original source: `experiments/train_denoising.py`; config: `configs/denoising/default.toml`. Seed 24002, batch 2048, full budget 7,000.

Preserved terminal error: `RuntimeError: This experiment requires a GPU; CUDA is unavailable`.

Last source frame: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-03/source/experiments/train_denoising.py:178` in `train`.

Full external log: `/ml2/hypergan/toy-conditional-sources-20261001/source-family-03/execution.log` (SHA-256 `571fd79f35c3a1ef0a179efea488d610cfb3c1972a0bb82173c0ccc761f8fa52`).

The identity prefix has no HQ modes and severe joint/support defects at its last frame. The split prefix has much stronger class/symbol correspondence but lacks HQ coverage, leaks onto inactive coordinates and has overly broad active residuals. Both use the source's linear real head, which provides no structural exact-zero mask. These are measured prefix defects and a source-level limitation; neither capped prefix establishes the unknown 5,000-update scientific outcome or isolates an optimizer cause.

## Reproduction and receipts

`coverage.json` retains all four entries, terminal statuses, protocol/effective settings, source hashes, actual frame steps, known costs and original errors. Bulk source snapshots, checkpoints and tail-able logs stay outside Git under `/ml2/hypergan/toy-conditional-sources-20261001`. No retry or failed-prerequisite extension was performed. A complete original budget and five passing terminal observations are required for an added-gate PASS; blocked or partial runs retain their required denominator.

```sh
python -m benchmarks.toy_audit.source_conditional_capture \
  --case source-family-00 --output /ml2/hypergan/new-conditional-case

python -m benchmarks.toy_audit.source_conditional_report \
  --artifacts /ml2/hypergan/toy-conditional-sources-20261001 \
  --output reports/toy_audit/conditional_sources
```
