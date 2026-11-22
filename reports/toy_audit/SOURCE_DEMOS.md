# Original source-demo convergence evidence

These two entries were source-only in the original 109-case review. This addendum runs the unaltered examples and package from develop `6ec7e5788e14ea15ddc3e16ac71110458108b6a6`, on CPU with one thread. Existing source-only receipts and quality ratings remain unchanged. No recipe, library or configuration was repaired. The strengthened evaluator is pinned separately; these are new diagnostic results, not Atlas or Forge qualification.

| Example | Declared / actual full budget | Completed pairs / last scored | Live / EMA diagnostic | GIF |
| --- | --- | --- | --- | --- |
| Five-word latent autoencoder demo | 20,000 / 20,001 | 20,001 / 20001 | full-budget live PASS / EMA PASS | [actual checkpoints](media/source-five-modes.gif) |
| Single Gaussian quickstart: original batch2048 | 1,000 / 1,000 | ≥18 / 1 | TIMEOUT: incomplete evidence | [actual checkpoints](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/source-quickstart.gif) |
| Single Gaussian quickstart: separate CPU128 profile | 1,000 / 1,000 | 1,000 / 1000 | full-budget live FAIL / EMA FAIL | [actual checkpoints](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/media/source-quickstart-cpu128.gif) |

## Five-word interpretation

The original BiGAN-style joint critic loop has `range(total_steps + 1)`: its default 20,000 means **20,001 D/GE update pairs**, and dashboard step zero is already after the first update. The source dashboard displays EMA reconstructions with hard decoding and strips padding. The audit samples the actual prior distribution for both clean/live and EMA cohorts, while separately reconstructing each canonical input in its correct paired row. All six tokens, including underscore padding, enter the confidence and NLL measures. It tests five-word mass, confidently decoded generated words and paired reconstruction; it does not test language generation or held-out word/typo generalization.

Final observed live: 5/5 valid modes, confident-word fraction 1.000000, word/reject TV 0.012256, paired reconstruction exact=True, minimum correct-token probability 0.999938; 59 consecutive passing terminal observations.
Final observed ema: 5/5 valid modes, confident-word fraction 1.000000, word/reject TV 0.012256, paired reconstruction exact=True, minimum correct-token probability 0.999936; 58 consecutive passing terminal observations.

## Gaussian interpretation

The quickstart targets `N((1,1), .04 I)` and has an original 1,000-update budget. The new gate tests standardized mean, covariance eigenvalues, radial distribution and sixteen fixed projected CDFs; a point at the mean or an equal-covariance circle cannot pass. Clean live sampling is the original example's law. Clean EMA is an additional separate diagnostic, and cannot replace a failed or incomplete live run.

Last scored live at update 1: mean error 6.419556 sigma, covariance eigenvalues [0.054485318251507306, 0.3358668798125167], radial KS 0.999999, maximum projected KS 1.000000.
Last scored ema at update 1: mean error 7.331788 sigma, covariance eigenvalues [0.039288944606393185, 0.37717480337571857], radial KS 1.000000, maximum projected KS 1.000000.

The original budget was not completed within the fixed 120-second cap. The default batch is 2,048, and `BatchDistanceDiscriminator` computes differentiable batch-pair distance features with four kernels, including higher-order penalty derivatives. This is an execution-budget failure of this CPU cohort, not a demonstrated 1,000-update distribution failure. The GIF shows only saved actual checkpoints; no final state or convergence is inferred beyond them. The failed attempt was not retried.

## Separate CPU-sized Gaussian profile

A separately frozen diagnostic keeps the original public example, networks, initializer, seed, Gaussian data law and 1,000-update budget, changing only the public recipe batch size from 2,048 to 128 through a wrapper around `particlegan.get_recipe`. Its complete resolved recipe, exact wrapper, claim and analytic gates were registered before launch. It has its own 120-second cap and is never substituted for the original timeout.

CPU128 live: FAIL; mean error 0.135608 sigma, covariance eigenvalues [0.9230768849210969, 0.9706176716365458], radial KS 0.071986, maximum projected KS 0.066848.
CPU128 ema: FAIL; mean error 0.018816 sigma, covariance eigenvalues [0.9290288858975172, 0.9353262302203414], radial KS 0.083352, maximum projected KS 0.046441.

At the full CPU128 budget, the live cloud misses the centering and projected-CDF bounds. EMA misses the radial-CDF bound. These measurements identify the distribution defects; this one unchanged run does not isolate an optimizer mechanism causing them.

## Engineering interruptions

The first five-word attempt received external SIGTERM at a last valid 6,401-pair checkpoint before the declared 900-second cap. The signal origin is **UNKNOWN**. The heartbeat/child-ownership change improves supervision and does not prove why it happened. Its source, raw checkpoints, last metrics and known costs remain archived. One corrected full-budget attempt uses the same source/recipe/seed/budget; its endpoint is the evidence, with no choice between attempts.

The first CPU128 launcher hit a concrete JSON-read race while progress was being rewritten, ending after at least 38 update pairs and a valid update-20 checkpoint. Atomic receipt replacement and tolerant heartbeat reads fixed this observer bug. One engineering recovery used the identical frozen profile. Saved prefix identities are checked separately for both recoveries. The genuine original CPU timeout was preserved without retry. All five attempts and their receipts/costs remain distinct.


## Purity and provenance

Short baseline/observed software probes compare exact model/optimizer tensors, training inputs, owned RNG streams and caller Torch/Python/NumPy RNGs. Observation preserves module modes and uses a separate fixed evaluation generator. Public `prior.sample` retains MoG kernel noise when present; prior locations are never substituted for its sampling law. Original five-word frame saves occur after EMA updates. No training loss or update is replaced. The observers, pinned source files, evaluator and raw checkpoint clouds have SHA-256 receipts. Bulk logs, streams, tensors and the original dashboard frames remain outside Git at `/ml2/hypergan/toy-source-demos-20261001`.

Full-run caps are 900 seconds for five words and 120 seconds for quickstart. Repeated scientific seed studies, confidence/mass tuning, best-checkpoint selection, longer budgets and default promotion were not performed. The complete default budget plus five passing terminal observations is required for a new diagnostic PASS.

```sh
python -m benchmarks.toy_audit.source_demos --problem five_modes \
  --evaluator /ml2/hypergan/toy-source-demos-20261001/five_modes/evaluator.py \
  --output /ml2/hypergan/new-five-word-artifact --wall-seconds 900
python -m benchmarks.toy_audit.source_demos --problem quickstart \
  --evaluator /ml2/hypergan/toy-source-demos-20261001/five_modes/evaluator.py \
  --output /ml2/hypergan/new-quickstart-artifact --wall-seconds 120
python -m benchmarks.toy_audit.source_demos --problem quickstart --profile cpu128 \
  --evaluator /ml2/hypergan/toy-source-demos-20261001/five_modes/evaluator.py \
  --output /ml2/hypergan/new-quickstart-cpu128-artifact --wall-seconds 120
python -m benchmarks.toy_audit.source_demo_report \
  --artifacts /ml2/hypergan/toy-source-demos-20261001 \
  --output reports/toy_audit
```
