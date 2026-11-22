# BCAP: constant learning rates and six Tier 1 passes

BCAP with **`optimizer_smoothing=1e-5` passes all six unchanged required
Tier 1 tasks**: Gaussian smoke, two-pole movement, unused-token hold, AE/GAN
hold, ring16 acquisition and five-word acquisition. The only trainer change
from the recorded incumbent is enabling the existing fixed-scale DualNorm
smoothing. G/E stays at `.012`, D at `.018`, and sampled-prior rows at `.03`.
Momentum is zero; learning-rate floors are one throughout. There is no
learning-rate annealing.

The reusable configuration is
[`bcap-dualnorm--8db70e3c…`](../../../configs/forge/configurations/bcap-dualnorm--8db70e3cb9fd3da9b5cc6a117731e8572cba837d64e7ee721d012d3157c9a3fe.json).
On the executed public API, its trainer settings are also obtained with:

```python
from particlegan import get_recipe

recipe = get_recipe("bcap", optimizer_smoothing=1e-5)
```

Task declarations still supply their architectures, priors and budgets; this
example does not replace those conditions. The public preset remains unsmoothed.
The single [current technique inventory](../technique-inventory.md) records the
whole selected configuration, while [readout.json](readout.json) preserves every
trial's final metrics, original receipt hashes, source, cost and archive identity.

The [subsequent frozen-recipe Tier 2 stage](../bcap-tier2/README.md) is complete:
6 PASS, 11 FAIL and four image setup errors across its 21 required tasks.
It reuses this Tier 1 evidence, retains constant learning rates and updates the
same inventory. BCAP remains qualified through Tier 1.

The [declared three-recipe study](../../../configs/forge/searches/bcap-six-smoothing-v1.json)
tests global smoothing strengths `1e-5`, `1e-4` and `1e-3`, with a reservation
ceiling of **7,560 seconds**, **2,520 per recipe**. It finishes every runnable
Tier 1 task and the existing optional clock audit for each recipe. All comparisons
use protocol seed `0`, the public deterministic initializer and the original
task-owned fixture exceptions. Task declarations, initial learned states and
consumed named streams are checked across recipes. The Gaussian's actual target
batch digests match too. The study changes no architecture, target law, prior,
served sampling law, update budget, evaluation cadence or numerical bound.
Numerical-rank truncation stays enabled; autograd multithreading stays disabled.

All three complete global recipes pass **6/6**, plus the optional clock audit:

- `1e-5`: Gaussian confirms first at 959, with 1/24 passing pairs; word
  acquisition confirms first at 4,167, with 10/24 pairs and a passing endpoint.
- `1e-4`: Gaussian confirms first at 459, with 3/24 passing pairs; word
  acquisition confirms first at 834, with 12/24 pairs and a failing endpoint.
- `1e-3`: Gaussian confirms first at 875, with 1/24 passing pairs; word
  acquisition confirms first at 1,667, with 22/24 pairs and a passing endpoint.

The declared PASS-count/content-hash tie-break selects `1e-5`. This does not
establish that it converges faster or retains quality better than the other two.
All **21 CUDA attempts** complete, with **zero scientific retries**, for
**2,510.268425 paid worker seconds**. Reservations are released. Saved-sample
analysis reproduces all **150 Gaussian sample sets** exactly and verifies
**75 unchanged-state primary/confirmation pairs**, including the initial snapshots.
The archive contains 1,586 byte-verified original files, SHA-256
`44de286fe6adfe3bde97f27a927fe224328391025083c3d15759c2fc89a16ab8`.

The previous unsmoothed configuration failed Gaussian confirmation despite
passing the other five tasks. Its primary KS at update 167 was `0.047977`, but
the independent same-state draw scored `0.054185`, above the `0.05` limit.
The [diagnosis in PR #349](https://github.com/255BITS/ParticleGAN/pull/349) and
[word-split readout](../word-split-inventory/README.md) retain that original
source and evidence. Those archived results motivate this study; they are not
pooled into a new candidate or treated as a newly measured matched control.

At smoothing `1e-5`, Gaussian first confirms acquisition at **update 959**.
Primary KS is **`0.044836`**, and independent confirmation KS is **`0.033797`**.
Primary mean/std are `1.984291 / 0.514837`; confirmation mean/std are
`1.987562 / 0.510631`, against target `N(2, 0.5²)`. Both draws meet every bound,
and the training-state hashes agree. The run completes all 1,000 updates and
24 confirmation pairs. Only one scheduled pair fully passes, and the endpoint
KS is `0.084096`; this is acquisition evidence, with retention still untested.

The same recipe preserves ring16's full gate at **1,600 updates**: all 16 modes,
quality fraction `0.968750`, mass TV `0.052002`, component covariance error
`0.451691`, and minimum component eigenvalue ratio `0.300655`. Its five-word
run finishes all **20,001 updates**, first confirms at **4,167**, and passes
**10/24** pairs. At the endpoint, all five words are generated, quality is `1`,
mass TV is `0.018945`, paired reconstruction is exact for every word, and minimum
correct-token probability is `1`. The three behavioral tasks also PASS under
their unchanged gates.

Fixed smoothing weights each retained matrix singular direction by
`s / hypot(s, lambda)` and vector/prior-row directions by
`g / hypot(norm(g), lambda)`. Weak directions therefore receive smaller updates
without using elapsed time to reduce learning rates. The measured Gaussian
acquisition is consistent with this hypothesis, but it does not establish
convergence: permitted acquisition states and sustained retention are separate
questions. See the [implementation and structural-search contract](../../../docs/dualnorm-smoothing.md).

Freeze the selected whole recipe for any subsequent ordinary Tier 2 study.
Those 21 requirements, including Gaussian continuation/target shift and the
strict own-checkpoint word hold, remain unmeasured here. The expanded screening
profile is provisional; this report changes the experimental family selection,
with public-default adoption still false.

Seven [actual-training GIFs and their receipts](media/index.json) use this same
completed recipe. The [Gaussian](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-six/media/gaussian1d_smoke.gif),
[ring16](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-six/media/ring16_acquisition.gif), and
[word](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bcap-six/media/five_word_joint_smoke.gif) animations illustrate the numerical goals.
Export disables model construction, forward calls, training and sampling, and
uses only saved observations. The GIFs add no updates or draws and do not
determine qualification.

Reproduce in the project environment from the repository root. Training was
frozen at commit `96b03a5e577dab9c566e346df8e2a100f79787d6`, scientific source
digest `801d07b11ff269f441d7960dbc445939aa770dbceebdca5e315432c76b46b97a`.
Bulk stdout, JSONL,
checkpoints and tensors remain outside Git in the byte-verified archive listed
by the readout. Hydrate its original paths before independent receipt regrading.

```sh
python reports/forge/bcap-six/run.py --stage plan > runs/bcap-six/plan.json
python reports/forge/bcap-six/run.py --stage run --gpus 0,1 \
  > runs/bcap-six/controller.log 2>&1
tail -F runs/forge/bcap-six-smoothing-v1/progress.jsonl
python reports/forge/bcap-six/run.py --stage report > runs/bcap-six/report.json
python reports/forge/bcap-six/publish.py
python reports/forge/bcap-six/export_media.py
python reports/forge/regenerate_technique_inventory.py \
  --source-commit 96b03a5e577dab9c566e346df8e2a100f79787d6 --device cuda
python reports/forge/bcap-six/select.py
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
```

Existing artifact archives and media directories are immutable. A fresh run
needs a fresh checkout and queue; restoring saved evidence permits read-only
analysis and publication without repeating training. The coordinator briefly
stopped its per-attempt memory rebuild and recovered the same queue with batch
publication. Original logs are archived; completed training was not repeated.

The [selection change receipt](selection-change.json) retains the exact previous
measurement pin. An explicit source pin also preserves the existing unmeasured
native-Adam declaration when family selections span sources; its rows remain
unmeasured and receive no gate credit. The
[publication-only software amendment](publication-amendment.json) binds the
archived and corrected selection helpers. Use the committed helper for publication;
the scientific archive, training source and measured outcomes are unchanged.
