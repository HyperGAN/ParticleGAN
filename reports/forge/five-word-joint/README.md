# Five-word joint BiGAN acquisition

The example already has successful full-budget source and public-API evidence.
It was absent from Forge because it lacked a registered joint-word adapter,
task, evaluator binding and view assignment. This registration connects those
pieces and gives the tier report an explicit link to its retained question.

The [registered task](../../../configs/forge/tasks/five_word_joint_acquisition.json)
asks whether a generator, encoder and joint critic can learn five equally likely
words and the correctly paired inverse mapping. It preserves the actual question
from [`examples/five_modes.py`](../../../examples/five_modes.py): `apple_`,
`grape_`, `lemon_`, `melon_`, `berry_`, each represented by six positions over
28 characters, including underscore padding. The source calls this BiGAN-style
joint adversarial learning, also known as ALI; it is not five Gaussian clusters.
The question covers this finite vocabulary, not language generation or unseen
word/typo generalization.

## Review the evidence

- [Current experiment guide](../EXPERIMENTS_BY_TIER.md#experiment-five-word-joint):
  task/view policy, original API result, actual-training GIFs and compact evidence.
- [Original-example convergence readout](../../toy_audit/SOURCE_DEMOS.md) and
  [source-example GIF](../../toy_audit/media/source-five-modes.gif): full-budget
  clean/live and EMA PASS under their original source protocol.
- [Public-API word-host explanation](../../toy_audit/api_contract/images/README.md)
  and [full API training GIF](../../toy_audit/api_contract/media/image-five-words-joint-ae.gif):
  a separate 20,001-update KA2 CPU PASS. Its [compact published receipt](../../toy_audit/api_contract/runs.json)
  records five modes, quality fraction 1.0, TV 0.02890625, exact paired
  reconstruction and minimum correct reconstruction-token probability 0.999753.
- [Frozen short-demo protocol](demo-protocol.json): one 32-update API integration
  check, 60 seconds maximum, seed 0, 1,024 samples, 24 post-update observations.
  [Compact receipt](../../toy_audit/api_contract/five_word_joint/publication.json)
  and [actual-training GIF](../../toy_audit/api_contract/five_word_joint/goal.gif)
  are separate from the historical campaign, with no full-task credit.

The historical/source and API results keep their original source, recipe, prior,
initialization, sampling, budgets and runtime. They do not qualify this new Forge
task. The [current solution leaderboard](../technique-inventory.md) retains its
frozen revision-2 research cohorts; this registration supplies no full-task
qualification.

## Short-demo result and recommendation

The sole integration run executed the frozen code at
[`1ff8e385`](https://github.com/255BITS/ParticleGAN/tree/1ff8e3858bd711b67e391063b52f5dcdfa99b9b6).
Its original receipt, GIF and raw-artifact hashes are unchanged. The current
reproduction wrapper additionally saves its inspected source manifest before
execution and observations before rendering, retaining them if media export
fails. This export hardening has not rerun or relabeled the original experiment.
At 32 updates its instantaneous scientific metric is **FAIL**; against the full
20,001-update task the recorded verdict is **INCOMPLETE**.

| Measurement at update 32 | Observed | Required |
| --- | ---: | ---: |
| Confident canonical-word fraction | 0 | ≥ 0.95 |
| Confident modes | 0 | 5 |
| Word/rejection TV | 1.0 | ≤ 0.10 |
| Exact paired reconstruction | False | True |
| Minimum correct reconstruction-token probability | 0.118271 | ≥ 0.90 |
| Actual optimizer updates, each of G/E/prior/D | 32 | 20,001 |
| Unintended RNG deviations | 0 | 0 |

State stayed finite and all four roles updated. The 0.781-second measured
adapter loop is a partial CPU cost, excluding GIF rendering and imports. The
receipt distinguishes actual critic-penalty activation from synthetic component
probes for branches not activated in this short host run; those probes are not
training or convergence evidence. Nine GIF frames show real targets, generated
strings and correctly paired reconstructions. Raw stdout, observations and
state remain under ignored `runs/forge/five-word-joint-demo-v1` in the worktree;
the published compact receipt retains their exact SHA-256 identities.

This verifies the registration path and its evidence guards. It does not test
whether the Forge cohort succeeds at its full budget and does not contradict
either older full-budget PASS. Review the preserved joint objective and numerical
controls first, then decide whether to run the normal gated campaign. Keep Tier 1
provisional until a separately budgeted calibration measures rejection cost and
false rejections; no automatic continuation, tuning or promotion follows.

## Why it was missing from Forge

The [original question catalog](../../toy_audit/catalog.json) retains this example
as `source-family-15`, originally a source-reviewed question. The later source
and [API migration](../../toy_audit/api_contract/cases.json) added runnable
training, strengthened numerical gates and actual-state GIFs. That migration did
not register a Forge task or qualification-view assignment.

Forge's [image adapter](../../../experiments/forge/adapters.py) builds scalar
image GAN hosts through `GANTrainer`; it has no inverse encoder or joint
`D(x,z)` objective. The existing [ae_gan_hold](../../../configs/forge/tasks/ae_gan_hold.json)
is a different 250-update behavioral test with reconstruction-MSE/hold gates.
Neither supplies the five-word mass, confidence or paired-correctness evidence.
There was also no explicit retained-question join from a Forge host name to
`source-family-15`, so the generated tier report could not find its explanation
and GIF. These are registration, host-binding and evidence-identity gaps; the
original example was not omitted because it failed to learn the five words.

## What this registration contains

The [shared WordFixture](../../../benchmarks/toy_audit/api_images.py) retains the
source-shaped networks: G 2→64→128→168, E 168→128→64→2, and joint D
170→256→128→1. It distinguishes real `(x,E(x))` pairs from fake `(G(z),z)`
pairs. Its relativistic generator/encoder objective reverses those pair scores;
there is no reconstruction lookup or auxiliary reconstruction loss. Prior
spread regularization retains useful code separation.

The [Forge binding](../../../experiments/forge/word_adapter.py) injects the
candidate's public recipe, five learned 2D particle rows and named initialization,
data, prior, input/output-noise and evaluation streams into that existing host.
The finite vocabulary gets an explicit `particle_cloud`, sigma-zero exception;
the new Forge prior is nonstandardized. A learned-MoG variant needs its own task
identity and calibrated measurement. Observations use actual clean/live public
sampling and reconstruct each canonical input in its correctly paired row.
EMA/controller serving and independent-row controls are blocked before
reservation; incompatible resources/objectives/priors and supplied initialization
are rejected instead of silently dropping mechanisms.

The [source-shaped KA2 candidate](../../../configs/forge/ideas/five-word-joint-ka2-v1.json)
uses LR 0.0006, D multiplier 1.5, prior spread 1.0, Adam `(0,0.999)`, EMA decay
0.995 and zero training input/output noise. The original 20,000-update recipe
schedule is preserved while the actual training allowance is 20,001 updates,
matching the source's inclusive loop. Its explicit `host_adaptation` delegates
`prior_reg` to each existing behavioral prerequisite's own objective; the joint
word task keeps candidate spread 1.0. This preserves host ownership without
discarding a formulation mechanism. No production recipe or default changes.

The main-view registration changed only this card's `goal` metadata. Its
[legacy admission registry](../../../configs/forge/legacy-ideas-v1.json) now pins
that corrected card and records the original commit, Git blob and SHA-256 beside
the corrected identities. The [exact original card](../../../configs/forge/legacy-history/five-word-joint-ka2-v1-before-goal-correction.json)
is retained for comparison. Recipe and mechanism fields are identical; goal
metadata does not enter the scientific candidate identity. Admission still
requires the exact pinned declaration, and historical results keep their original
scope.

| Numerical condition at every terminal observation | Bound |
| --- | --- |
| Generated samples | 1,024 |
| Confident canonical-word fraction | ≥ 0.95; each of all six tokens has probability ≥ 0.90 |
| Accepted modes | Exactly 5 |
| Word/rejection total variation from uniform five-word law | ≤ 0.10 |
| Correctly paired six-token reconstruction | Exact for all five inputs |
| Minimum correct reconstruction-token probability | ≥ 0.90 |

The shared [five_word_metrics / score_words](../../../benchmarks/toy_audit/definition_quality.py)
scorer is checked against oracle samples, correct-argmax low confidence, collapse,
unbalanced mass, wrong padding and swapped reconstruction. A perfect generated
marginal with wrong reconstruction pairs fails. The existing transfer grader
requires all 24 observations and five consecutive passing terminal checks.
Finite state, G/E/prior/D optimizer updates, mechanism activation and RNG
isolation also remain required; an endpoint or PASS stamp cannot qualify a run.

The [main discriminator_stability view](../../../configs/forge/views/discriminator_stability.json)
places this acquisition task provisionally in Tier 1, alongside
`ring16_acquisition` and the three existing mechanism checks. Revision 3 requires
5/19/2 tasks. The 20,001-update word budget and 900-second allowance stay unchanged;
placement alone does not establish cheap rejection or a calibrated screen.
Historical revision-2 3/19/2 results, calibration and search contracts retain
their frozen identities. A scientific tier study should preregister
positive/negative references, false rejections and cost-to-rejection before
adopting the expanded screen.

## Reproduce and review

Use the project Python environment. Validation and planning are read-only;
report regeneration writes the Markdown report and launches no training:

```sh
python -m experiments.forge validate
python -m experiments.forge plan five-word-joint-ka2-v1 --view discriminator_stability --through-tier 1 --device cpu
python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md
```

The sole short integration demo is explicitly outside ordinary qualification:

```sh
mkdir -p runs/forge
python reports/forge/five-word-joint/reproduce_demo.py --output runs/forge/five-word-joint-demo-v1 > runs/forge/five-word-joint-demo-v1.log 2>&1
tail -F runs/forge/five-word-joint-demo-v1.log
```

It renders actual targets, generated strings and paired reconstructions, showing
numerical status at real updates. Reduced-budget metrics can fail at 32 updates;
the evidence is always **INCOMPLETE against the full 20,001-update task**.
Bulk stdout, observations and state remain under ignored `runs/forge`; publish
only the compact receipt and final GIF. Output directories are exclusive.

For a later explicitly requested ordinary run, the
[2,100-second campaign](../../../configs/forge/campaigns/tier1-acquisition-v1.json)
reserves the three existing 300-second behavior allowances, the 300-second ring
allowance and the 900-second word allowance. The original 1,800-second
`five-word-joint-v1` campaign stays frozen and cannot cover the complete revised
Tier 1:

```sh
python -m experiments.forge run five-word-joint-ka2-v1 --view discriminator_stability --through-tier 1 --gpus cpu --campaign configs/forge/campaigns/tier1-acquisition-v1.json
python -m experiments.forge logs --follow --candidate five-word-joint-ka2-v1
python -m experiments.forge board --goal discriminator_stability
```

Ordinary failures stop remaining tasks in the same tier and all later tiers;
word acquisition follows the four earlier Tier 1 checks. The
[existing-config refresh](../tier1-refresh/README.md) measured two current word
failures; R1/R2's word cell remained unmeasured after its ring prerequisite failed.
The later [bounded root-cause study](../word-root-cause/README.md) completed 18
direct task-only diagnostics and obtained sustained full passes for K3P, KA2 and
R1/R2 with existing settings. Its
[exact task recipes](../../../configs/forge/selections/word-joint-task-v1.json)
and actual [K3P](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-root-cause/media/k3p.gif),
[KA2](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-root-cause/media/ka2.gif) and
[R1/R2](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/word-root-cause/media/r1r2.gif) training GIFs retain the unchanged
word core, gates and budget. All new generic diagnostic arms use prior_reg0;
the historical/source-shaped KA2 card above retains prior_reg1. Noise/horizon
ablations have explicit diagnostic identities rather than ordinary same-signature
search credit. No prerequisite has been bypassed for qualification, no historical
or diagnostic pass fills the ordinary leaderboard, and old receipts stay frozen.
A release choice still needs full Forge qualification and calibration.
