# Longer training for the closest prior configurations

**Yes, longer training can solve ring16 acquisition; it does not fix the scalar Gaussian.**
The 256-row ring with MoG sigma .1 passes the unchanged full gate at 1,600
updates, with six consecutive terminal passes. The sigma-.025 ring passes
intermediate 800/1,200-update cuts, then loses sustained local spread by 1,600.
The closest Gaussian remains FAIL at 4,000 updates and regresses in width.

The subsequent [Gaussian saved-state diagnosis](../gaussian1d-diagnosis/README.md)
documents the actual z_dim=2, width-32 architecture, an affine capacity control
and the output/prior drift. It adds zero training updates and retains this
duration study's exact evidence and verdicts.

The user subsequently selected the passing ring conditions; the
[adopted GPU smoke](../ring16-smoke-v2/README.md) records the current task/caller
and preserves this original diagnostic. Gaussian work continues from
[the handoff](GAUSSIAN_HANDOFF.md) after compaction.

This is the user-requested follow-up to [PR #316’s prior grid](../tier1-prior-smoke/README.md).
It changes the execution allowance only. Original 400/1,000-update failures,
protocols, receipts, archive identities and leaderboard results remain intact.
The current ordinary BCAP selection is still 4/6; extended-budget diagnostics
do not fill its two failed cells.

## Frozen duration protocol

[Protocol](protocol.json) selects three exploratory post-screen arms before
additional training: the closest scalar CDF endpoint, the strongest ring
smoke configuration, and the ring’s lowest full covariance-error endpoint.
This is a duration study of selected configurations, not another prior search.

Each run restores its exact saved GPU checkpoint under the original allowance,
verifies model/prior/optimizer/stream identity, then calls the public
`GANTrainer.extend_execution` API. Only the external cap changes:
Gaussian 1,000 → 4,000; rings 400 → 1,600. The full BCAP recipe remains unchanged:
G step .012, D multiplier 1.5, prior multiplier 2.5, zero momentum, constant
rates, no additive training noise or spread regularization, learned uniform MoG.
Architecture, prior, target law, batch 128, initialization and seed remain fixed.

Resume proofs certify exact restored-state hashes and an unchanged state after
removing the execution-cap field. Data, constructor, training-noise and
evaluation streams are preserved; both ring continuations consume identical
additional target batch sequences. Every originally pinned implementation file
is checked unchanged before resuming. No prefix optimizer updates are repeated.

The continuation preserves the original evaluation spacing: 24 checks per
original-budget block, 96 total checks across four blocks. Every check samples
4,096 clean live outputs and applies the original bounds. At least five
consecutive terminal passes are still required. All three runs continue through
their declared caps, rather than stopping at a favorable observation.

The finite round permits three GPU continuations, **5,400 additional host updates**,
**2,880 reserved seconds**, no retries, and no further automatic round. All three
complete without errors on the same RTX A6000. The measured continuation loops
total **60.077 seconds**; construction, restore, source capture,
serialization and rendering are excluded. The original 134.613-second prior-grid
cost is separate and is not charged again.

## Completed results

| Target / prior | Total updates | Full terminal suffix | Result | Endpoint quality | Actual training |
| --- | ---: | ---: | --- | --- | --- |
| Gaussian / MoG .1 | 4,000 | 0 | **FAIL** | KS 0.09594; mean error 0.27536σ; std ratio 1.74796 | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-duration/mog100-n256-gaussian1d_acquisition.gif) |
| Ring16 / MoG 0.025 | 1,600 | 1 | **FAIL** | covariance error 0.58537; minimum eigen ratio 0.20444; HQ 0.98267 | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-duration/mog025-n256-ring16_acquisition.gif) |
| Ring16 / MoG 0.1 | 1,600 | 6 | **PASS** | covariance error 0.51431; minimum eigen ratio 0.38370; HQ 0.93774 | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-duration/mog100-n256-ring16_acquisition.gif) |

[Receipts and budget cuts](results.json) include resume proofs, final/terminal
metrics, source and recipe identities, and first sustained passing windows.
This is an unranked comparison; the existing [technique leaderboard](../technique-inventory.md)
remains the single leaderboard for the goal.

### Ring sigma .1: additional updates restore precision and covariance

At 400 updates it has all modes and acceptable mass balance but HQ .79907
and covariance error 6.67240. At 1,600 it has **16 modes, mass TV .05859,
HQ .93774, covariance error .51431 and minimum eigen ratio .38370**.
All bounds pass at updates 1,534, 1,550, 1,567, 1,584 and 1,600; the complete
terminal suffix is six, starting at 1,517. The first five-pass window completes
at 1,584. Thus a fourfold allowance can make this exact prior/trainer acquire
the full sixteen-Gaussian law at the declared evaluation resolution.

### Ring sigma .025: faster acquisition, followed by a spread failure

The original covariance error 9.61552 falls to .56936 at 800, .49848 at
1,200 and .58537 at 1,600. The full gate first sustains five passes by update
750; the 800-update cut has suffix eight and the 1,200-update cut suffix sixteen.
Its longest full passing streak is nineteen checks.

At 1,600 the endpoint itself passes every numerical bound. Nevertheless,
checks 1,550, 1,567 and 1,584 fail minimum component eigen ratio ≥.15, leaving
a terminal suffix of one. Mode coverage, HQ and mass balance remain good.
This is later local-spread instability after successful acquisition. Selecting
only its good final frame or its best earlier cut would hide that distinction.
The intermediate cuts are saved observations of this diagnostic trajectory,
not independently registered ordinary qualification results.

### Gaussian sigma .1: additional constant-step training is not a repair

The original endpoint KS .04297 was a passing snapshot, with no sustained
full pass. The 2,000/3,000/4,000-update KS values are **.16982/.13952/.09594**.
There are only three instantaneous full passes across all 96 checks, never
five consecutive passes. At 4,000, mean error .27536σ, std ratio 1.74796 and
KS .09594 all fail their bounds. The largest observed std ratio is 3.71290.

Both location and width wander during the extension. More compute at these
unchanged constant normalized step sizes does not establish stable Gaussian
learning. These observations do not isolate the optimizer, prior motion or
critic as the cause, and they do not authorize treating a scheduled-rate
experiment as a continuation of this same formulation.

## Recommendation

Keep **256 particles** for these small hosts. For ring acquisition, the duration
study supports testing a separately declared **800-update narrow-MoG question**
or a **1,600-update sigma-.1 question**, retaining the exact full bounds.
The narrower prior acquires sooner but needs an explicit endurance check,
because it later loses local spread. The wider prior passes only near the
end of the longer allowance; longer endurance is unmeasured.

For the Gaussian, retain the short-budget failure as a real stability/shape
problem. Further budget increases on this unchanged configuration have no
support from this readout. A future bounded trainer-stability hypothesis could
test existing rate/schedule settings while fixing the prior, rather than adding
another particle-count or duration grid. No such extra study is launched here.

These results justify revisiting the ring’s 400-update smoke allowance.
They do not justify weakening its covariance bounds, claiming both original
tasks pass, or adopting a new ordinary profile without separate calibration.
All current ordinary definitions and the original prior-grid report retain
their recorded scope; this extension adds a distinct budget cohort.

## Reproduce and verify

Scientific continuation source: `b4d1f95a074ffa64ac8f64e12aa0a049f836c0af`.
Use that commit and the original verified local prior-grid archive. From the
repository root, choose a new ignored destination; the default is exclusive:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -m cuda -q tests/test_tier1_prior_duration.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -u -m benchmarks.toy_audit.tier1_prior_duration run \
  > runs/api/tier1-prior-duration-v1.runner.log 2>&1
tail -F runs/api/tier1-prior-duration-v1.runner.log
# Each start event identifies the active JSON observation log.
# Publish from the final PR checkout in the existing rendering environment:
/home/martyn/dev/ParticleGAN/.venv/bin/python \
  reports/forge/tier1-prior-duration/publish.py \
  --raw runs/api/tier1-prior-duration-v1 --output reports/forge/tier1-prior-duration
```

Two CUDA software tests verify cap-only extension and atomic rejection of an
invalid allowance. Three actual resume proofs check the complete original
model/optimizer/RNG state without replaying training. Final optimizer counters
match 4,000/1,600 updates. Three nine-frame GIFs render saved prefix/continuation
outputs, including observed failures, with no new model sampling draws.
The frozen target generators and numeric scorers keep their CPU reference law;
all model training and prior/G sampling remain on GPU.

Raw logs, 96-check curves, tensor observations and checkpoints remain ignored.
[Artifact provenance](artifact-provenance.json) retains their exact local archive
and original-parent identities. No historical verdict or archive is overwritten.
