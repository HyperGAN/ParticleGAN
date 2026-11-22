# Adopted ring16 GPU smoke conditions

The current ring16 task uses **256 learned particles, MoG sigma .1 and 1,600
updates**. Its numerical bounds remain unchanged. The new public caller,
`api-ring16-acquisition-v2`, uses the exact selected BCAP dualnorm configuration
and Forge's named initialization, data, training and evaluation streams.

The [duration evidence](../tier1-prior-duration/README.md) passes all full bounds
with six consecutive terminal passes: 16 modes, mass TV .05859, HQ .93774,
component covariance error .51431 and minimum eigen ratio .38370.
The [actual-training GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/tier1-prior-duration/mog100-n256-ring16_acquisition.gif)
illustrates this same target, recipe, prior and sampling law.

The task keeps the passing continuation's original recipe horizon of 400;
the external execution allowance is 1,600. Both learning-rate floors are 1,
so rates remain constant. G rate is .012, D multiplier 1.5, prior multiplier
2.5 and optimizer momentum zero. The generator/critic architecture, target
law, batch size 128, uniform MoG weights, initialization scale 1 and clean
live serving law match the evidence. The API fixture constructs the models
and trainer through the existing FormulationContext, with CUDA required.

The scoring allowance is **96 checks**, preserving the original approximately
17-update spacing. Five terminal checks are required at 1,534, 1,550, 1,567,
1,584 and 1,600. A separate declared-cadence evaluator uses the existing
numerical grader and suffix rule. The old evaluator retains its fixed
24-check contract. Extra media observations read saved samples or preserve
the evaluation stream; they cannot shift training or subsequent scoring draws.

[Verification](verification.json) checks the current caller against the saved
initial networks, prior, recipe, optimizer and all named streams, restores the
final CUDA checkpoint exactly, and regrades all 96 saved observations as PASS.
The initial comparison excludes the changed external cap and ambient global
CPU/CUDA RNG states; consumed streams match exactly and are checkpointed.
No scientific optimizer updates or new sampling draws were added by this
evidence check. Software checks exercise a two-update CUDA prefix, checkpoint
continuation, observer purity, destructive controls and incomplete/failed
cadence rejection. The focused suite passes 89 checks.

The original `api-ring16-acquisition` caller remains a distinct 400-update,
sigma-.025 K3P case. Its original task bytes are retained in
[the frozen task](../tier1-prior-smoke/frozen-tasks/ring16_acquisition.json),
SHA256 `e6b53ba29fbe9ead47e842cfa01e40ba57821bd1b4e6aa5b297631fa0f6525c1`.
The original prior/duration protocol files and archives are unchanged.
Run historical training from their pinned scientific commits; current diagnostic
software checks resolve only exact hash-matched archived task bytes.

The current discriminator-stability view advances to revision 6. Earlier grades,
the recorded 4/6 selection and its leaderboard remain historical evidence;
this task change does not turn the duration diagnostic into ordinary qualification
or complete scientific calibration. The Gaussian task is unchanged.

To execute the adopted smoke in an environment with Torch, CUDA, NumPy, SciPy,
Matplotlib and Pillow, choose an unused ignored output directory:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 python -u -m benchmarks.toy_audit.api_run \
  --case api-ring16-acquisition-v2 --device cuda:0 --recipe auto \
  --wall-cap-seconds 300 --output runs/api/ring16-smoke-v2 \
  > runs/api/ring16-smoke-v2.log 2>&1
tail -F runs/api/ring16-smoke-v2.log
```

This command performs the full public-API smoke and writes its actual-state GIF.
Short prefixes remain incomplete. Model training and sampling use GPU; frozen
target draws, numerical scoring and rendering retain their CPU reference law.

The [Gaussian handoff](../tier1-prior-duration/GAUSSIAN_HANDOFF.md) records the
unresolved scalar problem for the next session.
