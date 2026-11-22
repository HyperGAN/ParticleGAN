# E22 convergence animation

![E22 on grid100, rotated100 and staggered100](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/e22-animation/e22-100gaussians.gif)

The [E22 configuration](../../docs/e22.md) trained on the three frozen native 100-Gaussian problems for 7,000
updates each: seed 1234, a 20,000-row particle table, batch 2,048, an identity `nn.Linear(2, 2)` generator and
the Fourier-feature MLP critic. Each frame draws 4,096 points from the served model with fixed latent and
noise seeds, so a point is the same particle in every frame. There is one frame per 25 updates, played at
25 fps, with a 2.5 s hold on the last frame. The captions (modes with at least 10 of the plotted points
within 3σ, and the share of points within 3σ) are computed from the plotted points, not from the frozen gates.

The runs use this repository's package (`init.deterministic_orthogonal_(D, seed=1)` for the critic) and
reproduce the archived E22 gate runs. The step-7,000 evaluation draw, scored with the harness's evaluation
seeds, matches their final checks:

| task | hq (this run) | hq (E22 gate run) | mass TV (this run) | mass TV (E22 gate run) |
|---|---|---|---|---|
| grid100 | .98485 | .9849 | .0322 | .0322 |
| rotated100 | .98575 | .9858 | .0334 | .0334 |
| staggered100 | .98265 | .9827 | .0325 | .0325 |

## Reproduce

```bash
# CUDA; about 12 minutes per task on an RTX A6000 (the three can run side by side)
for t in grid100 rotated100 staggered100; do
  python reports/e22-animation/record_e22.py $t frames-$t.npz
done
python reports/e22-animation/render_gif.py e22-100gaussians.gif --dir .
```
