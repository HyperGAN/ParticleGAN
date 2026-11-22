# E22 → ParticleGAN Atlas

Two vector infographics explain the shared GAN loop and Atlas’s population
controls for readers familiar with GAN basics.

1. [E22 explained](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/atlas-explanation/e22-explained.png): trainable latent particles, generator,
   critic, adversarial gradients, and E22’s existing training feedback.
2. [Atlas changes](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/atlas-explanation/atlas-changes.png): neighbor evidence versus temporary
   critic-feature regions, separate count/placement/support actions, and
   capability-based selection.

Both figures have editable SVG versions. Dots, regions, and arrows are
schematic; no benchmark samples or measured scores are invented here.
The [illustrated guide](../../docs/atlas.md) gives the formulation, advantages,
measured comparison with current PR155 E22, usage, and limits.

## Sources

- PR155 `cabe2084284db923d525918cbf3e18de6f20faac`:
  `docs/e22.md`, `configs/100gaussians/e22-noout.json`, and E22 policy source.
- Atlas’s tested package:
  `500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61`.
- `docs/feature-cells.md`, `particlegan/feature_cells.py`,
  `feature_policy.py`, `mean_transport.py`, `anchor_birth.py`, and
  `population_continuity.py`.

Placement repairs use checked row reallocation/cloning within a group to
correct its output mean. Birth proposals are latent codes tested through G
in critic features. E22 already balances mass
and checks support; those concepts are not introduced by this PR.

## Reproduce

```sh
python render_infographics.py
```

The renderer uses Matplotlib and exports SVG plus high-resolution PNG. The
figures have no dependency on CUDA, models, datasets, or training RNGs.
