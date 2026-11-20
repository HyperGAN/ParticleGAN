# BISPF failure investigations, 2026-10-22

These eleven audits explain why the selected BISPF recipe failed its recorded tests.
They inspect existing measurements and add no training, sampling, repairs or qualification.
The reports, numerical readouts and audit sources are preserved byte for byte from their
original PR commits. They are consolidated in PR #473 so the shared evidence can be reviewed once.

The [current technique inventory](../technique-inventory.md) owns current benchmark status.
Counts, source hashes and mechanism observations on the child pages describe their dated cohorts.
Their original FAIL results remain unchanged. Later held-test policy and repair studies do not
turn those failures into passes or establish a replacement default.

| Original PR | Preserved investigation | Original immutable commit |
| --- | --- | --- |
| [#463](https://github.com/255BITS/ParticleGAN/pull/463) | [gaussian1d_stability](gaussian1d_stability/README.md) | [f50d94e8258f](https://github.com/255BITS/ParticleGAN/commit/f50d94e8258f75565e87c2240a50d18659266d3b) |
| [#464](https://github.com/255BITS/ParticleGAN/pull/464) | [img_bars4](img_bars4/README.md) | [a0bfac2d5dcf](https://github.com/255BITS/ParticleGAN/commit/a0bfac2d5dcf2deb248fbe4f9b8a1b1f7ebdb07c) |
| [#465](https://github.com/255BITS/ParticleGAN/pull/465) | [img_blobs4](img_blobs4/README.md) | [8166ee119bed](https://github.com/255BITS/ParticleGAN/commit/8166ee119bed520f0ccfecfaf63607aaf77e6469) |
| [#466](https://github.com/255BITS/ParticleGAN/pull/466) | [img_intensity2](img_intensity2/README.md) | [08737791a852](https://github.com/255BITS/ParticleGAN/commit/08737791a85231bb0ea8e52d4b7b5d503b3ca595) |
| [#467](https://github.com/255BITS/ParticleGAN/pull/467) | [grid100](grid100/README.md) | [7f50d9a08eb7](https://github.com/255BITS/ParticleGAN/commit/7f50d9a08eb731fca1ac3aaab7ec3ba168d06d6c) |
| [#468](https://github.com/255BITS/ParticleGAN/pull/468) | [rotated100](rotated100/README.md) | [f3bac9fc1a4f](https://github.com/255BITS/ParticleGAN/commit/f3bac9fc1a4fee89b7cf075d21a87c7d9dfa2899) |
| [#469](https://github.com/255BITS/ParticleGAN/pull/469) | [staggered100](staggered100/README.md) | [d70f009a7a18](https://github.com/255BITS/ParticleGAN/commit/d70f009a7a1892e648dc34a5326f39735bf36c00) |
| [#470](https://github.com/255BITS/ParticleGAN/pull/470) | [vector_unequal_mass](vector_unequal_mass/README.md) | [bbf315460863](https://github.com/255BITS/ParticleGAN/commit/bbf3154608636cdd7d70de0bd8c1bb89c6bc3d29) |
| [#471](https://github.com/255BITS/ParticleGAN/pull/471) | [vector_unequal_width](vector_unequal_width/README.md) | [22b082c5605b](https://github.com/255BITS/ParticleGAN/commit/22b082c5605bb2ff6ae8df3ad00567b9e054c45b) |
| [#472](https://github.com/255BITS/ParticleGAN/pull/472) | [vector_anisotropic](vector_anisotropic/README.md) | [4a6b50118252](https://github.com/255BITS/ParticleGAN/commit/4a6b501182529fa5adadac36092b9b4a7c7fb564) |
| [#473](https://github.com/255BITS/ParticleGAN/pull/473) | [vector_overlap](vector_overlap/README.md) | [5ff21abf5e6f](https://github.com/255BITS/ParticleGAN/commit/5ff21abf5e6f63325b3ea628c91fab5079bf7ded) |

## Reproduction and limits

Each child page declares its original readout archive, executed source, recipe, task,
initialization, update budget, sampling law and actual-training GIF. Run the retained audit
from its original commit and supply that exact archive; summaries cannot replace the original
observations. Some audit outputs also hash the investigation-time source and publication,
so running `--check` on a later checkout is not an equivalent reproduction.

The historical archive path `/mnt/ml7tb/ParticleGAN-forge/bispf-optimized-20261016/`
was unavailable during the 2026-11-08 integration. The integration verified original-file
identity and report links; it did not repeat the original readout audits or recover missing
checkpoint/sample tensors. Those missing tensors still limit causal attribution.

No leaderboard or original evidence was rewritten during consolidation.
