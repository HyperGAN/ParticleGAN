# Retained goal views for the Atlas diagnostic cohort

These four supplements make the original failed questions visible. They preserve the original source, nine media steps, saved metrics and **FAIL** verdicts. The original GIFs remain byte-identical. This is display work with zero updates, model calls, checkpoint loads, new draws or scorer calls; it adds no ordinary qualification, default selection or speed credit.

| Question and original measured failure | Supplement | Original |
| --- | --- | --- |
| TwoPole1D asks whether particles escape the near-zero state while the critic's median input gradient stays controlled. At update 80, mean absolute particle value is .01366978, below .30; median gradient .02012139 meets its upper bound of 1. The displayed poles are the target; these gates alone do not establish balanced recovery of both poles. | [Horizontal particles and retained gradients](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-publication-views-20261003/two_pole-1d-goal.gif) | [Original](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-diagnostics-native-v2-20261003/gifs/two_pole_policy_selected_cloud_v1.gif) |
| Grid100 asks for 100 equally weighted Gaussian modes with sigma .03, including width. Final minimum covariance eigenratio .001795 and radial median ratio .06215 fail the .40/.65 bounds. Absolute covariance trace bias .96497 and radial KS .84886 also fail .10/.04. | [Global geometry and mode 0 width](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-publication-views-20261003/grid100-mode0-goal.gif) | [Original](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-diagnostics-native-v2-20261003/gifs/grid100_policy_selected_cloud_v1.gif) |
| Rotated100 asks the same shape question after a 25-degree rotation. Final minimum covariance eigenratio .001471, radial median ratio .11187, trace bias .95855 and radial KS .81743 fail the same bounds. | [Global geometry and mode 0 width](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-publication-views-20261003/rotated100-mode0-goal.gif) | [Original](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-diagnostics-native-v2-20261003/gifs/rotated100_policy_selected_cloud_v1.gif) |
| Staggered100 changes neighbor geometry through contracted row spacing and alternating offsets. Final minimum covariance eigenratio .003183, radial median ratio .07657, trace bias .95556 and radial KS .79944 fail the same bounds. | [Global geometry and mode 0 width](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-publication-views-20261003/staggered100-mode0-goal.gif) | [Original](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/atlas-current-gpu-diagnostics-native-v2-20261003/gifs/staggered100_policy_selected_cloud_v1.gif) |

The native numeric values come from the saved final **20,000-sample primary checkpoint**. Their independent 100,000-sample endpoint grades use a separate draw. These movies retain the original 4,096-sample display clouds. They compute no metric from the display cloud or local viewport. Recovering all centers in a global plot cannot establish Gaussian width.

The zoom translates all original coordinates by the declared mode 0 center and clips the axes to ±4 sigma. Gray point contours are deterministic circles at one and two sigma, derived from the original geometry in [problems.py](../../../benchmarks/toy100/problems.py): grid center (-4.5, -4.5), its 25-degree rotation, or staggered center (-3.825, -4.75). They are reference contours, not random draws or confidence regions. A single-mode zoom illustrates the failure; the original full-population tests provide the verdict.

TwoPole displays every saved scalar as a horizontal coordinate with a display-only vertical coordinate of zero. Its gradient panel preserves all 24 original real/particle input gradients; gray level 1 marks the median-gradient bound, not a per-bar gate. Its actual media steps are 4, 14, 24, 34, 44, 50, 60, 70 and 80. Each native supplement uses 0, 50, 750, 1750, 2750, 4000, 5000, 6000 and 7000.

The native archive's legacy `live` array label denotes the actual state-selected public policy output. Its final selection is averaged with the feature-cell backend and 20,000 prior rows; forced EMA is diagnostic only and is not shown. Evaluation omits output noise while retaining the actual latent policy. TwoPole measures the selected table and critic gradient without a latent draw. The receipt retains the actual selection/backend declaration for every displayed step. Neither law borrows success from historical noisy Atlas tests.

[receipt.json](receipt.json) binds the raw study, original grades/media/arrays, scientific source `9563dea57bb150f2a0275bbe8d785bf76210fca3`, renderer bytes, display projections and output GIFs/posters. The reused renderer's “Default test FAIL” badge denotes the original full fixture's failed test; these remain supplemental diagnostics. The scientific cohort's paid cost remains 2996.7973392466083 seconds. CPU display time is separate and excluded from that cost.

Reproduce from the immutable local raw archive into a **new empty directory**:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/pg-atlas-retained-view-mpl \
/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
  reports/forge/atlas-current-gpu-publication-views-20261003/render_retained_views.py \
  --raw /ml2/hypergan/forge-atlas-current-gpu-diagnostics-20261003-native-v2 \
  --output /tmp/atlas-goal-views-new
```

The exporter verifies source and input hashes before/after rendering and checks nine decoded GIF frames. Inspect the four final posters before recording pixel-review completion; exporting alone leaves that review pending.
