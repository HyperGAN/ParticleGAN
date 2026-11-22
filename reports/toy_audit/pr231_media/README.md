# PR231 supplemental actual training media

Every GIF uses all 13 recorded clean held-out evaluations and all 1,200 recorded rate/noise updates for its exact source cohort. Final checkpoint dictionaries verify completed updates, actual metadata, recipe and prior without constructing a model. No intermediate sample clouds were captured, so these are measurement animations. No new training, metric computation or replay occurred.

The diagnostic asks whether erasing the whole additive FiLM branch removes code sensitivity, and whether that initialization explains rate cuts or late quality loss. The zero-hidden analytic Jacobian tests answer the geometry question with independent controls. The full native toys do not reproduce the real repeated ordinary G cuts. Spatial width 16 reverses the initialization ordering, and the matched real GPU G-rate bypass worsens LPIPS by 11.47%. Keep those counterexamples. Raw gradient energy and Adam displacement are separate measurements; neither is a convergence guarantee.

No frozen trained-convergence acceptance gate exists for these cohorts. `COMPLETE` means the budget and measurement stream completed; it does not mean a model PASS. Historical source snapshots, early backfilled protocols, and the separately verified final toy-source width 4 reproduction on `70de5a3e` remain distinct. The earlier report's no-final-source-rerun statement refers to historical cohorts; its later publication reproduction does execute the final common source for two arms.

| Cohort | Purpose | Actual training media |
|---|---|---|
| initial-width16 | Historical outer-identity architecture control; weaker backfilled provenance | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/initial-width16.gif) |
| native-handoff-width16 | Recipient handoff with large target; ordinary G cuts not reproduced | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/native-handoff-width16.gif) |
| calibrated-handoff-width16 | Calibrated target; ordinary G cuts not reproduced | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/calibrated-handoff-width16.gif) |
| faithful-width64 | Capacity/antithetic control; ordinary G cuts not reproduced | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/faithful-width64.gif) |
| faithful-width64-antithetic-bypass | Matched narrow antithetic rate intervention; no real-fix qualification | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/faithful-width64-antithetic-bypass.gif) |
| spatial-native-width4 | Small BF16 spatial handoff; original finishes better | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/spatial-native-width4.gif) |
| spatial-native-width16 | Wider spatial counterexample; shift zero finishes better | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/spatial-native-width16.gif) |
| publication-width4-replay | Final toy-source reproduction on 70de5a3e; two spatial width4 arms | [13 frames](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/pr231_media/publication-width4-replay.gif) |

The [receipt](receipt.json) binds every raw source/protocol/metadata/trace/final-state file hash. All 29 native package files match published PR231. A separate read-only check reproduces every scientific publication trace row after only the acknowledged corrected D-gradient reporting field is excluded, and verifies the matched antithetic prefix through 504 with first applied-rate divergence 505. Ambient global checkpoint RNG differences remain documented in the original publication receipt. The integrated develop branch contains later native changes; these historical bindings confer no new current-source qualification. No full real-image rerun, real fix, robustness, public-default or Forge qualification follows.

```sh
python -m benchmarks.toy_audit.pr231_media --validate-only
python -m benchmarks.toy_audit.pr231_media
```

Original raw archive: `/ml2/hypergan/routed-generator-damping-toy-artifacts-20261002`. Source report is pinned to PR231 head `68cf68e599d4ac8f8608434c417ed021647e2ff4`.
