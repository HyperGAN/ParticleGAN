# V6 actual-training visualizations

These seven GIFs display certified saved training observations under the V6 source. The pre-run selected BCAP/DualNorm recipe supplies every task; selection is independent of grades. Failed gates remain failed. No models were constructed or queried, and export added no training updates, sampling draws or rescoring.

[Exact selection, source and artifact receipts](index.json) · [Final numerical readout](../final-v6/README.md)

| Task | Recorded gate | Actual-training GIF |
| --- | --- | --- |
| ae_gan_hold | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/media-v6/ae_gan_hold.gif) |
| clockfree_audit_measurement_v1 | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/media-v6/clockfree_audit_measurement_v1.gif) |
| five_word_joint_acquisition | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/media-v6/five_word_joint_acquisition.gif) |
| gaussian1d_smoke | FAIL | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/media-v6/gaussian1d_smoke.gif) |
| ring16_acquisition | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/media-v6/ring16_acquisition.gif) |
| two_pole | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/media-v6/two_pole.gif) |
| unused_token_hold | PASS | [GIF](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/gaussian-smoke-inventory/media-v6/unused_token_hold.gif) |

Clockfree audit is a diagnostic. The remaining tasks are the six required Tier 1 gates. No recipe passed all six, so the twenty required Tier 2 tasks remain ineligible and have no new training media.

Reproduce from the retained original queue and attempt artifacts in the project Python environment:

```sh
python reports/forge/gaussian-smoke-inventory/export_media.py \
  --queue-root runs/forge/gaussian-smoke-inventory-v6 \
  --attempts reports/forge/attempts --campaign gaussian-smoke-inventory-v6 \
  --output runs/software/gaussian-smoke-inventory-v6-media-reproduction
```
