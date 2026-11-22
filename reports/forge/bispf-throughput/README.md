# BISPF throughput readout

The `bispf-throughput-v1` recipe is **2.15× faster at 256×256 DCGAN** and **about 2.3× faster on the small Forge image tasks**, compared with the selected quintic BISPF winner. It preserves all **19 incumbent numerical passes** across the same 30 task contracts. The [current technique inventory](../technique-inventory.md) remains the single leaderboard. The original opt-in publication at `72db8af31` preserved selection pins and public defaults. The owner subsequently selected this execution policy as the **0.9.1 public default**; archived qualifications and Forge selection pins retain their original identities.

## Measured throughput

All six existing mock-DCGAN particle-cloud tasks ran serially on the same physical RTX A6000, with seed 0, three warmup updates and ten measured complete public `GANTrainer.step` updates. These are median-update rates; input preparation, scoring and media are outside the timed region. Initial model/prior hashes, actual critic/generator batch digests, runtime and physical GPU match within every pair.

| Workload | Control steps/s | Candidate steps/s | Speedup |
| --- | ---: | ---: | ---: |
| 32×32, batch 16 | 35.45 | 36.12 | 1.02× |
| 64×64, batch 16 | 20.05 | 29.27 | 1.46× |
| 128×128, batch 16 | 8.24 | 14.59 | 1.77× |
| 256×256, batch 16 | 2.94 | 6.34 | 2.15× |
| 64×64, batch 64 | 10.89 | 23.32 | 2.14× |
| 64×64, width 128 | 9.01 | 15.25 | 1.69× |

The six-workload geometric mean is **1.66×**. The 2× target is met at 256×256 and batch 64; it is not met uniformly. At 256×256, median latency falls from 339.80 to 157.84 ms and peak allocated memory falls from 2,119 to 1,522 MiB. CUDA graph capture increases cold-start work and can increase memory on smaller workloads. Contention telemetry is unavailable, so ratios describe observed runs rather than a certified exclusive-device ranking.

The unchanged small-image controls use mean synchronized training-update time over all 600 updates, including startup; both arms retain all 24 quality observations. Wider controls exercise BF16 and the graph path.

| Image task | Control ms/update | Candidate ms/update | Speedup | Control / candidate gate |
| --- | ---: | ---: | ---: | --- |
| Stripes, original width 12 | 35.56 | 15.48 | 2.30× | PASS / PASS |
| Bars, original width 12 | 35.29 | 15.38 | 2.29× | FAIL / FAIL |
| Stripes, width 64 diagnostic | 27.02 | 19.23 | 1.40× | PASS / PASS |
| Bars, width 64 diagnostic | 26.27 | 19.30 | 1.36× | PASS / PASS |

Both wider candidate runs finish with HQ 1.0 and all required modes: two stripe modes and four bar modes. Their terminal passing suffixes are 12 and 9 observations, above the required 5. The original narrow bars failure remains a failure. Matching initialization, initial RNGs and final consumed data streams are verified for all four image pairs.

## Quality and policy scope

The combined numerical comparison is **19 PASS / 11 FAIL**, with all 19 incumbent passes retained and no repaired failure claimed. All 26 nonimage final-metric hashes exactly match the archived incumbent. The four ordinary image endpoints differ numerically, while retaining their original gate outcomes.

The ordinary **revision 12** study completes every runnable task in the current tier: **17 PASS / 11 FAIL** over 28 tasks. Tier 1 passes 6/6; Tier 2 passes 10/14 counted tests. Gaussian stability and the three 100-mode acquisition tests fail as they did for the incumbent. The seven held failures remain visible. The source-bound ordinary qualification is **Tier 1, overall FAIL**, with 16/21 counted requirements satisfied and ordinary Tier 3 blocked.

The incumbent’s archived Tier 3 qualification used revision 11, before those four failures were restored as requirements. A separately declared endurance-preservation diagnostic passes the original **1,200/1,200 hold checks and 300/300 extension checks**, completing the unchanged first-convergence stop rule at update 3,282. Those two passes supply regression evidence for the original task contracts and **confer no revision 12 Tier 3 qualification**. The existing verified-parent publication bridge independently validates the word hold’s own restored acquisition checkpoint; all 25 word-hold checks pass.

## Implementation and diagnosis

The current quintic winner spends about 53 ms of its 336 ms 256-pixel update in optimizer polynomial work; convolution forward and higher-order backward dominate. At 64 pixels the optimizer polynomial costs about 15 ms of a 50 ms update. Small kernels still launch one native SVD per spatial offset. The [component profile](profiling.json) keeps unwrapped timing separate from synchronized diagnostic spans and verifies identical plain/instrumented replay states.

One global recipe changes two fields:

- `optimizer_svd_backend="batched"` batches native, rank-masked, absolutely smoothed SVDs for CUDA channel matrices of at most 32 channels. Larger quintic kernels replay the existing FP32 polynomial in optimizer-local CUDA graphs. Graph inputs use contiguous storage, which can change reduction rounding for strided kernels. Dense, bias, sampled-prior and CPU updates retain their original laws. Version-1 checkpoints bind the backend; graph caches are discarded on copy/restore.
- `training_precision="bf16_channels_last"` uses BF16 autocast and channels-last storage for CUDA G/D networks containing a convolution whose largest channel-matrix dimension exceeds 32. Only network forwards autocast. Outputs, losses, master parameters, optimizer arithmetic and SBNP prior regularization remain FP32. Evaluation remains FP32; smaller and nonconvolutional networks keep their original precision.

Learning rates, momentum, Born–Infeld penalty, learned-prior policy and SBNP regularizer are unchanged. Every comparison keeps task architecture, target/data law, seen batches, initial prior law/capacity, public deterministic initialization, seed 0, sampling, update allowance and evaluation cadence fixed. Constructor, data, training-noise and evaluation streams remain separate and checkpointed. Batched SVD and layout/precision changes are explicit numerical policies, without a general bitwise-equivalence claim.

```python
from particlegan import get_recipe

recipe = get_recipe("bispf_optimized", optimizer_svd_backend="batched",
                    training_precision="bf16_channels_last")  # also get_recipe() in 0.9.1
```

Use this default for the measured CUDA image workloads. The 0.9.1 release adds a native-BF16 capability check, falling back to FP32 network execution on unsupported CUDA devices. That fallback is software-tested and supplies no measured speedup or trained quality claim for those devices. Profile the candidate’s remaining 32/64-pixel overhead before another bounded implementation change; no optimizer retuning or seed study is warranted by this result. Natural-image quality is not measured by the synthetic timing workloads or the simple wide-image controls.

## Evidence, budgets and reproduction

The measured implementation was frozen at `bbd95834c`, based on `origin/develop` `de43f22f0`. Scientific source digest `014d8783a85da5a2814e278d59f2e6749620d79fafdbb884cea66a0183e022f8` matches every captured source file in the original publication checkout `72db8af31`. The 0.9.1 default promotion and unsupported-device fallback are subsequent source changes; they do not rebind the frozen experiment evidence. Original receipts, source snapshots, checkpoints, stdout, JSONL streams and profiler traces remain in the local artifact archive. Published receipt projections are display/provenance summaries, not substitutes for original qualification evidence.

- [All final metrics, matched-pair proofs, current-policy qualification and original archive receipts](results.json).
- [Source integrity](source-integrity.json), [unchanged leaderboard/pin proof](publication-integrity.json), and [software checks](software-checks.json).
- [0.9.1 default-promotion verification](release-verification.json) and its [reproduction source](verify_release.py): exact recipe binding, two changed scientific source files, unchanged training body outside the capability gate, and 190 passing release checks.
- [Development probes](probes.json), including the first probe’s mismatched preliminary declaration and the retained layout-test failure. They grant no quality credit; the final registered comparison has separate correct bindings.
- Actual-training GIFs: [small stripes](media/bispf-throughput-v1/img_stripes2.gif), [wide bars](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bispf-throughput/media/bispf-throughput-v1/img_bars4_w64_optimizer_diagnostic.gif), [256-pixel finite updates](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bispf-throughput/media/bispf-throughput-v1/dcgan_performance_r256.gif), and [endurance checks](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/25f915a0f166e4f57a271f352a5c89447bd33c1a/media/forge/bispf-throughput/media/bispf-throughput-v1/ring_extension.gif). Publication adds no updates or sampling draws.

| Campaign | Charged worker seconds | Frozen cap seconds |
| --- | ---: | ---: |
| `bispf-throughput-endurance-v1` | 152.74 | 5,400 |
| `bispf-throughput-performance-control-v1` | 64.73 | 1,200 |
| `bispf-throughput-performance-v1` | 60.99 | 1,200 |
| `bispf-throughput-quality-v1` | 2463.78 | 46,620 |
| `bispf-throughput-small-v1` | 85.99 | 7,200 |
| `bispf-throughput-wide-v1` | 81.62 | 1,200 |

Initial profiling additionally reserves 480 seconds and each of the three development probes 240 seconds. There are no paid retries. Small and endurance diagnostic view revision 1 bytes remain archived; revision 2 changes only family-page presentation. Frozen requests retain their original view identities.

Use the original publication checkout `72db8af31` and the project environment with the experiment dependencies to regenerate its reporting from the existing archive, without training:

```sh
python reports/forge/bispf-throughput/publish.py \
  --artifacts /mnt/ml7tb/ParticleGAN-forge/bispf-throughput-20261017
python reports/forge/bispf-throughput/provenance.py \
  --artifacts /mnt/ml7tb/ParticleGAN-forge/bispf-throughput-20261017
```

The registered idea, studies and performance manifests under `configs/forge/` are the reproduction declarations. Exact original execution uses the frozen request/source snapshot and original view identities, rather than a new run silently substituting a current policy. A new source or binding needs its own study identity.

```sh
tail -f /mnt/ml7tb/ParticleGAN-forge/bispf-throughput-20261017/progress.log
```
