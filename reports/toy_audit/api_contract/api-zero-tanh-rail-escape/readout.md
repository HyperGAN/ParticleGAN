# Public API Readout: `api-zero-tanh-rail-escape`

**Goal**: Escape the tanh saturation rail at (+0.9, -0.9) after update 1000 shift and achieve sustained settling at (0, 0) within 400 updates.

**Verdict**: `PASS` (PASS: All criteria satisfied)

## Execution Summary
- **Provider**: `api_vectors`
- **Adapter**: `Public GANTrainer host with 2-layer MLP and nn.Tanh() output head; dynamic target shift at step 1001.`
- **Default Recipe**: `k3p`
- **Protocol Seed**: `0`
- **Completed Updates**: `2000 / 2000`
- **Evaluation Samples**: `4096`
- **Runtime Device**: `cpu`
- **Elapsed Wall Time**: `8.96s`

## Numerical Gate Status
| Metric | Required Bound | Final Observed Value | Status |
|---|---|---:|:---:|
| `mean_abs_error` | `<= 0.005` | `0.0029841` | **PASS** |
| `rms_error` | `<= 0.01` | `0.0042143` | **PASS** |
| `p99_abs_error` | `<= 0.025` | `0.014158` | **PASS** |
| `saturated_fraction` | `== 0.0` | `0` | **PASS** |

## Sustained Terminal Checks
- **Terminal Window**: Last `5` post-update metric observations
- **Sustained Status**: `PASS (all 5 passed)`

## Actual Training Media
![Actual Training Observations](https://huggingface.co/datasets/HyperGAN/particlegan-forge-evidence/resolve/496272b90f1c06483db8ff1846ca5f07186d859d/media/toy_audit/api_contract/api-zero-tanh-rail-escape/goal.gif)

## Provenance & Integrity
- **Source Commit**: `de43f22f0fb743af73f61492de84f8d7d4a01fad`
- **Batch Sequence SHA256**: `5b7940ada600d3cec7f2a01680a3e1f120ebd33d0021024d29c7d54a54358c2b`
- **Initial Model Hashes**: `{"D": "672dfa6da531419cdb7ec109ca7463582b9a0c88c4ded3c9e3cf5dac865c10f5", "G": "dd84fef44174cd9100787029c753244e8f469777decb14182622a3de8de66707", "prior": "96bc02424adf20c7c5f20911e29112262262399c0d263ef7b597cbfbac12fb4a"}`
