# Wall-clock outcomes

A run killed by its task's wall-clock limit was never graded. It is reported as **NOT MEASURED (wall-clock)**, not FAIL, and is excluded (with any task it blocks downstream) from paired recipe-versus-baseline deltas. A same-key solo rerun under the one-job-per-GPU lease replaces the hit only when it produced a real PASS/FAIL verdict under the unchanged declared limit.

Totals: 254 PASS, 295 FAIL, 19 not measured (wall-clock), 8 blocked by an unmeasured parent.

| Recipe | PASS | FAIL | Not measured (wall-clock) | Blocked by unmeasured parent |
| --- | ---: | ---: | ---: | ---: |
| bcap-default-baseline-direction-v1 | 13 | 12 | 0 | 1 |
| cell-quota-dualnorm-cross-v1 | 5 | 6 | 3 | 0 |
| confidence-lr-finite-cap-gaussian-v1 | 11 | 14 | 1 | 1 |
| generator-idle-ema-bcap-direction-v1 | 12 | 12 | 0 | 2 |
| kinetic-local-dualnorm-g-history-bank8-u-v1 | 7 | 9 | 2 | 0 |
| kinetic-local-dualnorm-g-history-bank8-v-v1 | 8 | 8 | 2 | 0 |
| kinetic-local-dualnorm-g-history-reference8-u-v1 | 7 | 6 | 4 | 0 |
| kinetic-local-dualnorm-g-history-reference8-v1 | 6 | 8 | 3 | 0 |
| kinetic-local-dualnorm-g-history-v1 | 11 | 15 | 1 | 1 |
| kinetic_transport_local_v2 | 13 | 13 | 1 | 1 |
| positive-evidence-lr-gaussian-v1 | 13 | 13 | 1 | 1 |
| signed-confidence-lr-gaussian-v1 | 13 | 13 | 1 | 1 |
