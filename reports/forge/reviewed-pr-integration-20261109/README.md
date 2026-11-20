# Reviewed model PR integration

This integration combines PRs #460 (optional DDGAN conditional energy), #461
(VAE observation checkpoints and explicit transport admission), and #462
(independent MisGAN critic histories) with develop after #473 and #474.
The original heads are recorded in [the source binding](source-binding.json).

All three mechanisms remain explicit and optional. R1/R2 weights and joint
reconstruction from #474 remain available. Direct model-provider calls reject
active objectives that their host cannot execute instead of silently dropping
an objective introduced by another PR.

## Evidence and execution scope

The new source bindings authorize future execution against the combined code.
They preserve exact parent task commits, Git blobs, SHA-256 hashes and source
maps. Numeric gates, execution contracts, architectures, data and sampling laws,
initialization, budgets and observation cadence retain their parent declarations.
Previously stale source contracts remain stale; this merge does not readmit them.

Original research results, readouts, per-PR integration receipts and the MisGAN
v2 contract remain byte-for-byte unchanged. Corrected MisGAN v2 quality is still
unmeasured. Numerical failures and incomplete results remain visible. This
software integration launches no scientific training, regrades no historical
result, grants no qualification and changes no selected recipe or public default.

## Validation

Focused controls cover the combined recipe schema, numerical objectives,
unsupported-host refusals, checkpoint replay, exact source ancestry, tampered
sources and preserved evidence. The full repository CI runs on the combined
source before merge. Execution logs remain outside Git under
`/tmp/particlegan-model-pr-integration/` and in GitHub Actions.

The canonical [technique inventory](../technique-inventory.md) is refreshed from
retained records without rerunning or regrading the archived research.
