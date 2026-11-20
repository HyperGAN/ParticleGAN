# PR #474: audited develop integration

Integrated develop `5a905a254bf5`, including scientific-contract audit #535. Its source-bound correction registry, distinction between scientific and operational outcomes, and strict current-measurement policy are preserved. Retained publication rows are refreshed from their original records without a scientific rerun or historical regrading.

The [source receipt](pr-474.json) preserves exact parent task commits, blobs, hashes and source maps. Updated bindings authorize future execution only; original results remain attached to their original source and contract. This integration grants no automatic quality inheritance.

Validation is software-only. Raw test/CI logs remain outside Git under `/tmp/particlegan-pr-fixes-20261108/models/audit-integration-20261109/` and GitHub Actions.

## Reviewed integration receipts

The combined merge retains all four original per-PR source receipts without changing their bytes:

- [PR #460](pr-460.json)
- [PR #461](pr-461.json)
- [PR #462](pr-462.json)
- [PR #474](pr-474.json)

[MisGAN independent-game v2 contract](misgan-execution-v2.json) remains source-bound and unmeasured.

[Combined integration and software validation](../reviewed-pr-integration-20261109/README.md) records the subsequent source binding.
