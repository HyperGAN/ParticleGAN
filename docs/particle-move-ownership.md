# Particle move ownership and failure boundaries

The recipe owns prior updates and regularization. The task owns the initial
prior distribution, capacity and sampling law. A particle move implements its
recipe's population change inside that contract; it does not redefine sampling
or decide whether an experiment qualifies.

## Owners and ordering

| Responsibility | Owner | Boundary |
| --- | --- | --- |
| Decide which independent rows may be copied | `ParticleBirthDeath`, feature-cell planners | Evidence, capacity, reservations and private reaction RNG authorize a proposal. |
| Prepare independent copy history | `row_transport.prepare_copy_history` | Validate paired row indices and the supported row-state layout; snapshot donor history before any coordinate write. |
| Apply copy coordinates | The active birth/death backend | Copy fast and averaged latents using that backend's existing jitter law. Feature cells share one noise draw but measure each table's own geometry. |
| Apply copy optimizer history | The active birth/death backend | Copy all table-shaped tensor states and optional A2 latent history from the prepared donor incarnation. Scalar and other non-table-shaped state, including Adam age, stays shared. |
| Track overwritten identities and invalidate support caches | `FeatureCellBirthDeath`, `LatentLineage`, `FeatureCellSnapshot` | Register copy links and refresh affected queries after the coordinates move. The kNN backend owns its own locality restarts. |
| Rebase stationarity and row-gradient evidence after a successful reaction | `UpdatePolicy.finish_step` | Consume the backend's moved rows after reaction completion, before advancing the completed-update count. |
| Save a completed update and restore matching owners | `UpdatePolicy` / `GANTrainer`; application for external resources | Particle transport does not own optimizer-step rollback, data cursors or callback state. |

The shared preparation only covers **independent copies**. Dense strided
table-shaped optimizer tensors must be on the table's device; their dtype may
differ from the table, so mixed precision history remains supported. A2 history
must also match the table's shape and dtype. A child is unique within a batch;
donors may repeat or overlap children. Overlapping donors refer to their
pre-copy incarnation, including for optimizer history.

## Distinct population changes

- Novel real-anchor births (`birth_phase.apply_anchor_births`) install certified
  new fast/average coordinates, zero overwritten row history and remove old
  lineage links. They do not inherit a donor's momentum or call copy jitter.
- Routed splits (`RoutedRowControl._commit`) own coupled table/router weights,
  mass and averages. Their separate preflight accepts explicit Adam layouts;
  it transports first moments by one half and second moments by one quarter,
  preserves shared age, and restarts A2 and coupled evidence. The independent
  copy helper must not impose its equal-mass atom law on these splits.
- The frozen feature reference keeps its recorded historical implementation.
  It is not silently migrated to the current feature-cell contract.

## Failure guarantees

For current kNN and feature-cell copies, malformed paired indices, unsupported
table-shaped history layout/device, or incompatible A2 history are rejected
before writing fast/average coordinates, optimizer history or lineage and
before drawing copy jitter. Preparation allocates only the affected donor
rows; it does not snapshot the whole optimizer or training step.

This is validation before publication, not a general atomic commit. Allocation,
device or later callback failures may still interrupt a multi-owner write or a
later reaction phase. A feature-cell evaluation restores its evidence,
evaluation clocks, counters and serial on failure, retains consumed RNG and
already moved rows, and empties its local evidence for changed rows. That
recovery does not certify a completed update or perform the policy's successful
reaction rebase. The caller must follow its trainer/policy recovery contract;
retrying a partially applied reaction is not exact checkpoint continuation.

No new transaction manager spans these different population laws. The narrow
copy boundary leaves acceptance mathematics, sampling, serving eligibility,
experiment grading and application resources with their existing owners.
