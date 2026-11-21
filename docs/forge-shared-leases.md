# Request ownership and external fleet leases

See [queue state and ownership](forge-queue-state.md) for the distinction between
submission status, shared jobs, process leases, scientific grades and cost owners.

Normal Forge admission, tier barriers, budgets, original evaluators and
source-bound receipts remain authoritative. A bounded scheduler that already
owns an external fleet can retain its open lease descriptors through ordinary
Forge drain, worker supervision, training and independent grading:

```python
subprocess.run(
    [python, "-m", "experiments.forge", "--queue-root", queue_root,
     "drain", "--gpus", "0,1", "--request", request_id,
     *[argument for fd in owned_fleet_fds
       for argument in ("--shared-fleet-fd", str(fd))]],
    pass_fds=tuple(owned_fleet_fds),
    check=True,
)
```

The owner acquires those leases before launching drain and supplies the actual
open descriptors explicitly. Drain checks them before claiming work and passes
them to every child. Queue launch clears ambient shared-descriptor variables;
it cannot silently reuse another process's environment as ownership. These
options create no leases, change no allocation, add no budget and start no
model session. The caller still owns its finite campaign and cancellation.
Requests using external leases must freeze the current FD-forwarding worker;
an older snapshot is refused before launch rather than silently dropping
ownership at the training/grading hop. Ordinary launches without external
leases retain their original source compatibility.

`--request` restricts both claims and drain completion to that exact submitted
request. Other pending requests, including requests in the same campaign,
remain available to their own coordinator. A required lower-tier failure still
blocks higher tiers through the existing queue policy.

The lease descriptors share the original open-file descriptions. Closing a
coordinator descriptor, or killing a supervisor, does not unlock a surviving
training/grading child's descriptor. No explicit unlock, file replacement or
inode change occurs in this path. Leases become available after the last
holder exits. The external owner must close its descriptors rather than call
`flock(LOCK_UN)` while children survive: an explicit unlock releases the shared
open-file description for those children too. Descriptor inheritance supplements the existing independent
attempt lease and never substitutes for normal task resource checks.

This is the independently verified software portion extracted from
[PR #407](https://github.com/255BITS/ParticleGAN/pull/407). The full proposed
runner remains archived at
[`44094c769464fcf3df6cf588f36a5c8916d4d461`](https://github.com/255BITS/ParticleGAN/tree/44094c769464fcf3df6cf588f36a5c8916d4d461),
including its [loop design](https://github.com/255BITS/ParticleGAN/blob/44094c769464fcf3df6cf588f36a5c8916d4d461/docs/grok-build-loop.md).
The local cancellation receipt dated `2026-10-15T02:14:42Z` records the user's
instruction to stop the Grok idea, zero provider requests and zero new training
jobs. Its STOP remains in place. This extraction does not resume that work or
publish any new scientific pass. The original revision 9 gauntlet and existing
tier leaderboard retain their gates and scores.
