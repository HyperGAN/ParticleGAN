# Retrieving original Forge evidence

`python -m experiments.forge artifacts` verifies and restores original bytes. It
does not train, regrade, change qualification, or substitute a compact summary
for original evidence. A hydration receipt records the archive SHA-256, executed
commit, selected members and their hashes. Keep hydration receipts and restored
logs, tensors and checkpoints in ignored `runs/` or an artifact archive.

The resolver understands the existing inventory/release archive cards (flat
`archive`, `archive_sha256`, `files`, `full_readouts`), frozen-round cards
(`archive` object and receipt/source manifest), and policy/capacity/setup cards
(`path`, `sha256`, `files[]` containing `archive_path`). It preserves their
original source and receipt identities. Policy `original_path` values document
provenance; extraction uses the safe relative `archive_path`, never an absolute
path on the author's machine.

Lookup uses these locations, in order:

1. `<checkout>/runs/forge/artifacts/sha256/<SHA-256>.tar.gz`.
2. The card's original location (relative locations use the checkout root).
3. Explicit paths from `--locations <JSON>`.
4. Mounted mirrors provided with repeated `--mirror <directory>` or
   `PARTICLEGAN_FORGE_ARTIFACT_MIRRORS` (OS path separator). A mirror contains
   `sha256/<SHA-256>.tar.gz` or `<SHA-256>.tar.gz`.

Every location must match the committed archive digest and declared byte count.
A corrupt location cannot supply evidence; lookup can continue to another valid
location. Mirrors are user-configured local or mounted storage. By default,
there is no implicit network transfer, credential discovery, guessed remote
bucket, or automatic Git fetch. Obtain the exact archive from existing storage
and place it at one of these locations, or explicitly opt into Hugging Face
hydration below. A hash identifies bytes; it does not promise that the archive
exists.

### Opt-in Hugging Face hydration

Remote Hugging Face downloads are strictly opt-in and never occur unless you
pass `--allow-hf`, pass `--hf-repo <owner/repo>`, or set
`PARTICLEGAN_FORGE_HF_REPO`. Local candidates (cache, card path, `--locations`,
and `--mirror`) are always checked first; if a verified local copy exists, no
network request is made.

When opted in and no verified local copy is found, the resolver locates the
archive using either:

- An optional `hf` locator committed in the archive card:
  ```json
  {
    "hf": {
      "repo": "HyperGAN/particlegan-forge-evidence",
      "revision": "<40-char-commit-sha>",
      "path": "archives/sha256/<64-char-sha256>.tar.gz"
    }
  }
  ```
- Or explicit `--hf-repo <owner/repo>` (and optional pinned `--hf-rev <40-char-sha>`,
  or `PARTICLEGAN_FORGE_HF_REPO` / `PARTICLEGAN_FORGE_HF_REV`), which resolves
  `archives/sha256/<64-char-sha256>.tar.gz`.

The downloaded archive is streamed into `<checkout>/runs/forge/artifacts/sha256/<SHA-256>.tar.gz`
and verified against the committed SHA-256 digest, declared byte size, tar
path/type safety checks, and declared per-member hashes before any file is
cached or extracted. Corrupt remote downloads are rejected as `INVALID` and
never written to the local cache or destination.

```sh
# Opt-in HF hydration using a card's pinned hf locator (or --hf-repo/--hf-rev):
python -m experiments.forge artifacts hydrate \
  reports/forge/family-winner-round1/phase1-archive.json \
  --allow-hf \
  --hf-repo HyperGAN/particlegan-forge-evidence \
  --hf-rev <40-char-commit-sha> \
  --destination runs/forge/hydrated/phase1-originals
```

Requires the optional `hf` extra (`pip install 'particlegan[hf]'`).

In particular, the published first-round `/ml2/hypergan/...` archives are
unavailable on this machine. The resolver reports `MISSING` until a real copy is
configured. No new remote retention or availability is claimed.

```sh
# Verify original receipt/source members; this performs no writes.
python -m experiments.forge artifacts inspect \
  reports/forge/family-winner-round1/phase1-archive.json \
  --mirror /mounted/research-artifacts

# Restore all declared original request/evidence/result and source members
# into a NEW isolated tree. The current checkout is never overwritten.
python -m experiments.forge artifacts hydrate \
  reports/forge/family-winner-round1/phase1-archive.json \
  --mirror /mounted/research-artifacts \
  --destination runs/forge/hydrated/phase1-originals

# Select one checkpoint or receipt by its exact archive member path.
python -m experiments.forge artifacts hydrate \
  reports/forge/family-winner-round1/policy-round4-archive.json \
  --member archive/<family>/<configuration>/<case>/final-state.pt \
  --mirror /mounted/research-artifacts \
  --destination runs/forge/hydrated/selected-state
```

Without `--member`, restoration selects every individually hashed member in the
card. Extra queue checkpoints/logs in an inventory or frozen-round archive can
be selected explicitly by their exact archive member names. Their integrity is
reported as `archive_only`: the entire archive was verified, and their own
SHA-256 is recorded, but the card has no independent per-member hash for them.
This distinction grants no qualification. Original receipt/source members and
policy state entries with declared hashes report `member_and_archive`.

Hydration validates all archive member paths and types, even unselected members;
it rejects path traversal, duplicate names, symlinks, hardlinks and devices.
Selected hashes are verified before publishing the complete tree. An existing
destination, including an empty directory or symlink, is rejected. A failed
verification leaves no published destination. Do not unpickle retrieved states
unless you trust their recorded publisher; retrieval never loads a checkpoint.

For an independent regrade, first create an isolated checkout of the recorded
executed commit. Restore its complete original envelope and frozen source;
verify its scientific binding with the original evaluator. The resolver itself
verifies bytes, not causal claims or scientific compatibility. Partial hydration
does not prove that every evaluator dependency is present. Existing pinned
reproduction instructions remain authoritative; missing artifacts block the
analysis rather than trigger new training.

Git-pinned originals can also be restored when the exact object is available:

```sh
python -m experiments.forge artifacts git \
  --commit <full-archive-commit> --path <original-relative-path> \
  --blob <full-original-blob> --sha256 <original-byte-sha256> \
  --destination runs/forge/hydrated/git-original
```

All three identities are checked: the commit/path must reference the declared
blob, and the blob bytes must match SHA-256. Missing objects produce an explicit
`git fetch origin <commit>` recovery instruction. Never replace the pinned
commit with a branch name. Git can restore only tracked originals; a Git source
commit cannot retrieve an externally archived checkpoint. Retired publications
remain summaries even when their exact Git bytes are restored.

Retention belongs to the experiment publisher and the configured archive owner,
not to the resolver. Before deleting a local execution envelope, the publisher
must verify that its complete archive is accessible, publish its exact digest
and member mapping, and assign ownership/retention in the storage configuration.
Keep exact old archive commit/blob IDs when relocating tracked evidence and
repair report links. Do not shorten retention solely because a study failed.
Older cards without a declared owner/retention report `unassigned`/`undeclared`;
this is an explicit operational gap, not an availability guarantee.

Example local configuration (untracked; relative paths use this file's directory):

```json
{
  "archives": {
    "<64-character-archive-sha256>": {
      "paths": ["/mounted/research-artifacts/sha256/<sha256>.tar.gz"],
      "owner": "<actual-team-or-storage-owner>",
      "retain_until": "<actual-retention-date-or-policy>"
    }
  }
}
```

Outcomes are machine-readable JSON: `AVAILABLE` or `HYDRATED` exits 0; `MISSING`
exits 2 with exact lookup/recovery guidance; `INVALID` exits 3 for a malformed
card, checksum mismatch, unsafe archive or existing destination. The tests in
`tests/test_forge_artifact_resolver.py` build a real tar fixture and verify full
original envelope/source hydration from a fresh directory using a configured
mirror, selected checkpoint retrieval, Git originals, and unsafe/corrupt archive
rejection. The fixture is synthetic; no claimed research evidence is rerun or
regraded, and no raw artifact is committed.
