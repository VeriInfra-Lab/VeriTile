# fp-examples integration into main

The application source is taken from `fp-examples` at
[`d67b1d0444c2e0fb8066414c2041129b53bf2931`](https://github.com/VeriInfra-Lab/VeriTile/commit/d67b1d0444c2e0fb8066414c2041129b53bf2931),
after the ordered merges of PRs #25–#36. The integration starts from main at
`16abbb7e77bf111bf7ede620ec6bf91744e7325b`; it preserves main's repository-link
migration and website dependency update. The seven text conflicts were all
link migrations applied to the complete fp-examples documentation.

This is a content integration onto main, so the large historical snapshot
commits do not enter main's ancestry. The original branch is preserved at
`codex/fp-examples-archive-20261030`, fixed at the commit above. The original
PR #22 archive at `5ea0e46126cc1c6eb05e4219ddc5dc13246ec3ce` is also retained.
Commit-addressed links are authoritative; neither archive branch should receive
new work.

## Current source and historical evidence

All current Lean/Python application source remains tracked. The current
formula, IO, reader and body-correspondence checks run without restoring the
archive. `archive-summary.json` records the reduced checkpoint footprint;
`archived-paths.json.gz` records every moved path, byte count and Git blob ID.
The full archive preserves unsuccessful attempts as well as successful runs.

The historical audit suite reads old receipts, compiler outputs and source
snapshots. These are restored from the fixed archive before running that suite:

```sh
python3 scripts/restore_historical_evidence.py --download
python3 scripts/restore_historical_evidence.py --check
```

Offline alternatives are `--git-repository /path/to/repository` containing the
archive commit, or `--archive /path/to/VeriTile-commit.tar.gz` downloaded from
GitHub for that exact commit. `historical-inputs.json.gz` lists the restored
members and their Git blob IDs. All recovered bytes are verified before the
batch is published, and differing local files are never overwritten. Generated
historical files are ignored by Git. Current selected checkpoint records remain
tracked. No source or receipt hash is refreshed to pretend an old audit applies
to changed code.

The website walkthrough now reads the current Python/Kernels/Correct files.
Its proof recorder compiles the actual kernel, helper proofs and contract
in a temporary standalone module. The positive example must compile, changing
addition to subtraction must fail with an unsolved proof obligation, and both
displayed sample outputs are proved separately.

## Validation and remaining integration blockers

The original stopped migration scope remains 124/183 selected tasks, 19/19
worked examples, with 59 task migrations still pending. This integration does
not expand that scope or claim to complete the imported benchmark drafts.

The old original-program acceptance ledger is a different scope: it contains
172 pending tasks, 10 verified with deferred numerics and one verified task.
The historical global gate rejects pending original-program reviews, as before.
In addition, its October 16 catalog no longer matches current source files,
and the historical numerical authority rejects the changed
`VeriTile/Triton/Float/Structural.lean` hash. Those gates remain enabled.
They must not be made green by deleting pending cases, refreshing unreviewed
hashes, or treating the body-correspondence count as full original-program
acceptance. The PR remains draft while main's required validation is unresolved.

Fresh integration results are recorded separately from the archived receipts.
