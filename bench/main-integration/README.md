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

## User-approved release scope

On 2026-10-30 the maintainer chose to publish the already verified scope and
leave other work explicitly pending. `verified-scope.yml` is the main release
gate. The old Artifact and Bench audit workflows remain manually runnable
diagnostics for the historical all-target/original-program requirements.

The selected build contains 6,169 roots: the previous 6,166-root set plus the
previously unlisted DequantizeRowwise Contract and SwiGLU Correct/FPEquiv public
entries, now explicitly rebuilt. Its local source closure is pinned by hash.
All 124 body-migrated tasks have their public entries inside that closure; 59
other task entry sets remain pending for this release. Body correspondence is
124/183 tasks and 19/19 worked examples. All 183 task dispositions are checked;
omission or silent promotion fails the gate.

The Lean audit enumerates actual theorem objects in the selected modules,
including private helpers, and checks their dependency union against only
`propext`, `Quot.sound` and `Classical.choice`. This is a Lean build and axiom
audit, not independent hardware validation. The Guarded program IO and Proof
comparator integration workflows retain their official comparator checks;
their results are limited to their respective interfaces and fixtures.

Selected Python regressions have an explicit complete test-file inventory.
The old `test_exact_numerical_bindings.py` remains pending because its
historical authority binds an older Structural.lean; it is still run by the
manual historical Bench audit. Its frozen hashes and assertions are unchanged.
The selected suite's three immutable metadata fixtures (about 5.5 MB) are
tracked locally so release CI does not download the large historical archive.

`legacy-diagnostics.json` records the interrupted broad build and its old
module failures. The historical original-program ledger separately records
172 pending reviews, 10 verified with deferred numerics and one verified.
Those original-program statuses, the remaining body migrations and imported
benchmark drafts are not promoted by this release.

Website coverage uses the source-bound selected inventory and displays both
selected and pending entries. The main dependency ranges and existing security
overrides are retained; the lockfile is refreshed within compatible versions
and adds zod/undici patch-range overrides for the current advisory checks.
Fresh integration results are recorded separately from archived receipts.
