# Review evidence

Start with the [current source inventory](../current_source_acceptance.md) and
the [184-task progress audit](2026-10-15/development-progress-audit/README.md).

Git retains the current source/catalog dependencies, immutable review receipts,
scope summaries, the requested backward real/typed checkpoint, and the current
experimental tools with their reference data and tests. The frozen metadata
baseline is reconstructed from its checked `inputs.zip` by the command in the
current inventory; expanded copies of that tree are not needed in Git.

The 2026-10-15 cleanup moved historical full-tree copies, old replay output and
intermediate attempts to the ignored local directory
`.local-review-archive/2026-10-15-before-cleanup/`, preserving their relative
paths. Its `archive-files.json` and `move-operations.json` record the move. The
previous unpublished commit is also retained locally on
`codex/fp-examples-before-cleanup-20261015`.

Historical receipts are preserved byte for byte. Their references describe the
original evidence package; they do not promise that every old run's artifacts
ship in a fresh checkout. Whole-package historical verification scripts are
archived with their inputs. Current reproduction and test checks are recorded
in the [cleanup report](2026-10-15/development-progress-audit/cleanup.md).

Only the local archive directory is ignored. New source files and future review
checkpoints remain visible to Git. When adding evidence, include the inputs
required by the current checks and tests, rather than recursively adding every
path mentioned by an old receipt.
