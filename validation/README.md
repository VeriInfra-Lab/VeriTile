# Validation records

- `task_progress.json` selects the 183 current repaired tasks.
- `contracts.json` records reviewed specifications, conditions and numerical scope.
- `source.json` binds current sources and imports; `source-correspondence.json`
  separately records pasted-body coverage and pending translations.
- `imported_kernel_review.json` preserves findings for the 334 imported challenges.
- `bench-layout.json` records the path migration and unchanged Lean/JIT code.

Task-specific compiler and numerical receipts are under `tasks/`. Their JSON
bytes, source hashes, outcomes and original paths are preserved. Resolve those
original paths against [the source snapshot before the layout migration](https://github.com/VeriInfra-Lab/VeriTile/tree/4edf710ba663205d9bbc16ed1efe12ea1a72a803); they are historical evidence,
not claims that an experiment or compiler probe was repeated after moving files.
Current proof/source bindings are the top-level records listed above.

Prior runnable Python implementations and retired task validators remain in Git
history. The live benchmark tree exposes only each task's current repaired program
and the comparison candidates/components required by its present FP interface.
