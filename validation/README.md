# Validation records

- `task_progress.json` selects the 183 current repaired tasks.
- `source-correspondence.json` records pasted-body coverage and pending translations.
- `body_receipts/benchmark-layout-replay.json` records independent proof replay for the standardized task layout.
- [`archive/contracts.json`](archive/contracts.json) preserves the earlier review of specifications,
  conditions and numerical scope. [`archive/source.json`](archive/source.json) preserves the earlier
  source and import inventory. Generate a current inventory with
  `python3 scripts/bench/final_inventory.py --write PATH`.
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
