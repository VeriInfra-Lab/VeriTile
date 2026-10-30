# TritonBench-G

Each of the 183 task directories under `bench/tritonbench_g/` contains one
current repaired Python implementation, `Kernels.lean`, and canonical
`Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries. Required source,
mathematical and proof helpers support that implementation. Earlier runnable
implementations and version directories are retained in Git history.

The [selection ledger](../../validation/task_progress.json) identifies every
current entry. Use the [corpus workflow](current_source_acceptance.md) for
builds and checks; [contract review](../../validation/contracts.json) records
scope and numerical conditions. Source correspondence and pending work remain
separate from Lean compilation.
