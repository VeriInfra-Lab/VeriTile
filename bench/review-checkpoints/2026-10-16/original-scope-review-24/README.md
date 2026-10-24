# Original-program scope review: 24 additional tasks

Whole-original-program acceptance remains **0/183**, with **183 pending**.
The ledger now distinguishes **66 tasks with confirmed structural gaps** and
**117 awaiting a full scope review**. This batch adds 24 reviews, checks 79 public
statements and records 49 specific structural gaps. Existing proofs are preserved.

The reviewed cohort is the first 20 remaining tasks plus `var_len_copy`,
`destindex_copy_kv1`, `destindex_copy_kv2` and `square_matrix`. Each evidence record
binds the original source, current proof statements and concrete omitted behavior.
Ordinary bounds/layout premises are not classified as defects. Eligible numerical
correspondences are described without registering or validating them in this batch.

Representative findings are the missing `llrint` and int8 store in the two
quantization programs, unassembled wrapper grids, and dtype routes outside the
current materialized families. Existing rowwise maximum computation, canonical
seeded RNG, complete reverse scans and all 24 nested-loop stores remain credited.

`current-declaration-index.json` separately records the actual source location and
file hash of all 936 current public declarations. It also lists differences from
retained historical coordinates, including the two rowwise quantization records.
Historical catalogs and the original metadata baseline were not rewritten.

The scope gate actually exited 1: all 183 rows remain pending, with no extra
catalog/population/binding error. No Lean, comparator, frontend or GPU execution
was rerun for this source-preserving review. The preceding 13-module batch and
its build/global-gate results remain in `../fp64-admission-relu-access/`.
