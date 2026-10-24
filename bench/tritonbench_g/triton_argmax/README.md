# triton_argmax

- Source file: `triton_argmax.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Typed bit contracts

`FPCorrect.lean` proves successful execution, maximal-value/least-lane-index selection, and memory framing for the global stage1 and stage2 source ports, and for every active row of the dimension-specific single-block source (`0 < N ≤ BLOCK_N`), for fp16 and fp32. The public proofs use actual 16/32-bit encodings and a bit-order comparison, with independently stated maximum and tie specifications. `Proofs/TypedSource.lean` makes the source pointer dtypes and the implicit stage1 int64 store conversion explicit.

The domain excludes NaNs in the padded block; infinities, subnormals, and both signed zeros are included. Equal values, including signed zeros, select the least lane index. Stage2 returns the initial index payload of its selected lane. This is a source interpreter contract, not a compiler/device refinement. The dimension-specific branch uses actual32-bit intermediate encodings for its mixed-format comparison and proves the returned index is below the logical axis length. Composition from global stages to the Python wrapper and arbitrary multi-block dimension loops remain outside the already accepted stage scope. The typed-source shape agrees with the legacy port after dtype erasure; stage1 agrees before its final store, where the missing implicit int64 conversion is explicitly restored. Source provenance and the exact scope are recorded in `fp_source_evidence.json`.
