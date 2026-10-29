# embedding_triton_kernel

`Kernels.lean` contains the complete signed-int32-ID `embedding_kernel` and
its integer-index commuted candidate. Weight/output precision matches at
fp16, bf16 or fp32. The whole strided loop, vocabulary checks, row/column
masks, token default `vob_end_id` and zero weight default are retained.

`Correct.lean` displays two real specifications: each in-range signed token
reads its row of the initial weight matrix; out-of-range tokens output zero.
They prove successful finite flat execution, every active output and the
complete tagged-cell frame. Chunk sizes are arbitrary, including multiple
rows per iteration and a last chunk extending past `BLOCK_N`. Only the actual
`n_ctx` and hidden-column masks suppress stores. Output stride need cover
`min(BLOCK_DMODEL, hidden_size)`, so padded columns may exceed the stride.

The two input regions must differ from output, ensuring later iterations
still read initial inputs. Actual address bounds, finite allocation/coverage
and zero unspecified values for the flat adapter are explicit. The masked ID
default is implicitly converted to signed int32. If that conversion wraps into
the vocabulary interval, a masked output row still reads weights. `Bounds` now
covers these actual reads; it does not require `vob_end_id` to fit int32. Zero chunk
stride follows formal range semantics; it is not a Triton compilation claim.

`FPEquiv.lean` proves complete typed equivalence by commuting only the natural
index product `program_id * BLOCK_N`. It preserves states and failure without
floating-point atoms or numerical guards; assumption output is `none`.

`bench/tests/EmbeddingTritonExecution.lean` instantiates the public contract
with two-row chunks, `BLOCK_N=3`, padded width four and physical stride two.
It covers negative invalid and valid IDs, the overshooting final row and an
integer-tagged untouched tail. With the representable default 1, an all-masked
run uses empty allocations. A full-source counterexample proves empty weights
are unsafe when default `2^32` wraps to zero; a companion witness executes both
versions with one weight cell and no ID/output allocations, preserving memory.
`bench/tests/FPEmbeddingTritonSource.lean` checks actual signed comparisons,
subtraction, storage precision, load defaults and masks. The inhabited typed
model in `Proofs/TypedExecution.lean` runs three complete iterations from
empty registers for all storage choices and both candidates, including a
masked row and column. This does not certify physical GPU rounding.

Scope is one program, excluding host launch correctness, concurrent races
and fixed-width index overflow.

Current Lean validation: [embedding and frontend regression receipt](../../review-checkpoints/2026-10-14/embedding-final-regressions/README.md)
records 3,508 successful Lake jobs and 219 unchanged inputs, including both
public proof entries, typed execution and all registered frontend regressions.
[Offline native evidence](../../review-checkpoints/2026-10-14/embedding-native-default/README.md)
compiles the unchanged Python JIT for six storage/runtime-integer profiles and
confirms the i64-to-i32 default truncation. It does not execute a GPU or certify
native refinement. Official independent replay is currently blocked by this
container's systemd/Landlock isolation support. Historical comparator records
do not certify these updated inputs; see the [current acceptance report](../../review-checkpoints/2026-10-14/ACCEPTANCE.zh-CN.md).
