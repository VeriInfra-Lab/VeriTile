# bgmv_expand_slice

Current repaired Python implementation: [bgmv_expand_slice.py](../bgmv_expand_slice.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

For each active batch/output row, sum over logical K of X[b,k]*gatheredW[b,n,k], plus the initial Y[b,n] exactly when ADD_INPUTS is true. Signed -1 rows are outside the output write mask.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` copies the complete repaired `_bgmv_expand_slice_kernel` body,
including its 13 top-level statements, leading docstring and every nested
statement. Caller dtype, runtime stride and constexpr bindings stay outside the
quotation. Signed casts, the early return, EVEN_K, CAST_TYPE, ADD_INPUTS, the
signed range, pointer dtype query, masks and implicit output conversion remain.
Both signed32 and signed64 metadata variants have complete-kernel equality
certificates in `Proofs/MetadataStorage.lean`.

`bgmv_expand_slice_output_first.py` is a new Python realization of the retained
output-first addition comparison. Only the optional addition operands change.
The FP reference and IO graph follow the actual operations: the extra
FP32-to-FP32 cast from the former handwritten Lean source is removed, with no
cast-identity premise added. The existing strict software FP32 addition
certificate remains the sole rewritten numerical operation.

The source retains the CAST_TYPE branch selected by the checked caller.
`SignedProgram.lean` extracts proof-only blocks and proves both execution equality
and access-safety equivalence for its constant-true branch before using the
loop decomposition. The canonical kernel is never assembled from these views.
Full actual-source execution, tagged outputs and exterior frames remain the
public conclusions; no hardware-conformance or numerical-experiment claim is made.

This migration stays pending until independent proof replay accepts the exact
current source and import closure. Matching bodies or compiling is insufficient.
