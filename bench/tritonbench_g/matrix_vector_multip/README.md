# matrix_vector_multip

The current repaired [Python](matrix_vector_multip.py) and [Lean source](Kernels.lean)
retain all six uint64 scalar casts, program/lane casts, the scalar uint64
`tl.full`, the actual `while m < M`, its final increment, both loads and their
FP32 conversions, and the original implicit output conversion. The explicit
FMA candidate changes only the accumulator update.

The call signature and argument values sit outside the pasted function bodies.
The function bodies can be pasted with their original Python indentation, colons
and comments. The frontend accepts this affine while only when its counter has a unique final
increment and its bound/constexpr step remain unchanged. The checked connection
in `Proofs/SourceWhileExecution.lean` gives ordinary while execution with the
same complete final state, including the counter; it does not assume a whole
kernel result.

[Real correctness](Correct.lean) proves the independent complete M-column dot
product for every active row in an invocation, including zero columns and tails,
with execution, memory safety, readback and the exterior frame.
The invocation must belong to the original grid. `SourceBounds` expose
the representability conditions introduced by the original
casts. `Sources/Bounds.lean` derives them from the existing input contract and
all 48 autotune configurations, and proves counter/valid-grid index bounds.

[FP equivalence](FPEquiv.lean) compares the actual two sources over the full
N-row grid and any permutation of its legal program IDs, including the empty
grid. Its fixed compiler profile uses local product/carry nodes and the selected
per-row reduction tree. This is a binding to a fused compiler carry graph;
it is not an IEEE identity between separately rounded multiply/add and FMA.
No whole-kernel callback supplies the result.

The old complete Lean `for` source and its exclusive one-block executor have
been removed. `Mathematical_WholeSource.lean` contains only pure operands,
ordered recurrence and partial state invariants, connected to the actual source
by the `Source*` execution proofs. `SourceStorageView.lean` is a lossless payload
tag adapter for those pure helpers, not another program.

[Input contract](Contract.lean) · [IO](IO.lean) · [proof helpers](Proofs/)
