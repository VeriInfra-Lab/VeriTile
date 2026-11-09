# bmm_optimized

Current repaired Python implementation: [bmm_optimized.py](../bmm_optimized.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

For each logical batch/row/column, sum k=0..shape.k-1 of A[batch,row,k]*B[batch,k,column]. Logical k is independent of all block/tile dimensions.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` copies the complete `bmm_kernel` body from `bmm_optimized.py`,
including all eight constexpr mask combinations, explicit `None` mask values,
positional mask arguments, batch pointer updates, the original tuple assignment
and anonymous loop counter. The previous eight expanded source copies are
replaced by this one quotation. The existing integer quotient/remainder
candidate is recorded in `bmm_optimized_candidates.py`; only the group-remainder
expression changes. Caller storage, dimensions and heuristic flags are external.

Boolean constexpr specialization selects only source compile-time branches.
The generic frontend resolves immutable `None` mask aliases as absent masks;
it retains runtime branches, loops, loads, pointer advances, dot instructions,
`allow_tf32=False` and stores. Complete-program reference tests cover masked and
unmasked reads/writes and every original loop iteration. Mutation, shadowing,
runtime-defined None aliases and alias rebinding are rejected.

The existing model retains the live FP32 accumulator/FMA order. Real correctness
connects the complete source grid to independent batched matrix multiplication;
FP equivalence compares both actual complete source schedules with the same
numerical interpretation, active cells and exterior frame. Validation progress
records source correspondence, retained proofs and independent replay separately.
