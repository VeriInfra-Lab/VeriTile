# destindex_copy_kv1

Current repaired Python implementation: [destindex_copy_kv1.py](../destindex_copy_kv1.py).

Both complete JIT bodies are copied into `Kernels.lean`; the existing
[candidate](../destindex_copy_kv1.py) commutes only the source-row address product.
The original masks, signed int32 destination channel and terminal return remain.

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

destindex_copy_kv1_kernel_real: complete original scatter schedule implements the logical payload copy.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
