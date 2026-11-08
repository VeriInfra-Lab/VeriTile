# destindex_copy_kv2

Current repaired Python implementation: [destindex_copy_kv2.py](../destindex_copy_kv2.py).

Both complete JIT bodies are copied into `Kernels.lean`; the existing
[candidate](../destindex_copy_kv2.py) commutes only the source-row address product.
The original masks, signed int32 destination channel and terminal return remain.

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

destindex_copy_kv2_real: logical row/head/dimension copy under the exact original no-feature-mask width condition.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
