# triton_attention

Current repaired Python: [triton_attention.py](../triton_attention.py).
Existing address candidate: [triton_attention.py](../triton_attention.py).

[Kernel bindings](Support/KernelEntryAdapters.lean) · [complete pasted bodies](../Kernels.lean) ·
[input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) ·
[FP equivalence](../FPEquiv.lean).

The source module contains the complete original forward, gradient-preprocess
and backward bodies, plus the candidate preprocess body. The candidate changes
only the fixed-128 preprocess row base from multiplication to a shift.
Its unchanged forward/backward functions share the same source declarations.
[source correspondence inventory](../../../../validation/source-correspondence.json) checks all six JIT pairs.

Pointer/scalar types and constexpr values are supplied outside the bodies.
The backward host call fixes DQ, saved maxima and scalar scratch to FP32, and
passes equal BLOCK_M/BLOCK_N tile sizes. The source retains both distinct names;
the typed call supplies that existing configuration explicitly. Python names
including `do`, all casts, barriers, masks and pointer advancements remain in
the quotations. The frontend preserves list-form block advances and resolves
only definitionally equal immutable shape constexprs. FP32 storage-to-compute
conversion at an adjacent loaded `+=` remains explicit in the ordered model;
no arbitrary floating cast is assumed to be an identity.

The independent Real specification is causal attention and the complete Q/K/V
VJP, using sums over the full logical sequence and head dimension. The public
judgment keeps all three actual launch schedules, saved scratch, initialized
in-place DQ, source safety and the exterior memory frame. The FP entry compares
the complete actual programs on the original scalar ABI and native primitive
interfaces, with an empty experimental rewrite rule set.

[Proofs](.) contain execution, safety, scalar/native bindings and the
mathematical connections. [Sources](Sources) contain independent mathematical
and geometry definitions. [Validation records](../../../../validation) keep
source correspondence, local proof validation and official independent
acceptance separate; a successful build does not complete this case.
