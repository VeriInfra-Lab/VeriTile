# rotary_transform_ops

Current repaired Python implementation: [rotary_transform_ops.py](../rotary_transform_ops.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Explicit conjugated/ordinary rotary prefix a*co-b*signed or a*signed+b*co, with identity coefficient fallback; suffix equals initial input.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The repaired Python files remain unchanged. The complete JIT bodies, including
all casts, masks, early return and source branches, are copied into
[the shared source entry](../../rotary_transform/Kernels.lean). Invocation
metadata outside the bodies distinguishes scalar/tensor offsets and all four
signed int32/int64 metadata-width combinations. Reassigned `seqlen` receives its
actual initial argument before the copied body; reads and later assignments
are part of the proved execution. `Kernels.lean` contains invocation adapters.

The unchanged Python `rotary_kernel_xor` is the existing candidate: only the adjacent partner index changes to `rk ^ 1`.
The original Real specification, native FP scope, input domain and exterior
frames are retained. The existing proofs now execute the copied bodies,
including real temporaries for nested loads and explicit float32 casts.
Source correspondence and local proof validation are recorded separately;
independent exact-source replay is still required before completion.
