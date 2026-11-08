# rotary_transform

Current repaired Python implementation: [rotary_transform.py](../rotary_transform.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

For each logical output: rotated prefix a*co-b*signed or a*signed+b*co, optional conjugation; identity coefficient fallback; unchanged input suffix.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The repaired Python files remain unchanged. The complete JIT bodies, including
all casts, masks, early return and source branches, are copied into
[the shared source entry](../Kernels.lean). Invocation
metadata outside the bodies distinguishes scalar/tensor offsets and all four
signed int32/int64 metadata-width combinations. Reassigned `seqlen` receives its
actual initial argument before the copied body; reads and later assignments
are part of the proved execution. `Kernels.lean` contains invocation adapters.

The candidate materializes the existing index CSE: only `rk_repeat = tl.arange(0, BLOCK_K) // 2` becomes `rk_repeat = rk // 2`.
The original Real specification, native FP scope, input domain and exterior
frames are retained. The existing proofs now execute the copied bodies,
including real temporaries for nested loads and explicit float32 casts.
Source correspondence and local proof validation are recorded separately;
independent exact-source replay is still required before completion.
