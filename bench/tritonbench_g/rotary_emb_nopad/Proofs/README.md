# rotary_emb_nopad

Current repaired Python implementation: [rotary_emb_nopad.py](../rotary_emb_nopad.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

(a*c-b*s, a*s+b*c), explicitly for Q-first/Q-second/K-first/K-second; cached K values are written into the metadata-selected cache.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The four bodies in `Kernels.lean` are copied directly from the selected repaired
Python and [rotary_emb_nopad.py](../rotary_emb_nopad.py). The latter is
an explicitly new Python realization of the existing Lean Q-address reuse
candidate, derived from the repaired source by changing only `off_q1` to
`off_q0 + (HEAD_DIM // 2) * head_dim_stride`. It is used to check the existing
FP equivalence claim. Its provenance is not an upstream implementation.
`jit-correspondence.json` checks all four bodies, including the cached early
return, signed64 widenings, metadata accesses and cache-before-Q store order.
All pointer dtypes and runtime/constexpr bindings are outside the quotations.
