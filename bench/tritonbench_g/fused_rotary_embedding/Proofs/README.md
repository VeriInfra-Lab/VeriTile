# fused_rotary_embedding

Current repaired Python implementation: [fused_rotary_embedding.py](../fused_rotary_embedding.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fused_rotary_embedding_tile_checked.fused_rotary_real: rotaryIO implements Q/K rotation and V copy on logicalMetadataDomain.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` copies the complete repaired Python body verbatim. The external
call signature records the FP32 data pointers, int32 metadata arrays and widened
integer scalar profile; all bindings remain outside the quotation. The original
body retains the signed64 PID/lane/metadata conversions, grouped K/V branch,
separate K-cache split layout, full-width V copy and original store order.

[fused_rotary_embedding.py](../fused_rotary_embedding.py) is an
explicitly new Python realization of the existing Lean Q-address reuse
candidate: only `off_q1` changes to `off_q0 + (HEAD_DIM // 2) * q_dim_stride`.
It is derived from the repaired source, not an upstream version.
`jit-correspondence.json` checks both full bodies.
