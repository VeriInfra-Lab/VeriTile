# fused_rwkv6_kernel

Current repaired Python implementation: [fused_rwkv6_kernel.py](../fused_rwkv6_kernel.py).

`Kernels.lean` copies the complete repaired function body into `originalBody`
and the existing early-U candidate body into `candidateBody`. Both retain all
16 top-level Python statements, including the three-axis tuple assignment,
conditional pointer advances, both traversal directions and optional initial
and final state. Pointer storage, scalar ABI and constexpr metadata are declared
outside the quotations. The two Python sources are checked independently by
`jit-correspondence.json`.

The proof binds each retained load and cast primitive explicitly. The initial
state's augmented read and the separately typed vector reads have distinct
bindings in the ordered reference; no load/cast identity is assumed. The Real
entry keeps the independent array recurrence, and the FP entry compares the
complete original and early-U grids with their safety, outputs and frame.

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fused_rwkv6_kernel_scalar_abi_checked.rwkv_real: kernelIO implements per-key-shard partialValue and optional terminal recurrence over named tensors.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
