# int8_matmul_kernel

Current repaired Python implementation: [int8_matmul_kernel.py](../int8_matmul_kernel.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

kernel_low32: signed low32 of the four packed-field dense integer sum, using the original defined FP32-to-byte cast.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` now quotes the complete repaired Python function body (23
outer statements, with both nested loops retained). The exact `add.u32`
inline-assembly instruction and all ABI arguments remain in that quotation;
its checked primitive lowering uses the separately proved strict word-add
model. No alternate Python implementation replaces the assembly barrier.
The existing comparison covers the complete grids of two retained autotune
configurations, with the original FP32-to-byte domain, output/frame and safety.
The final INT32 store is connected to the already wrapped accumulator for every
input in that domain, without requiring the dense mathematical sum to fit INT32.
Source comparison or a successful build alone does not certify completion;
current source and dependency hashes require an accepted independent replay.
