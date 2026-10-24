# int8_matmul_kernel

The current standard `Kernels.lean`, `Correct.lean`, and `FPEquiv.lean` entries
import the [checked FP32 repair](../../repairs/tritonbench_g/int8_matmul_kernel/README.md).
The unchanged original Python remains here. The repair checks its supported
input contract before forwarding the original function and kernel.

`Whole001/` contains the complete accepted original-body source proof: all
15 configurations, four field-major passes, full grouped grid, completed
legal autotune trials, selected int32 return, source-order integer/pointer
bounds, and caller-memory frame. The checked entries reuse that result.
The original logical-negative-view counterexample and earlier narrower
int32-input proofs remain in `Proofs/` and compatibility entries; they are
not substituted for the complete selected FP32 result.

The two unresolved numerical families are the defined FP32-to-signed8
conversion and fixed-32-lane live-seed integer instruction behavior. The
finite target retains each selected instruction's signed32 overflow rule.
No native equality to an unbounded integer dot product is asserted.
