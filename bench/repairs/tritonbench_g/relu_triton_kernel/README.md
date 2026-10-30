# relu_triton_kernel

max 0 (xs i) for every active lane of the arbitrary-PID tile.

- [relu_triton_kernel_repaired.py](relu_triton_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete program tile at arbitrary PID; all active lanes pid*1024+i<N. It is not a full-grid schedule theorem.

## Conditions

Explicit conditions:

- N arbitrary, including empty/tail tiles
- The MaskedKernelIO₁ judgment supplies coherent logical input binding, input/output bounds, safe masked accesses, successful source execution, active output values and exterior memory frame.

Contract correspondence:

Selected repaired source removes the original benchmark pid-zero store gate. Correct explicitly concerns that repaired source; it does not assert correctness of the defective benchmark gate.

Exact FP program pairs and numerical dependencies are recorded in the [relu_triton_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
