# rms_rbe_matmul

Adjacent normalizedProjection and projectedRotary mathematics, indexed by logical batch/row/column; optional rotary and the source storage decoder stay explicit.

- [rms_rbe_matmul_launch_guard.py](../rms_rbe_matmul_launch_guard/rms_rbe_matmul_launch_guard.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](../rms_rbe_matmul_launch_guard/Contract.lean)
- [Correct.lean](../rms_rbe_matmul_launch_guard/Correct.lean)
- [FPEquiv.lean](../rms_rbe_matmul_launch_guard/FPEquiv.lean)

## Scope

Whole original rectangular grid KernelGrid.programs B p, every active output, including masked K tails and empty launch cases.

## Conditions

Explicit conditions:

- d8:Float8E5Bits→ℝ with d8(ofInt 0)=0
- fp8:Bool, B:Nat, p:KernelSource.Parameters
- p.N%2=0
- realStorageDomain: signed-byte storage for FP8 branch, arbitrary FP16 values otherwise
- ProgramIO.ImplementsOn retains coherent IO, bounds, safety, successful grid execution, all outputs and exterior cell frame.

Contract correspondence:

Selected launch-guard Contract retains original input contract and bounds only actual nonempty batch and flattened M/N tile axes. Kernel theorem remains over the original grid, not a newly narrowed input domain.

Exact FP program pairs and numerical dependencies are recorded in the [rms_rbe_matmul contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
