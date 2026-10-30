# isfinite_kernel

isfinite_real_correctness: every observed Real output of this masked tile is true.

- [isfinite_kernel_repaired.py](isfinite_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete masked tile invocation (n,B,pid), for both original FP32 and FP64 source dispatches; signed byte-address views and stride/output window are retained. No whole-host-grid claim.

## Conditions

- Original output-address injectivity for logical indices below n remains explicit.
- ProgramIO retains active pid*B+lane<n masks, typed bindings/bounds, safety, successful source execution and exterior frame.
- The independent Real predicate does not assert arbitrary IEEE words are finite; all-word exponent classification remains optional.

Exact FP program pairs and numerical dependencies are recorded in the [isfinite_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
