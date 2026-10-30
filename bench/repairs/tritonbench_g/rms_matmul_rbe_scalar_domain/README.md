# rms_matmul_rbe

Independent adjacent normalizedProjection and projectedRotary definitions; displayed logical tensor coordinates and phase/launch row and column mapping.

- [rms_matmul_rbe_repaired.py](rms_matmul_rbe_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

For every finite launch order, each launch runs Q→K→V. Typed result channels observe scheduled final winning write events, preserving output aliases; arbitrary order is not advertised as a rectangular logical-grid coverage theorem.

## Conditions

Explicit conditions:

- fp8:Bool, p:Fin 3→Arguments
- epsilon/theta:HostScalar with exact value binding
- Each phase N is even
- Every reader X/W/RMS region differs from every writer OUT region
- order:List(Nat→Nat) arbitrary, including empty
- IO.InputDomain fp8 p: signed-byte W/RMS storage only in FP8 branch
- ArrayProgramIO.ImplementsOn retains coherent reads, array capacity/bounds, safety, success, every final output and frame.

Contract correspondence:

Selected scalar-domain Contract retains inherited tensor/scalar checks and adds exact Python dispatch predicates: EPS Python-integer overflow bound, THETA Python-float with normal finite FP32 range. The kernel theorem binds HostScalar values and preserves all original precision/strides; no extra host/Tensor proof is imposed.

Exact FP program pairs and numerical dependencies are recorded in the [rms_matmul_rbe contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
