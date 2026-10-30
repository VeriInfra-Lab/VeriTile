# chunk_gate_recurrence

Independent exclusive forward and reverse suffix/product formulas; logical forward state or backward state, DG = sum over key:Fin64 and value:Fin16 of reverse*saved, and initial gradient. All 1024 contraction terms are retained.

- [chunk_gate_recurrence_repaired.py](chunk_gate_recurrence_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Original legal 64x16 per-program scope; both full N loops, seed modes and four repaired seed strides retained. This is not presented as a new whole-grid Tensor theorem.

## Conditions

Kind, Shape, SeedStrides, seed flag, legal Program c and 0<c.N remain visible. IO.program retains coordinate/memory/mask conditions; Contract preserves N=0 host no-launch handling and ordinary descriptor/allocation checks.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_gate_recurrence contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
