# chunk_gate_recurrence

Current repaired Python implementation: [chunk_gate_recurrence.py](chunk_gate_recurrence.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Independent exclusive forward and reverse suffix/product formulas; logical forward state or backward state, DG = sum over key:Fin64 and value:Fin16 of reverse*saved, and initial gradient. All 1024 contraction terms are retained.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
