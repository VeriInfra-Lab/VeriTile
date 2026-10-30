# kv_cache_copy

Current repaired Python implementation: [kv_cache_copy.py](kv_cache_copy.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

LogicalIO.io ⊨{length/block OutputUnique} fun a => ⟨a.key,a.value⟩

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
