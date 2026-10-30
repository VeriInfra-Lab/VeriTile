# chunk_gla_simple

Current repaired Python implementation: [chunk_gla_simple.py](chunk_gla_simple.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics.qhSum/qkSum display block/lane contractions; score displays the causal gated QK weight; expected displays gated history plus causal weighted V; result gives logical full-output coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
