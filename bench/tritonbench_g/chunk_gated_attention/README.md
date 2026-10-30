# chunk_gated_attention

Current repaired Python implementation: [chunk_gated_attention.py](chunk_gated_attention.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics.innerWeight/innerContribution/innerState expose independent chunk gate products and logical history/final contraction, including gateK choice. prefixResult exposes the finite within-chunk cumulative sum.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
