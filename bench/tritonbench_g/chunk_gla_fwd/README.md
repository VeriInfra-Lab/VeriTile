# chunk_gla_fwd

Current repaired Python implementation: [chunk_gla_fwd.py](chunk_gla_fwd.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics.attention explicitly sums logical K causal gated QK terms and scales them. consumerValue explicitly sums logical K gated history terms plus causal attention/value terms. producerResult and consumerResult show the logical output coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
