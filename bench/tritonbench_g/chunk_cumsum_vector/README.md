# chunk_cumsum_vector

Current repaired Python implementation: [chunk_cumsum_vector.py](chunk_cumsum_vector.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For the same feature column, inclusive sum over logical time k≤t and k<T of the initial input, with time/feature coordinates made explicit.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
