# chunk_retention

Current repaired Python implementation: [chunk_retention.py](chunk_retention.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics exposes headRate, weighted forward and reverse suffix sums, state/final state, forcing, local coefficients, chunk output/dQ/dK and key-block partial dV. result explicitly routes all seven possible output channels.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
