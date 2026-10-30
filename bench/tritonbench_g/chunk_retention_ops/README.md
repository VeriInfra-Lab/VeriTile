# chunk_retention_ops

Current repaired Python implementation: [chunk_retention_ops.py](chunk_retention_ops.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics exposes weighted/suffix retention, final/initial gradient, output/query/key/local coefficient/partial-value sums, and mathOutputs/result explicitly route all eight channels.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
