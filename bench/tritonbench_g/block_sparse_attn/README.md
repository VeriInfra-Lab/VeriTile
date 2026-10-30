# block_sparse_attn

Current repaired Python implementation: [block_sparse_attn.py](block_sparse_attn.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Causal CSR occurrence-multiset attention: full head-feature Q/K dot in each exponential weight; sum weight*V divided by sum weight over all CSR visits and block lanes. Repeated keys remain repeated occurrences, causal keys above the query get zero weight.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
