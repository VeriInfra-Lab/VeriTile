# batched_vecmat_mult

Current repaired Python implementation: [batched_vecmat_mult.py](batched_vecmat_mult.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For every (m,n), sum over all k : Fin K of realValue(decode,A[m,k])*realValue(decode,B[m,n,k]); arbitrary logical source expressions retain their original decoding.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
