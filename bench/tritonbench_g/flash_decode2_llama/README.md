# flash_decode2_llama

Current repaired Python implementation: [flash_decode2_llama.py](flash_decode2_llama.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Exponentially weighted split-summary average over ceil(length/Q) summaries at every output feature.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
