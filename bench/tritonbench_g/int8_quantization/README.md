# int8_quantization

Current repaired Python implementation: [int8_quantization.py](int8_quantization.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

per_block_int8_real: nearest-away 127*factor*x/blockMaximum, with scale=blockMaximum/127 and the original zero fallback for invalid blocks.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
