# int8_matmul_quantization

Current repaired Python implementation: [int8_matmul_quantization.py](int8_matmul_quantization.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

producer_correct: signed8 toward-zero(input/rowScale), rowScale=maxAbs(row)/127. gemm_correct: sum over Fin K of signed8 A*B, then row and column scales.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
