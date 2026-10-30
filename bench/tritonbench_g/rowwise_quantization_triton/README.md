# rowwise_quantization_triton

Current repaired Python implementation: [rowwise_quantization_triton.py](rowwise_quantization_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

m=max_{0≤c<columns}|x[row,c]| (empty maximum zero); byte=low8(roundEven(127*x/m)), or zero when m=0; second output is m under Real erasure of FP16 float cast.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
