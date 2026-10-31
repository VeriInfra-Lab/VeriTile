# fast_rms_layernorm

Current repaired Python implementation: [fast_rms_layernorm.py](fast_rms_layernorm.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Y and saved reciprocal RMS, DX with RMS correction, and complete-row DW sum in regular/Gemma modes.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
