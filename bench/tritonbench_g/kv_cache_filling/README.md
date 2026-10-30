# kv_cache_filling

Current repaired Python implementation: [kv_cache_filling.py](kv_cache_filling.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Full-head finite minimum/maximum over dk/dv, scale 1 on constant rows and otherwise (hi-lo)/levels; byte floor((x-lo)/scale+1/2); INT8 levels=255, INT4 levels=15 with low+16*high wrapping packing; copy and both metadata lanes.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
