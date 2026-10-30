# chunk_delta_fwd

Current repaired Python implementation: [chunk_delta_fwd.py](chunk_delta_fwd.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Correct.Mathematics.corrected = value minus the finite logical-K delta/state contraction; Mathematics.state is the independent seed/rounded-window update recurrence. formula routes history, corrected values and optional final state.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
