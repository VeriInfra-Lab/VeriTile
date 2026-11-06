# layer_norm_fwd

Current repaired Python implementation: [layer_norm_fwd.py](../layer_norm_fwd.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Same-reader independent row scaling, canonical Philox dropout, ordered residual/RMS/LN and two affine formulas over named tensor inputs.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
