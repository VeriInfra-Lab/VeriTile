# kldiv_compute

Current repaired Python implementation: [kldiv_compute.py](../kldiv_compute.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

correctness: output[i] = x[i] * Real.log(../x[i]/y[i]).

All 216 retained dtype combinations instantiate the two complete raw quotations in `Kernels.lean`. The profile names in `Sources/Mathematical_Kernels.lean` are thin adapters; the old source generator is retired. The existing native support classifier is unchanged.

[Transition evidence](../../../../validation/kldiv-source-transition.json) identifies the fixed Git commit containing the complete-program equality proof for all 384 previously supported original/candidate configurations.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
