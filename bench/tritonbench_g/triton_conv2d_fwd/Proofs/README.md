# triton_conv2d_fwd

Current repaired Python implementation: [triton_conv2d_fwd.py](../triton_conv2d_fwd.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` pastes both existing Python bodies: the original has 19 top-level
statements and the shift candidate has 20, including its `static_assert`.
Docstrings, all three loops, padding arithmetic, masks, optional FP16 casts,
TF32 flag, dot/add order and implicit store conversion are retained. Invocation
types and constexpr bindings stay outside the quotations. The former separate
FP16/default quotations now call the same body. `Model.lean` retains the
mathematical coordinates, primitive interfaces and reference interpretation.

The compared caller fixes the row tile to 128. Compile-time assertions are
checked using immutable integer/Boolean caller metadata; a rejected assertion
has no successful execution and cannot satisfy the safety judgment. Regression
proofs cover arbitrary complete bodies, successful execution, rejection, safety,
signed constexpr subtraction and invalid runtime/floating/rebound predicates.
`Proofs/UnitRanges.lean` connects the original single-argument loops to the
existing unit-stride invariants, for execution and safety, without editing the
source or assuming any floating identity.

Independent full grouped convolution sum over all channel/filter-height/filter-width coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

Completion requires current independent proof replay and axiom/specification
acceptance in addition to body correspondence and local rebuilding.
