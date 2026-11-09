# triton_argmax

Current repaired Python implementation: [triton_argmax.py](../triton_argmax.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Least maximizing logical index, independently defining maximality and first-index ties, for both global and dimension branches.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` quotes the three complete repaired Python bodies.
`CandidateKernels.lean` quotes the two existing shift candidates from
`triton_argmax_shifted.py`; that file's unchanged second-stage body shares the
original quotation. All six Python-to-Lean pairs are checked by the manifest.
Pointer storage types, signed scalar ABI and block/flag names are caller
metadata outside each body. No widening cast, NaN selection, original-address
reload, integer clamp, mask, dtype branch or first-tie rule is dropped.

The optional frontend evaluates immutable pointer dtype properties and their
static constexpr bindings. Unsupported properties, pointer mutation and
runtime-controlled constexpr bindings are rejected. Independent typed-program
equality tests cover all 15 source storage dtypes; constructor and signed-cast
checks include actual signed-address loads. The existing shift optimization
keeps its original product on unsupported block sizes. The public proofs retain
their original Real/FP scopes, caller codec/zero-clamp obligations, complete
schedules, safety and exterior frames. Source correspondence and successful
compilation alone do not certify completion; validation records track the
independent proof replay separately.
