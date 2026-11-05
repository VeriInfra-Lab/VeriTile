# triton_softmax

Current repaired Python implementation: [triton_softmax.py](../triton_softmax.py).

The complete original and [index-commuted](../triton_softmax.py)
function bodies are copied into `Kernels.lean`. The call interface supplies
allocation-relative pointer offsets and the signed int64 runtime arguments;
both original int64 casts remain inside the quotations.

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Unshifted exp(../row[i]) / sum over exactly Fin n active logical columns.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
