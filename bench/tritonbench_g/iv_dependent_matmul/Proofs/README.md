# iv_dependent_matmul

Current repaired Python implementation: [iv_dependent_matmul.py](../iv_dependent_matmul.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

kernel_real: C[row,col] = sum over Fin g.inner of A[row,k]*B[k,col].

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The two bodies in `Kernels.lean` are literal copies of the complete repaired
`iv_dependent_matmul_kernel_i64` and its retained live-accumulator candidate.
`jit-correspondence.json` checks all 29 top-level statements in each body,
including all INT64 conversions, five pointer schedules, staging assignments,
load masks, FP16 narrowing and masked store. Dtypes and runtime call arguments
are declared outside the body. The ordinary output allocation is FP32, so the
FP16 register is widened at its store boundary.

`iv_dependent_matmul_dotacc.py` materializes the existing native graph; its
provenance records the original module hash. It changes only `accumulator +=
tl.dot(a, b)` into `accumulator = tl.dot(a, b, accumulator)`. Real correctness
uses the original source. The retained FP theorem compares two distinct pointer
schedules of the live-accumulator graph under its documented native profile; it
does not assert an unconditional numerical identity between Dot+Add and DotAcc.
The ten former handwritten Lean bodies have been replaced by thin callers of
these two copied bodies. Proof decompositions select statements from the actual
source and are not separate implementations.
