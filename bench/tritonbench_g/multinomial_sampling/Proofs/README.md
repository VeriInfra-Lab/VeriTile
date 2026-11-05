# multinomial_sampling

Current repaired Python implementation: [multinomial_sampling.py](../multinomial_sampling.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

multinomial_categorical_correctness

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The complete original and static-unroll JIT bodies from `multinomial_sampling.py`
are quoted literally in `Kernels.lean`; `jit-correspondence.json` checks both.
Call signatures supply pointer storage, strides, counts and checked arange
sizes outside the quotations. The original runtime token count and candidate
constexpr token count remain distinct. The candidate retains its explicit
`b_idx = b_idx * BLOCK_N` statement and every scan, comparison and label read.

The ordered reference retains actual FP32 loads before explicit score casts,
including masked zero padding. `Primitives.load32` selects that one primitive
without imposing a numerical law; the Real instance uses identity. Other
storage profiles retain their own padding conversion and storage dtype. The
proof constructs every actual loop step, final integer output and complete
memory frame, then connects to the independent inverse-CDF formula. The
static-unroll proof compares both complete executions and access safety. Source
comparison and successful compilation do not certify completion; independent
proof replay is recorded separately in `validation/body-migration-progress.json`.
