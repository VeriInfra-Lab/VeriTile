# softmax_flaggems

Current repaired Python implementation: [softmax_flaggems.py](../softmax_flaggems.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Forward: mass(../q[o])/sum_{n<d.N} mass(../q[row,n,col]), mass(../-∞)=0 and mass(../x)=exp(../x). Backward saved-output VJP: p[o]*(g[o]-sum_{n<d.N} p[row,n,col]*g[row,n,col]).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

`Kernels.lean` contains the four pasted original JIT bodies, their four existing
address-commuted candidates, and both complete `next_multiple_of` /
`prev_multiple_of` helper bodies. The candidate Python reference
[`softmax_flaggems.py`](../softmax_flaggems.py) is derived from the
selected repaired source by the same integer address commutations. It preserves
all floating expressions and stores. Invocation dtypes and bindings are outside
the pasted bodies; the original casts, branches, helper calls and `+=` statements
remain in them.

The execution and access proofs use the source's signed indices. Reverse
writeback uses aligned tile bounds before relating signed subtraction to the
nonnegative mathematical index. The host returns before launching empty inputs;
the standalone helper still retains its negative result when `N = 0`.
FP equivalence uses only strict integer address equalities, with no experimental
floating assumptions. Final completion is recorded only by an exact-current
independent replay receipt in the validation records.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
