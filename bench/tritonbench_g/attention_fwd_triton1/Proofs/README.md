# attention_fwd_triton1

Current repaired Python implementation: [attention_fwd_triton1.py](../attention_fwd_triton1.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Chunk-causal O(../h,row,col)=1/sqrt(../D)*sum u<min((../row/32+1)*32,T) sum e<D Q*K*V; pre-update H(../h,chunk,row,col)=sum u<min(../chunk*32,T) K*V.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Compatibility imports](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` now contains both original Python function bodies verbatim,
including all pointer constructors, pointer/value dtype casts, zero-padding
axes, `allow_tf32=False`, history stores and complete `STORE`/`IFCOND` branches.
The candidate keeps its original `BT == 32` static assertion; this is checked
from the caller's fixed constexpr before lowering. Shape products use immutable
caller metadata, while the actual `i * BT` / `i << 5` offsets stay executable.
`Sources/Kernels.lean` is only a compatibility import and contains no second
implementation. `Proofs/Program.lean` connects the complete typed program to
the canonical source by kernel-checked equality. Public Real and FP results
retain the entire grid, history observations, safety and exterior frame.
Current completion still requires the exact independent proof-replay receipt.
