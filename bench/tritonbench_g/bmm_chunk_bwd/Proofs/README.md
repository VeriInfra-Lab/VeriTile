# bmm_chunk_bwd

Current repaired Python implementation: [bmm_chunk_bwd.py](../bmm_chunk_bwd.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

For each actual output lane, the finite Dout-transpose times A sum over the true shortened chunk length, plus the optional initial residual.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

`Kernels.lean` contains the original 45-statement repaired function body and
its existing integer head-index candidate. The 21 argument casts, signed IDs,
loop bound, masks, dot casts, residual order and output cast remain in their
original source order. Caller storage, scalar values, constexprs and geometry
are supplied outside the pasted body. The candidate changes only
`pid_c * ngroups` to `ngroups * pid_c` in `pid_h`.

[Parameters.lean](Support/Parameters.lean) holds host/call metadata. The compatibility
imports under [Sources/](Sources) forward to this same source; they do not
generate an alternative residual-order implementation. The independent
matrix-product mathematics remains in `Sources/Mathematical_RealMath.lean`.

[source correspondence inventory](../../../../validation/source-correspondence.json) checks both copied bodies
against their complete Python ASTs and the reviewed invocation profile.

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
