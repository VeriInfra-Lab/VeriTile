# sgmv_expand_slice

Current repaired Python implementation: [sgmv_expand_slice.py](../sgmv_expand_slice.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` pastes the repaired Python function and its existing tile-index
reuse candidate from `sgmv_expand_slice.py`. Both bodies retain all 38 top-level
statements, explicit int64 conversions, early returns, masks, constexpr branches,
the signed single-argument range, dot accumulation and output conversion.
Caller metadata stays outside the quotations. The 48 weight/output/metadata
source interfaces and eight candidate interfaces are thin calls to these two
quotations. `SourceTypes.lean` contains only invocation types;
`Sources/Kernels.lean` forwards the canonical implementation.

`Proofs/MetadataStorage.lean` proves equality of the complete kernels across all
eight native metadata storage combinations, after the source's explicit int64
widening. The retained public scope is FP32 input, FP16 weights and FP32 output,
with ordered segments and possible output overlap. Index reuse is structural:
the FP proof introduces no numerical rewrite or additional assumption.

Ordered segment fold: each active output is sum_{k<inner} X[row,k]*W[adapter,column-slice,k], optionally plus previous output; inactive/untouched cells retain previous value.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

Body correspondence and local rebuilding are intermediate checks. Current
independent proof replay and axiom/specification acceptance remain required
before this migration is marked complete.
