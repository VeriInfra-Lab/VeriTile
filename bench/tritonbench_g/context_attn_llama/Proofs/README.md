# context_attn_llama

Current repaired Python implementation: [context_attn_llama.py](../context_attn_llama.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Mathematics.score/numerator/denominator/expected display paged finite-sentinel base-2 normalized attention over all visited padded key lanes; mathematical maps original grid/lane coordinates and gathered logical inputs.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Sources/Body.lean` contains the complete pasted Python JIT bodies. The call
signature supplies storage types and runtime scalar bindings separately; the
bodies preserve all INT64 casts, `tl.multiple_of`, signed loop bounds,
load/store masks, the finite causal sentinel, accumulator dot, and
`p.to(../v.dtype)`. No `tl.region` declarations are inserted into the pasted body.
`Kernels.lean` and `Sources/Mathematical_Kernels.lean` instantiate these bodies;
the proof fragments are projections with explicit execution connections.

`context_attn_llama_candidates.py` materializes the two existing Lean
candidates. `scale_first` commutes the scale multiplication and is retained for
the existing Real compatibility results. The public FP theorem selects
`block_end_factored`, which factors the common integer prompt offset out of
`tl.minimum`, retaining the entire floating computation. The provenance record
and Python tests reverse each documented change and check the complete module
AST against the unchanged repaired original, including host input checks and
launch configuration.

The Real specification is finite-sentinel, base-2 attention over every visited
padded key lane, under the existing complete grid/geometry/input conditions.
The structural FP result compares both actual FP32 programs, including failure,
output and exterior-frame behavior, without an experimental floating rule.
Local migration checks and independent proof acceptance are recorded separately
in `validation/body-migration-progress.json`; this source description does not
assert independent completion.
