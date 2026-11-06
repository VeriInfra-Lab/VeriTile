# context_attn_fwd

Current repaired Python implementation: [context_attn_fwd.py](../context_attn_fwd.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Mathematics.score displays logical-D QK and finite -100000000 sentinel; mathematical displays the full visited-window base-2 exponential weighted numerator divided by the full padded denominator, with decoded original grid/lane coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Sources/Body.lean` contains the complete repaired `_fwd_kernel_int8kv` body and
the two existing candidate bodies. Each has 24 top-level statements and 18 loop
statements. Call-site storage types and runtime scalar bindings are outside the
pasted body. All INT64 casts, metadata order, grouped KV indices, signed key
loop, masks, finite sentinel, reductions, `p.to(../v.dtype)`, accumulator dot and
plain final store remain in source order. The checked Python host uses FP32
payloads despite the historical kernel name.

The old source-generation macro, complete quotations and manual prefix have
been removed. Typed adapters instantiate the copied bodies; proof fragments
project and connect to the actual source. `context_attn_fwd_candidates.py`
materializes the existing scale-order candidate and integer block-end factoring
candidate. Reversing each documented expression change and name restores the
entire repaired original Python module AST, including host checks and launch
configuration; see `candidate-provenance.json`.

The Real result retains finite-sentinel base-2 attention over all visited
padded key lanes. The public FP result compares both complete actual FP32
programs with all outputs, source access safety and exterior frames. Independent
FP reference outputs use the original plain store, without an artificial final
cast. Local source/proof validation and exact-current independent acceptance
are recorded separately in `validation/body-migration-progress.json`.
