# context_attn_nopad

Current repaired Python implementation: [context_attn_nopad.py](../context_attn_nopad.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

correct: complete-grid self-inclusive causal attention over keys 0 through query time.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Sources/Body.lean` contains the complete repaired Python body and two existing
candidate bodies: 24 top-level statements and 22 loop statements each. Invocation
storage types and scalar bindings stay outside the pasted body. All INT64 casts,
metadata reads, signed loop bounds, `tl.multiple_of`, negative-infinity causal
mask, ordered reductions, `p.to(../v.dtype)`, separate dot addition, plain output
store and terminal `return` are preserved. The old complete quotations,
source-generation macro and manually maintained prefix have been removed;
proof fragments project and connect to this actual source.

`context_attn_nopad_candidates.py` materializes the existing block-origin reuse
candidate and scale-order compatibility candidate. Reversing each documented
expression change and symbol renaming restores the complete original Python
module AST, including checked host arguments and launch configuration. See
`candidate-provenance.json` and the source-correspondence mutation tests.

The Real result retains the original causal normalized exponential formula and
its complete original launch scope, including underlaunch. The FP result
compares both actual FP32 programs, outputs, access safety and exterior memory
frames. Independent FP reference outputs use the original plain store without
an artificial final cast. Local validation and exact-current independent proof
acceptance remain separate in `validation/body-migration-progress.json`.
