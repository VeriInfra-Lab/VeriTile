# context_attn_mistral

Current repaired Python implementation: [context_attn_mistral.py](../context_attn_mistral.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

kernel_real: complete original ordinary grid implements finiteSentinelAttention on the original metadata domain.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Sources/Body.lean` holds the complete original repaired JIT body and the two
existing candidate bodies. Storage and runtime scalar bindings are supplied by
the call signature outside the pasted quotation. The bodies keep every INT64
cast, the exact subtract-then-compare sliding-window condition, the finite
sentinel, both numerical guard branches, the signed loop, `p.to(../v.dtype)`, the
plain final store and the terminal `return`. The old complete quotations and
manually maintained prefix have been removed; mathematical proof fragments
project and connect to this actual source.

`context_attn_mistral_candidates.py` materializes the existing index-reuse
candidate selected by the public FP theorem and the scale-order candidate used
by retained Real compatibility results. Reversing each documented change and
symbol renaming restores the complete original module AST, including the
checked host arguments and launch configuration. See `candidate-provenance.json`.

The Real specification retains finite-sentinel attention over every visited
padded lane, including the all-sentinel and zero-normalizer guards. The public
FP theorem compares the complete actual FP32 programs, full grid outputs,
access safety and exterior frames. The FP reference uses the source's plain
final store, without an additional artificial cast; the actual probability
cast inside the loop remains explicit. Local validation and independent proof
acceptance are tracked separately in `validation/body-migration-progress.json`.
