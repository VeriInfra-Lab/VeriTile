# token_attn_llama2

Current repaired Python implementation: [token_attn_llama2.py](../token_attn_llama2.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fun xs b h t => (sum d, xs.query b h d * xs.keyAt (m.page b t) (h/(../heads/kvHeads)) d) / sqrt D

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Pure parameters and compatibility imports](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

Both existing 15-statement Python bodies are copied into `Kernels.lean`, including the dynamic 0/1 loop, query/key loads, masks, reduction, scale multiply and store. The selected caller profile is FP32 Q/K/Out, signed-i32 metadata and FP32 runtime scale; typing and scalar binding are outside the quotations. There are no invented Q/K/output casts. The optional integer-where adapter assigns integral type to two representable weak integer literals without changing mixed or floating branches; its typed-program and range checks are in `tests/lean/IntegerWhereLiterals.lean`.

Source execution retains signed lengths, starts, page indices and complete signed addresses. The logical IO stores every original metadata word, including inactive page cells. The existing `Valid` domain supplies the nonnegative active metadata and address conditions needed to recover natural mathematical indices. Compilation and body comparison alone do not establish completion; current independent proof replay is required by the validation progress record.
