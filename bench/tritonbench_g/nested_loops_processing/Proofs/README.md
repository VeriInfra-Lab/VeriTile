# nested_loops_processing

Current repaired Python implementation: [nested_loops_processing.py](../nested_loops_processing.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` quotes the complete `nested3_repaired` function body and the existing index-commutation candidate. The Python counterpart of that candidate is explicitly marked as newly authored; the repaired original function body is unchanged. Call signatures and the retained contiguous caller's unit minor stride are supplied outside the quotations.

The `Proofs/Typed*` execution lemmas derive all three nested loops, signed coordinate registers, cached loads, masks, stores and barriers from those actual quotations. `Correct.lean` connects the resulting whole memory state to the independent ordered-copy formula; `FPEquiv.lean` compares both complete programs. A successful local build is recorded separately from independent proof replay.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
