# lora_expand_gemv

Current repaired Python implementation: [lora_expand_gemv.py](../lora_expand_gemv.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` pastes both original function bodies from the repaired Python
entry and the existing `lora_expand_gemv_unit_split.py` candidate. Each has
13 top-level statements, including the docstring and all nested operations.
Pointer/scalar types and constexpr bindings are caller metadata, outside the
quotations. The retained dtype profiles call these bodies; the old source
macro and duplicate selected quotation are retired. `SourceTypes.lean` holds
only invocation types and format helpers. The duplicate repaired Python
snapshot is also retired; its JIT body was identical to the canonical entry.

The source's implicit store conversion replaces the former handwritten
extra cast. The candidate specializes only `cdiv(N, SPLIT_N)` to one, under
the checked caller's `SPLIT_N=N` and positive `N` conditions. The public
comparison retains FP32 input, FP16 weights, FP32 output, every CTA, safety,
output cells and exterior frames, with no numerical rewrite assumption.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

Body correspondence and compilation are intermediate checks. Completion
requires current independent replay and axiom/specification acceptance.
