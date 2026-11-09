# parallel_attention

Current repaired Python implementation: [parallel_attention.py](../parallel_attention.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The seven complete original and existing fused-candidate Python JIT bodies are
copied in `Kernels.lean`; `jit-correspondence.json` checks their exact statements.
Runtime argument types are supplied outside the body. The backward quotation
retains PID decoding, both effectful helper calls, the barrier, and the signed
descending range. Typed FP32 loads occur before the original explicit casts.

`Correct.forward_grid_real` and `Correct.backward_grid_real` prove the independent
full-grid Real formulas, execution, safety and frame. The backward public IO runs
the complete parent quotation. `Proofs/CallerCombined.lean` connects it to the
ordered helper recurrences; launch registers are derived from program IDs.

`FPEquiv.fused_score_equiv` retains the original phase-specific native schedule
and its existing assignment-fusion candidate. `Proofs/FusedKernel.lean` also
proves execution and safety equivalence of the complete original and fused
backward callers, for arbitrary structural FP algebra and dot primitive.
No new numerical assumption or input restriction is introduced.

Source comparison and local builds do not certify completion. The target stays
pending until every retained proof and actual public theorem is accepted by
independent replay against the exact current source/import closure.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
