# chunk_bwd_dqkg

Current repaired Python implementation: [chunk_bwd_dqkg.py](../chunk_bwd_dqkg.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Independent dQ/dK contractions over all logical V with causal 64-token score terms and exponential gates; dG is the per-key-tile BK contraction with the terminal chunk correction. All three formulas are displayed together.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

The Lean quotation copies all 33 repaired Python statements. Invocation types and constexpr bindings stay outside the body. The existing FP candidate changes only the final integer tail bound; its [Python source](../chunk_bwd_dqkg.py) and [provenance](../../../../validation/tasks/chunk_bwd_dqkg/candidate-provenance.json) are retained. The former real-only dot-add reorder and repeated implementation blocks have been removed.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
