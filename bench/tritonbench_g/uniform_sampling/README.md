# uniform_sampling

The complete Python source at TritonBench-G commit `603e28a`
is imported unchanged, including the compatibility helper, host generator
state update, launch heuristics, and all tests. Its SHA256 is
`d2046880647babeb532c22d3fe39e5f9662c52de36ac9a465b6405aab352a4c1`.

The proof leaves record an independent four-word specification with uint32
counter wraparound, signed offset high-word extraction, ten Philox rounds,
and the actual two-rounding fp32 uniform decoder. They also establish address
coverage and uniqueness for all four stores, local expression evaluation,
and the exact memory effect of the mathematical store sequence. The complete
23-statement real execution is connected to that independent specification.
`Correct.lean` exports both source and candidate correctness, including
tagged-cell frame preservation and successful finite flat-memory execution.

`Kernels.lean` retains the complete kernel body and changes only one integer
multiplication's operand order in the candidate. Strict source comparison
passed for all six explicit storage ASTs; the declared fp32 profile also
passed the 24 production audit gates. An offline Triton
3.8.0/sm90 probe compiled 24 explicit profiles with all four stores retained.
It observed fp32 FMA contraction in the outer affine expression. Native
hardware conformance has not been established; see `source_scope.md` and
`source_pin.json` for the precise recorded scope.

`Proofs/FP.lean` proves complete typed source/candidate execution equality for
every floating algebra and initial state, including failure. It changes only
integer multiplication. `FPEquiv.lean` exports that relation with no floating
rewrite assumptions. `TypedExecution.source_executes` proves successful
complete typed execution for arbitrary floating algebras and legal inputs.
Its nonempty witness starts with empty registers and uses B1024, N4095,
seed=offset=-1: all four streams have active stores, the final lane is masked,
and the low counter wraps. Public real and FP nonempty witnesses passed the
final audit, including 29 axiom gates and no floating rewrite assumptions.
The native numerical interpretation remains a separate obligation.
