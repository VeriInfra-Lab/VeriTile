# chunk_gla_simple

Current repaired Python implementation: [chunk_gla_simple.py](../chunk_gla_simple.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The original quotation copies the complete repaired Python function body. The
[row reuse candidate](../chunk_gla_simple.py) materializes the retained
Lean candidate in Python: only `rows` reuses `o_i.to(../tl.int64)`. Its provenance
records the original source SHA256. Both complete bodies, call bindings and
pointer/scalar precision are checked by `jit-correspondence.json`.

All explicit INT64 casts, three tuple stride assignments, signed loop bounds,
dot options, masks, source casts and writeback remain visible. `Sources/Program`
proves its entire Real program equal to the precision erasure of the actual
quotation; address and loop proofs retain signed registers. FP execution follows
the source's ordered recurrence under the three available opaque dot operations.
The public equivalence proves the row-vector reuse for arbitrary floating
operations and introduces no numerical rewrite assumption.

Mathematics.qhSum/qkSum display block/lane contractions; score displays the causal gated QK weight; expected displays gated history plus causal weighted V; result gives logical full-output coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
