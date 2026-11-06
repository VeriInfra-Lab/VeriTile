# streamk_matmul

Current repaired Python implementation: [streamk_matmul.py](../streamk_matmul.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` pastes all six repaired JIT bodies and all six candidate bodies.
`jit-correspondence.json` checks the complete ASTs without renaming variables,
removing casts or replacing `ACC_TYPE`. The FP32 storage and accumulator dtype
are checked caller metadata from the repaired wrapper, outside each body.

`SourceAdapter.decodeSources` checks that every helper name links to the actual
supplied swizzle/linear/mac source member. Every helper argument and every
source statement is decoded before the execution plan is formed. Missing or
changed dtypes, pointer types and unlinked helpers fail closed; regressions are
in `Proofs/CallProfile.lean` and `tests/lean/PastedControlFunctions.lean`.

`candidate-provenance.json` records only two scratch tile-area reassociations
and the corresponding entrypoint/class renaming across the complete Python
module. The public Real specification is the independent finite K-term matrix
product; the FP statement retains the original fixed MMA, atomic ordering,
FP16 full-tile round trip, input domains, safety and frames.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
