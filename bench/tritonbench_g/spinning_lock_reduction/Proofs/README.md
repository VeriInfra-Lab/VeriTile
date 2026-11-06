# spinning_lock_reduction

Current repaired Python implementation: [spinning_lock_reduction.py](../spinning_lock_reduction.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` pastes both complete Python function bodies, including every
signed/unsigned cast, negative-view branch, CAS poll, barrier, and all nine
rounds. `jit-correspondence.json` checks both bodies without renaming variables
or deleting casts; `candidate-provenance.json` records the two documented area
reassociations and checks the complete Python host module.

The caller signature and three `False` view flags are outside the pasted body.
The repaired wrapper rejects negative/conjugate views before querying these
flags. `SourceAdapter.decode` checks that ABI and specializes only those bound
constexpr branches; changing a pointer dtype or a flag fails closed. The
frontend itself retains all branches. `Proofs/CallProfile.lean` and the shared
parser regressions check these boundaries.

Named partials/output/locks stage formula adjacent to the specification.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
