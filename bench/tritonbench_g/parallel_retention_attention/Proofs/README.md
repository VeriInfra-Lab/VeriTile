# parallel_retention_attention

Current repaired Python implementation: [parallel_retention_attention.py](../parallel_retention_attention.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` contains all seven original/candidate JIT bodies from the selected
Python repair. Parameter rebinding, int64 casts, int32 block-pointer offsets,
signed descending ranges, masks, helper calls and barriers remain in the pasted
bodies. The syntax adapters only bind arguments to these definitions.

`Correct.forward` and `Correct.backward` prove the independent causal formulas
for one launched feature shard: forward uses a 128-row tile; backward uses a
64-row tile and observes DQ, DK and DV. This is the kernel shard before the
Python wrapper's final shard sum. The public hypotheses include the wrapper's
existing `ShapeContract`/`IndexDomain` checks and the actual program coordinate.
These checks justify the retained narrowing casts; Python's input checks are
unchanged.

The backward IO executes the actual parent function. Its DK/DV helper sees
memory after DQ's stores, and the proof includes all three output channels,
checked accesses and the full cell frame. It does not substitute a hand-built
concatenation of standalone helper kernels.

`FPEquiv` compares the complete original and existing assignment-fusion
candidates. The contextual proofs preserve the typed primitive operations and
their order, and use no experimental FP assumptions.

Validation remains pending until every retained module and public theorem has
passed exact-current independent proof replay. Body correspondence and local
compilation are recorded separately from that acceptance.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
