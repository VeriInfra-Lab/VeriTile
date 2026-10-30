# Finite-precision stage correctness

`FPCorrect.lean` exposes execution-to-value contracts for the original
accepted forward and backward stage slices. The generic contracts preserve
every input conversion, ordered scalar operation, reduction and output
conversion. Output lanes have exact values and untouched cells keep their
dtype tags. The state-update and scalar dbeta statements specify the entire
resulting memory function directly.

The covered forward stages are v_new writeback, state update with headwise
and scalar beta, and output reduction. The backward stages are dk, dv,
dbeta and dq with both beta modes, plus the second-loop dk correction.
The precomputed HPrev/HNext/DHPrev carry boundary remains explicit. The
contracts do not assume that the supplied carry is a final result.

`Proofs/FPShape.lean` and `Proofs/FPBackwardShape.lean` prove that the typed
stage bodies agree with the original accepted slices after dtype erasure.
`Proofs/FPStateShape.lean` establishes the same correspondence for both
state branches after selecting the syntactically constant beta flag; it
also proves that this selection preserves `Structural.run`. These bridges
preserve arithmetic structure and do not establish hardware conformance.

`Proofs/FPSoftware.lean` defines the concrete reduction profile: ascending
axis index, a left fold beginning at positive zero, and a rounded fp32
`FP.add` at every step. Scalar add/sub/mul operations use `ScalarOps` on
`FP.Value .fp32`. Both reduction axes and the nested scalar dbeta reduction
use this profile. Output formulas in all software contracts are independent
of unsupported callbacks in `rest`. The software output/dk/dv/dbeta/dq
instances select scale=1; their generic backend contracts retain arbitrary
scale. V_new, state and dk correction have no scale restriction.

The nonempty output witness has two key terms, one valid output lane and
one masked tail. Under `Config.ieeeValue` (nearest-even, no input/result
flushing), the terms `1` and `2^-25` reduce to the binary32 value `1` while
the exact rational sum differs. Lean checks the bit calculation and proves
successful execution, output readback and tail preservation.

A native binding must establish the fp32 storage/address representation,
the configured scalar operation and cast behavior (including special
values, signed zeros and subnormals), absence of unmodeled FMA contraction,
and the exact reduction sequence/tree. The sequential software profile is
not asserted to match Triton's generated GPU reduction tree. No native GPU
experiment or compiler proof is included here. Complete forward/reverse
loop composition, carry initialization/finalization, host allocation and
autograd, and other storage dtype profiles remain outside these stage
contracts. No identity-rounding, zero-annihilation or reassociation premise
is used.
