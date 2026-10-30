# Finite-precision Welford and normalization contracts

The public entry covers the previously accepted stages at fp32 storage:
XBLOCK=1, one nonempty reduction block (0 < rnumel ≤ RBLOCK), the pure general
Welford helper update, and an arbitrary normalization chunk with materialized
mean/rstd. The reduction now starts from the real masked input reads and ends
at both mean/rstd stores. Its result is an independent per-operation value
recurrence and complete memory function, including alias order and the frame.
Cross-iteration reduction register scheduling remains outside this accepted
single-block scope. This is not a whole-kernel or native GPU correctness claim.

`Proofs/FPWelfordTree.lean` represents the actual helper source fixed in
`FP_WELFORD_PROFILE.json` and `Proofs/WELFORD_HELPER_SOURCE.txt`. It retains the
true first_iteration branch (value, +0, 1), the general delta/weight update,
the equality guard on the two means, and the zero-total-weight ratio guard.
A complete ordered binary `ReductionTree` is part of the numerical profile.
No tree equality, associativity, moment-sum identity or zero identity is
assumed. The tree expression and independent value recurrence are separately
proved equal by structural induction. Pure expression substitution duplicates
intermediates, not their arithmetic order or rounding points.

`Proofs/FPWelfordBlock.lean` preserves the original unit row axis, mask-only
loads, both input casts, three post-helper masks, tuple result shape, division
by rnumel, source epsilon, rsqrt, and ordered mean/rstd stores. The selected
single iteration has roffset=0. This is an explicit local expansion of the
pinned helper and tree, because the common frontend has no tuple-reduction
node. The old reduction transcription replaced Welford with three sums and
is not used as an FP source bridge.

`Proofs/FPWelfordSoftware.lean` instantiates each add/sub/mul/div, same-format
cast, equality and integer conversion with actual ScalarOps binary32 values.
The independent output mentions no unused `rest` callback. The remaining
`rsqrt32` primitive is explicit; that whole reduction entry is a backend value
contract, not a closed bit interpreter. The epsilon has fp32 bits 0x3727c5ac.
Concrete kernel-checked examples show that combining 1 and its next binary32
neighbor rounds the exact mean 1+2^-24 to 1, and that the equality guard for
two infinities preserves zero m2. A nonempty two-lane execution writes the
rounded mean and preserves an adjacent memory cell. Two lanes is an
interpreter witness, not a legal replacement for the native autotune choices
RBLOCK=1024 or 2048.

Normalization retains the original masked loads without `other`, both input
casts, subtraction, two multiplications, addition, final cast, output mask
and addresses. It has independent output values and a complete memory frame.
Its software instance depends only on concrete scalar FP32 arithmetic; a
nonempty example stores 1 after adding 2^-25, while a masked tail stays intact.

The eight public declarations are five backend value contracts
(normalize_source_fp32, welford_helper_fp32, welford_reduction_fp32,
welford_software_with_rsqrt, welford_nonempty_rounding) and three bit-interpreter
contracts (normalize_software_fp32, normalize_nonempty_rounding,
welford_bit_examples). The names do not assert native conformance.

Native binding still needs the exact runtime helper version, generated
Welford reduction tree and lane order, scalar rounding/FTZ and FMA contraction
settings, libdevice.rsqrt behavior, literal conversion, and storage/address
and launch mapping. The installed helper SHA is provenance for the selected
profile, not evidence that another benchmark environment uses that version.
These are explicit primitive/configuration correspondences; no hypothesis
assumes the final mean, variance, rstd or output is already correct.
