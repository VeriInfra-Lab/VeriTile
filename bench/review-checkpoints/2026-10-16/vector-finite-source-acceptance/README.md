# Vector addition finite source acceptance

Six additive modules connect the original eight-statement kernel and its wrapper
to existing typed execution, physical storage views, finite indices and pointers,
load-before-store memory effects, frames and complete schedule invariance.
The canonical 17 by 17 dtype inventory retains all 189 admitted profiles and 100
actual frontend rejections. Source n_elements is i32, with 0 <= N < 2^31 and
BLOCK_SIZE=1024. Physical y backing capacity suffices even when y.numel < N.

The source returns flat physical addition under actual preserve-format output
metadata. Arbitrary strided inputs are not falsely described as logical Torch
elementwise addition. Undefined active float-to-integer conversions remain an
explicit structural defined-domain restriction, never a numerical deferral or
an invented native trap/rollback result. Allocation and CUDA failures retain
their source order. Input views may alias; fresh output is disjoint.

The public contract binds 2,601 cases and 616 exact numerical ports in 43 new
families. Numerical conformance remains deferred. Source acceptance requires
the active integration build and schema2/exact-authority checks below; it does
not establish compiler implementation correctness or native GPU correctness.

## Observed integration results

The eight-target build passed (7,389 jobs). All 244 benchmark Python tests
passed: 34 exact numerical tests and the remaining 210 tests, run separately
without repeating the exact suite. All 25 sharding tests passed. The unchanged
schema2 checker accepts vector addition with exact numerical deferrals.
Source scope is **3 accepted / 183**: matrix transpose verified, ReLU and vector
addition verified with deferred numerics, and **180 pending**. All 183 are reviewed.

The actual global audit exited 1 with only scope_review (180 pending),
comparator_preflight and library_replay (official comparator unavailable) failing.
Every other gate passed. Whole-corpus CI, official replay and GPU correctness
are not claimed. No push was retried after the prior authentication failure.

The previous 953 public records, 60 supplemental declarations, 180 unrelated
scope rows, 41 numerical rules and original metadata baseline remain unchanged.
The current catalog has 954 public declarations and 61 supplemental contracts
(55 execution, six protocol) across 16 tasks. The numerical queue now has 84
rules, including 43 new reviewed vector families, with 786 exact ports across
2,999 cases in five declarations. Whole-queue equality still rejects changed
or missing sources, reviews, cases and scalar ports. ReLU's existing adapter
body and the reviewed vector adapter syntax tree are preserved. All producer
and integration failures remain byte-preserved.
