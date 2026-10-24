# reversed_cumsum_scalar

This port retains the complete `chunk_global_reversed_cumsum_scalar_kernel`
from TritonBench-G at upstream `603e28a`. The candidate changes only the input
pointer product `i_bh * T` to `T * i_bh`.

The new public modules are `Kernels.lean`, `Correct.lean` and `FPEquiv.lean`.
Both public correctness specifications, the floating equivalence and complete
typed success witnesses have passed root validation. Historical prepared-input
and carry-buffer slices are preserved in `Proofs/LegacyReal.lean` and are
excluded from these public imports. Overall source acceptance remains
incomplete because the native and host boundaries below remain open.

The real specification uses the initial input memory and unspecified-value
oracle. At every valid time index it gives the ordinary suffix sum plus the
sum of the padded final chunk's unspecified values. The load has no
`padding_option`; those tail values enter `tl.sum` before any output is
stored. The ordinary suffix formula therefore needs zero total tail, which
holds for aligned lengths or a zero tail oracle. This also affects the
original example T = 4, BT = 16.

The complete source theorem executes every signed reverse-loop iteration
from arbitrary registers, including T = 0, with distinct input/output
regions and BT > 0. It proves full tagged output values and preserves every
tagged cell outside the active row. The finite allocator theorem adds active
address bounds, disjoint covered allocations, and the generic bridge's
explicit zero-oracle condition. No prepared carry state is assumed.

The typed equivalence preserves all floating operations, the forward scan,
reduction, casts, padding and full state. It uses no numerical assumptions or
floating domain guard. The consistency witness uses actual mathematical
tile scans and reductions with bf16 input, fp16 output, empty registers,
row 1, T = 17, BT = 16 and nonzero padding. Identity casts in that witness
make no IEEE rounding claim.

The nonzero-tail real regression executes both originals and candidates:
constant input 2 and unspecified padding 7 produce 139 at the first valid
index and 107 at the last, with neighboring boolean cells preserved.

The source AST and native offline compilation receipts are in
`bench/review-checkpoints/2026-10-13/reverse-source-ast` and `reverse-native`.
The latter covers 150 scalar profiles and performs no GPU execution.
See `source_scope.md` for the linear-row host ABI, fixed-width index,
native/IEEE refinement, launch composition and independent replay boundaries.
