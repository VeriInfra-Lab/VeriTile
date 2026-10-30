# Checked direct `bgmv_shrink_kernel`

The checked wrapper executes the original 64-split kernel directly on supplied
contiguous storage. FP16 and BF16 X/W are both retained, with matching input
formats, FP32 Y, and int32 or int64 metadata. Each active output cell receives
64 atomic updates in its own actual runtime order, starting from its original Y.
The wrapper returns `None`. Original Python and the widened historical kernel
AST are unchanged.

Checks cover tensor/layout/rank/dtype, shared CUDA device, contiguous nonlazy
storage, finite Python-float scaling, matching shapes, output capacity, native
scalar/grid widths, every formed signed64 byte offset, output/read-backing
exclusion, nonempty pointer alignment, and adapter values. W may be `[L,N,K]` or
`[L,1,N,K]`; the latter retains its original squeeze alias. Aligned nonzero offsets
and read/read aliases are accepted. K=0 retains the source's 64 zero-accumulator
atomic contributions. Empty B or N returns before metadata reads or launch.

There are no snapshots, padded allocations, layout conversions or copyback.
The direct launch uses canonical strides K, N*K and Y's column count and enters
the tensors' actual CUDA device context. Sentinel rows and all output columns
outside `[0,N)` retain every original byte.

`Contract.lean` proves equivalence of the ordered Python checks and InputContract.
`Correct.checked_real` derives complete source events and proves the independent
K-term GEMV in any per-cell atomic order. `FPEquiv.checked_fp` derives those source
events independently for the exact typed graph, then proves four concrete output
bytes and the complete frame. `checked_caller_bytes` preserves the complete
original X/W/index backing allocations.

`Workset.fiber` proves each active destination has exactly 64 source events.
`Linearization.Run` is the ordinary runtime boundary of individual local atomic
transitions; it has no folded-output equation. Its final readback is proved.
`Bounds` derives active capacity and every generated lane's signed64 offset from
the checks. `Physical` derives actual pointer words, metadata byte spans, full
backing frames, empty behavior and arbitrary failed-write subset/order frames.
No numerical correctness or exception atomicity is claimed for a failed launch.

The native numerical graph uses explicit scalar reduction trees and a separate
fixed-arity atomic instruction. Its compiler binding pins the complete production
AST, selected trees, and primitive lowering. See `NUMERICS.md`.

Only Kernels, Contract, Correct and FPEquiv are root Lean entries. Historical
padded/snapshot theorem closures are preserved in Proofs; LegacyCorrect and
LegacyFPEquiv retain the earlier public proof credit. Public Real and FP entries
remain independent.

Run `python bench/repairs/tritonbench_g/bgmv_shrink_kernel/check_contract.py`.
It passes 18 complete direct cases and 37 rejections, covering FP16/BF16, K=0,
K=255/256/257/16385, 3D/4D weights, both metadata widths, varied split orders,
sentinel/all-sentinel/empty paths, aliases and aligned offset views. It checks
actual wrapper/kernel AST execution, original objects, canonical strides, full
caller byte frames and partial atomic-write/context restoration. These are CPU
and structural checks, not native GPU numerical conformance.
