Current checked revision: see `bench/repairs/tritonbench_g/pow_scalar_tensor/README.md`. The original conditional proof credit documented below is retained; explicit Python admission now lives in that separate wrapper.

# pow_scalar_tensor: retained accepted FP32 source contract

The selected original FP32 task already has a complete accepted whole-source contract. This integration restores its eight `Proofs/Whole*.lean` helpers and additive public entries instead of replacing the source with a copying/materializing wrapper. The original Python SHA-256 remains `142398199794dc7df724c341d3f86b1af60f20a9280b695efeac5e68d757471d`. `Kernels.lean` remains source-only; historical proof/host aliases have not been added to it.

`Whole.pow_scalar_tensor_real_source_outcomes` proves the independent `Real.rpow base (entryExponent i)` target. `Whole.pow_scalar_tensor_fp32_source_outcomes` proves the same complete original host call and both actual JIT branches with the native scalar primitives explicit. The exact supplied `out0` is returned, each logical result reads entry input, input accesses are preserved, and all output-exterior cells/bytes retain their entry values. `Whole.pow_scalar_tensor_fp32_host_equivalence` retains the original/candidate equality for the existing integer tile-offset commutation.

The retained successful domain is equal rank-one FP32 input/output views, `0 < n < 2^31`, the same CUDA device and successful guard, signed64 strides, valid signed views in finite aligned allocations, nonzero output stride, and disjoint accessed input/output footprints. Reverse views, zero input stride, gaps and disjoint subviews of the same allocation are admitted. Constructor shape/stride/offset metadata, capped grid, dynamic iterations, padded signed32 indices, signed pointer arithmetic and every active four-byte access are included. This is an explicit theorem domain for the unchanged original, not a newly checked broad Python domain. Empty input retains the original tile-heuristic division failure; shape/metadata/guard failures preserve entry state. Arbitrary overlapping, unresolved-negative and other-dtype Python calls are not newly admitted.

The mathematical Real domain admits positive bases, zero with nonnegative exponent, and negative bases with integer-valued exponents. FP structural correctness retains every FP32 exponent word and every admitted Python-float FP64 payload, including signed zeros, subnormals, infinities and NaNs. Python scalar payload is retained until `ScalarABI.extractFP32` after successful preparation; its resulting FP32 word binds the actual scalar slot.

The optional `Proofs/Numerics.lean` interfaces describe ordinary model boundaries: `ExactExtraction` binds the Python binary64 payload to launch-ABI FP32 conversion, and `ExactPower` binds the unchanged libdevice power primitive. They are not deferred numerical experiments for this task. Integer index commutation requires no numerical assumptions. Both interfaces quantify over raw words and supply no source, memory or whole-result premise. The current kernel-only entries in the checked revision start from the supplied FP32 scalar word. No GPU numerical claim is added.

Validation in this integration: the restored 21 unchanged original `relu_strided_buffer` dependency helpers, all eight whole-source helpers, numerical-port module and public Real/FP roots build with the standard axiom base. `Proofs/check_whole_source.py` executes the actual original AST with documented CPU adapters: six capped host partitions, four signed/broadcast/shared-allocation layouts, a complete dynamic-loop control, six power/ABI edge groups, and the original empty/shape failures. It checks exact output identity, input values and full output-exterior storage bytes. The power and host-conversion adapters are not native numerical validation. Historical compiler/ABI observations below remain evidence only for their recorded versions; CUDA GPU validation is pending.

## Historical notes

# pow_scalar_tensor

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

Public build: **3463 jobs passed**. Exhaustive audit: **406 theorems**.
Both FP equivalences use no additional assumptions. See `completion_audit.md`.

The fp32 and int32 exponent paths now have complete typed source models,
dimension-general real IO contracts, actual source safety and whole-backend
FP equivalence for natural strides. Signed strides, integer runtime scalar
conversion and nonfloating output conversion remain separate unverified paths.
This does not complete the full Python domain.

`Kernels.lean` and `Proofs/IntegerKernels.lean` retain both one-tile and
multi-iteration grid-stride branches. Each exponent family has 24 templates:
six formal floating output labels, both branches and both variants. The
scalar is the base. Block-pointer loads retain unspecified padding; the
candidate changes only the natural tile-offset multiplication order. Explicit
AST checks cover input precision, scalar fp32 conversion, the actual native
power operation, output conversion and complete control flow.

The integer path retains dynamic signed int32 exponents and the independent
`libdevicePowi` callback. Its formula is real `base ^ exponent` with an integer
exponent, including negative values. It does not convert exponents to floating
power. All computed lanes pass a signed32 range check before the callback.
General contracts require typed active input cells within range and zero
undef; the actual masked load converts that zero to integer zero.

`Correct.lean` exposes four new complete finite-flat contracts, for each
exponent family and each branch. They prove successful execution, independent
initial-input values, actual trace safety and preservation of every raw memory
cell outside active outputs. The one-tile contract permits aliases; the grid
contract requires distinct input/output regions and a positive grid size.
Both require positive output stride and finite allocation/access premises.

Separate finite witnesses discharge every premise with nonzero program id,
grid width two, strides 2/3, two loop iterations and a partial final tile.
Floating exponents 2/3/1 produce 4/8/2; integer exponents -2/-3/-1 produce
1/4, 1/8 and 1/2. Masked, skipped and adjacent outputs retain integer sentinels
of 37. Complete typed numerical executions cover all 48 templates and retain
all output storage tags. These use actual real arithmetic with identity casts,
not an IEEE rounding implementation.

The two historical specifications and all 36 historical declaration headers
and 33 docstrings remain unchanged. Their explicit zero-padding model is
retained in `Proofs/LegacyReal.lean`; `source_scope.md` states the current
source boundaries. `Correct`/`FPEquiv` reach 27 implementation modules, while
the separate `Proofs/Audit` entry reaches all 28 local modules.

`Proofs/audit_pow_dispatch.py` checks production overload lookup against
pinned Triton 3.5.1 snapshots: fp32 and int32 exponents select `__nv_powf` and
`__nv_powif`; thirteen other checked labels fail lookup. This is frontend
dispatch evidence, not native lowering, GPU execution or IEEE validation.
Independent comparator replay remains unverified.
