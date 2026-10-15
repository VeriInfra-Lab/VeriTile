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
