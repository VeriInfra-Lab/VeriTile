# Source scope and outstanding infrastructure

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

The source has one `@triton.jit` helper and one rank-1 kernel with two
`one_tile_per_cta` branches. The helper computes `pow(base.to(fp32), exponent)`;
the scalar is the base. The grid-stride branch visits `pid + j * num_programs(0)`
for every `j` in `range(0, tiles_per_cta)`. Both branches compute the unused
`num_tiles0`; `num_tasks` is unused. The rank-1 wrapper fixes both stride-order
constexprs to zero. Both block-pointer loads and stores check that axis.
Omitted load padding is unspecified, not zero.

## Pinned native dispatch

The pinned Triton 3.5.1 `core.extern_elementwise` looks up the argument tuple
before its arithmetic promotion walk. With the helper's explicit fp32 base,
the CUDA table accepts only fp32 exponent (`__nv_powf`) or int32 exponent
(`__nv_powif`), both returning fp32. Other exponent storage labels fail that
lookup; the wrapper does not validate or convert them first. This is a source
dispatch limitation, not a justification to replace the int32 branch by real
power.

`Proofs/audit_pow_dispatch.py` extracts and executes that production dispatch
function with minimal semantic stubs. It accepts explicit `--core` and
`--libdevice` snapshot paths, prints their SHA-256 hashes, checks the two native
symbols and records rejection of thirteen other dtype labels. It neither
compiles a GPU kernel nor supplies native numerical evidence. The checked
snapshot hashes are:

- core.py: `5c1281b67c0d949da34ccf1c3b68804a2caa2665953984d837d753e91af611fe`
- cuda/libdevice.py: `27b2a5d1e8db008bacefe6019f63922bbd65926de90bb1b527ee597477d2f365`

## Current formal source and remaining paths

`TypedSource.kernel` and `TypedIntegerSource.kernel` transcribe the two admitted
exponent families, fp32 and int32, with six formal floating destination labels
and natural strides. Each family has 24 templates: two complete constexpr
branches and two variants for every output label. All retain the base conversion,
input and destination conversions, unspecified padding and unused cdiv.
The candidate commutes only the natural `tile_id0 * tile_size0` product.
Python's `out0_bptr.type.element_ty` is resolved to the declared destination
label before `.to` expands. Explicit typed AST checks validate that resolution,
including fp64's retained fp32 source scope and the narrow storage tags.

The int32 path preserves dynamic integer provenance and dispatches to the
separate backend callback. Signed32 validity is checked on every computed lane;
values outside [-2147483648, 2147483647] cause evaluation failure rather than
wrapping. The real model uses integer `zpow`, including negative exponents and
negative bases. The general finite-flat contracts require actual integer input
cells for active lanes and range-valid exponents. Zero undef is explicit:
masked `unspecified` loads evaluate `Int.floor 0 = 0`, and those lanes pass
the same signed32 guard even though their output stores are masked.

Each exponent family has named one-tile and grid-stride real contracts for
arbitrary N, B, scalar base, natural strides and loop trip counts. They include
actual source trace safety, successful finite-flat execution, an independent
initial-input formula and the complete raw-cell frame. Positive output stride
ensures injective stores; the grid branch also requires distinct input/output
regions and positive grid size. Allocation disjointness/closure, active input
and output bounds, output membership and zero undef remain explicit premises.

Both families have separate nonempty finite witnesses that discharge all
premises: program id 1, grid width 2, strides 2/3, two actual grid iterations
and a partial last tile. Floating values are 4/8/2; signed integer exponents
-2/-3/-1 produce 1/4, 1/8 and 1/2. Full raw frames retain integer sentinels
of 37, including masked and skipped output cells. All 48 complete typed
numerical executions also preserve the destination tags. The numerical backend
uses actual real arithmetic and integer power; unsupported unary/bit operations
return unavailable `none`. Floating casts are identities, so these runs are
nonvacuity evidence rather than native numerical results.

The public imports reach 27 implementation modules; the separate exhaustive
`Proofs/Audit` entry reaches all 28 local modules. Formula audits rule out
source/execution dependencies in the independent expected-value functions.
Independent comparator replay remains unverified.

The following are outstanding source paths, so this directory is not complete:

- The wrapper's `StridedBuffer` explicitly supports signed strides and a
  shifted base pointer. Existing `BlockPtr.strides` and constructors use
  `List Nat`; full negative-stride block-pointer execution needs shared
  address semantics, bounds and flattening support.
- The wrapper does not constrain output dtype or runtime scalar dtype.
  Integer/bool destination conversion and integer scalar-to-fp32 conversion
  need their own source precision audit; floating destinations and a floating
  scalar binder do not certify those alternatives.

The fp32-exponent formula uses `Real.rpow`; the integer-exponent formula uses
real `zpow`. Both erase floating conversions. Neither proves IEEE rounding or
native libdevice accuracy. Negative bases
with noninteger exponents retain Mathlib's real convention, which differs from
native NaN behavior. Physical signed-width overflow, launch races, whole-grid
composition and host allocation validity remain separate obligations.

## Historical compatibility

All 36 historical declaration headers and 33 docstrings are retained.
The two historical specifications are directly in `Correct.lean`; supporting
declarations are in `Proofs/LegacyReal.lean`. The historical source uses
explicit zero block padding to preserve its prior mathematical meaning. It
does not serve as the new typed source and does not certify the missing paths.
