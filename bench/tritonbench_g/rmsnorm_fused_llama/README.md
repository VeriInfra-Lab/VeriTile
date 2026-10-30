# rmsnorm_fused_llama

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

The dependency-aware build of `Correct`, `FPEquiv`, and `Proofs.Audit` passed
(3457 jobs). The exhaustive audit checked 852 theorem declarations. Coverage
registry integration and independent replay remain separate gates.
The historical declarations are preserved in `Proofs/Legacy.lean` and `Correct.lean`.
Their retained single-block commentary describes the old proof scope; the new
source audit below corrects the actual wrapper boundary.

`Kernels.lean` retains both source loops for arbitrary `N`, zero-padded fp32 input
loads, fp32 weight conversion, and the explicit output fp16 conversion before
conversion to destination storage. Its 216 floating profiles vary X, W, and Y
independently across fp32, fp64, fp16, bf16, f8e4, and f8e5. The Python wrapper
correlates X and Y storage; the low-level family is broader. The candidate commutes
only the exact natural multiplication in the Y row-pointer update.

The wrapper overrides its earlier block choice with `128 * 2^7 = 16384`.
Consequently the earlier size guard does not imply a single source iteration for
all element sizes. The new contracts require positive block size, not `N ≤ B`.
The real formula uses the complete logical row sum of squares and weights.
Snapshot IO requires Y distinct from X and W; the FP equivalence permits all
aliases and quantifies arbitrary backend states, failures, and undefined values.
Masked W loads retain unspecified padding. Flat transport explicitly assumes
zero undef, allocation separation/coverage and active-address bounds.

The real contracts apply to dtype erasure (casts are ideal identities), while
FP equivalence retains both output casts and every backend precision dispatch.
The numerical witnesses use an exact-real backend with real square root and sum;
its identity casts are not an IEEE rounding implementation. Native fixed-width
pointer overflow, negative strides, grid scheduling/race freedom, device execution,
and discrete/mixed integer storage conversion are outside this checked family.
N=0 remains a totalized mathematical boundary; positive epsilon suffices for a
positive real square-root argument, and ordinary RMS interpretation uses N>0.

The five public heads and the explicit precision/numerical checks use only
`propext`, `Classical.choice`, and `Quot.sound`; FP assumptions are empty.
The numerical checks cover both variants, fp64 input/weight conversion, N=1 and
N=3 with B=2: rstd=1/3, active outputs=2, and adjacent/tail sentinels remain 7.
The separate finite erased-real witness covers all 216 profiles and both
variants with N=3, B=2 and a nonzero row. It discharges every allocation and IO
premise, returns three values of 2, and preserves the full raw-memory frame,
including integer sentinels of 37. `General.rowSpec` passed an explicit
noncircularity check against source and execution definitions.

All 33 historical declaration headers and 26 docstrings remain unchanged.
The public imports reach 18 implementation modules; `Proofs.Audit` is the sole
separate test entry and reaches all 19 local modules. See `completion_audit.md`
for the reproducible build command and current receipts.
