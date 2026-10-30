# rms_matmul_rbe

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

The complete typed callee and sequential QKV caller are in `Kernels.lean`.
Both weight branches retain every statement, the dynamic `cdiv(K, BK)` loop,
native integer power, typed dot, raw E5M2 reinterpretations and conversions.
Q always stores fp16; K and V store E5M2 in the int8 weight branch and fp16
otherwise. The candidate commutes only the final natural output-row product.

`Correct.lean` exposes both historical specifications unchanged and two new
complete non-FP8 IO specifications, plus callee and QKV byte IO specifications. They prove actual access safety,
successful finite-flat execution, independent initial-input RMS/GEMM formulas
and the full raw-cell frame. The QKV result is a single three-output contract,
using the three original weight/output stride tuples and framing cells outside
the union of active writes. K need not be divisible by BK: the bounds include every
unmasked padding lane. Only active output addresses require injectivity.

`FPEquiv.lean` proves complete callee and QKV state/failure equivalence for
arbitrary arithmetic, integer-power, dot and E5M2-decode backends, with **no FP
assumptions**. Nonfinite or invalid byte payloads remain within these equivalence
statements. They do not assert native frontend admission or IEEE equality.

`Proofs/Precision.lean` checks independent templates for all 48 callee profiles
(six floating X labels, two weight profiles, two independent output types,
two variants), and all three QKV loops and output casts. The templates cover
loads, fp32 power and moment, source-local multiplication promotion, separate
dot/carry, normalization and pointer increments. In particular fp64 X-local
multiplication is retained while the dot result/carry remains fp32.

`Proofs/Witness.lean` and `Proofs/QKVWitness.lean` discharge the finite allocation
and all execution hypotheses for nonempty partial K/N tiles with batch id one
and independent strides. The complete QKV run returns **Q=1, K=2, V=3** and
retains an integer sentinel in a masked output cell. These are exact-real
model runs.

`Proofs/NumericExecution.lean` additionally proves actual typed execution of
the complete three-pass source for both weight profiles and both variants.
Its nonempty K=1/BK=2 example produces **Q=2, K=3, V=4**, with Q stored as fp16
and K/V stored as fp16 or E5M2 according to the weight profile. All three masked
output cells retain their integer sentinels and the raw input bytes remain
unchanged. The backend uses real arithmetic, actual sum/dot/integer power and
finite E5M2 decoding; its identity casts do not establish IEEE rounding.

The 70 historical declarations (69 public, one private helper) keep their
source signatures and docstrings; the two historical specification blocks are
in `Correct.lean` for inventory discovery. Historical module prose is retained
in `Proofs/LegacyReal.lean`; `source_scope.md` supersedes its outdated claims
about host-provided K divisibility, omitted branches and output dtypes.

The byte IO contracts require actual integer-cell inputs and finite E5M2
decoding at every unmasked address. They compare against independent pure
`PaddedRMSGemm` arrays and retain the raw byte-memory frame. Callee byte
witnesses produce two; the complete byte QKV witness produces **Q=2, K=3, V=4**
with the same finite partial-tile geometry and masked integer sentinels.

Still pending: formal coverage of unconstrained nonfloating X inputs, native admission,
IEEE/storage-byte correspondence and independent comparator replay. No replay
result or GPU validation is claimed. See `source_scope.md` for exact boundaries.
