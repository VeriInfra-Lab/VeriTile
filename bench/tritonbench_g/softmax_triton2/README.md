# softmax_triton2

Public-entry cleanup (2026-10-14): the current benchmark specifications are the `specification` declarations in [Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean). Superseded entries have been retired; reusable mathematical facts remain ordinary helper theorems, and unused output-summary declarations have been removed. Historical preservation counts below describe the earlier snapshots. This cleanup adds no proof coverage or task acceptance.

`Kernels.lean` contains the complete fp32 row-softmax source and a reciprocal
candidate. Both retain `tl.exp`, reduction order, independent row strides,
negative-infinity load padding and the masked store.

`Correct.lean` proves both implement `exp(xs i) / ∑ active j, exp(xs j)`, with
termination, active-address bounds and a complete flat-memory frame. Block size
is positive; dimensions and program IDs remain symbolic. When `n_cols ≤ B`, the
active prefix is the complete row. The original public correctness theorem and
its signature remain available, with supporting lemmas in `Proofs/Real.lean`.

`FPEquiv.lean` preserves the full execution result under `div_mul_rcp`. Its domain
checks finite actual computed numerator lanes and a finite, nonzero actual
computed denominator after the common prefix, including inactive arithmetic
lanes. No exp identity or performance claim is made. Host launch composition
and fixed-width address overflow are outside these per-program contracts.
Independent comparator replay remains pending on this host.
