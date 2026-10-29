# kldiv_triton

forward: probability-zero-extended KL point losses or per-chunk Fin B sums with the original N tail mask. backward: -exp(target) or -target.

- [kldiv_triton_repaired.py](kldiv_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both full forward/backward row grids under arbitrary complete row permutations, independent runtime sy/st/so strides, every reduction code and both empty axes. Reduction outputs are per-chunk, not a host post-reduction.

## Conditions

- 0<B, explicit output row-address injectivity and work.Perm (List.range rows).
- Original all-payload mathematical domain, repaired zero-target/tail branches, signed64 casts and output masking are retained.
- Input binding and physical bounds remain in the shared IO judgment; no desired-output or run premise.

Exact FP program pairs and numerical dependencies are recorded in the [kldiv_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
