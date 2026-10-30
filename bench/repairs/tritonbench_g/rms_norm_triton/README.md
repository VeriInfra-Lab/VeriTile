# rms_norm_triton

xs row col * (1 / sqrt ((sum k:Fin N, xs row k * xs row k)/N + eps)) * weights col.

- [rms_norm_triton_repaired.py](rms_norm_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Whole M-row grid in any complete row launch order, including empty grids, with dense X/Y and shared W windows.

## Conditions

Explicit conditions:

- c:Whole.Layout, whose complete field proves N≤B
- X≠Y and W≠Y; X/W read-only aliases allowed
- rows.Perm(List.range c.M)
- ProgramIO retains coherent input binding, safe accesses, successful grid execution, every logical output and exterior frame.

Contract correspondence:

Whole.realFormula in the original Contract is the independent unpadded N-term RMS formula. Existing Layout.complete explicitly relates logical N to source block B; no new Python limit or domain was added.

Exact FP program pairs and numerical dependencies are recorded in the [rms_norm_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
