# logsumexp_fwd

- Name:

logsumexp_kernel_correctness

Statement:

Spec.Real (ReaderIO.view a scale jobs ⊨{ReaderIO.realInputs a} fun xs row segment =>
      (Real.log (∑ i ∈ Finset.univ.filter
        (fun i : Fin a.block => segment.val * a.block + i.val < a.columns),
        Real.exp (if a.hasScale then
          ReaderIO.realAt xs row (segment.val * a.block + i.val) * scale
        else ReaderIO.realAt xs row (segment.val * a.block + i.val))) : WithBot ℝ))

- [logsumexp_fwd_repaired.py](logsumexp_fwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete rows*ceil(columns/block) scratch-output grid in any permutation. Final Torch row aggregation is outside this kernel theorem. Empty grids remain empty.

## Conditions

- RawKernel.Geometry carries positive block; jobs.Perm(List.range(rows*blocks)) remains explicit.
- Interface.RealInputs requires every valid logical input to be a Real embedding in WithBot Real; it is not an extra native finite-word admission guard.
- Runtime scale, scale/no-scale branch, masks, typed windows/bounds and input binding remain unchanged.

Exact FP program pairs and numerical dependencies are recorded in the [logsumexp_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
