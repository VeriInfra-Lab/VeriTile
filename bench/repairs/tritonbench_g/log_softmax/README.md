# log_softmax

- Name:

forward_grid_correctness

Statement:

Spec.Real (ReaderIO.forward M N K BM plan ⊨ fun xs m n k => xs m n k - Real.log (∑ j : Fin N, Real.exp (xs m j k)))
- Name:

backward_grid_correctness

Statement:

Spec.Real (ReaderIO.backward M N K BM plan ⊨ fun xs m n k => xs.upstream m n k - Real.exp (xs.saved m n k) * ∑ j : Fin N, xs.upstream m j k)

- [log_softmax_repaired.py](log_softmax_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both complete canonical forward/backward source grids on all M*N*K logical outputs, including empty axes. Logical summation dimension is N, not the padded C.

## Conditions

- Grid.RetainedBlock BM and Plan N (Grid.blockColumns N) remain public. Plan.covers is N<=C and all reduction trees are permutations of all C leaves.
- PublicIO uses typed dense input bindings and standard source access bounds; saved backward values need not be outputs of a previous forward invocation.
- Contract retains axis normalization, empty-input bypass and original nonempty N/K bounds; no coverage condition is added as a new Python restriction.

Exact FP program pairs and numerical dependencies are recorded in the [log_softmax contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
