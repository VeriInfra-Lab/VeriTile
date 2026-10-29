# fast_ce_loss

Full-vocabulary log-sum-exp/loss, active-chunk LSE and target term, and saved-LSE input gradient with scaling/softcap derivative.

- [fast_ce_loss_repaired.py](fast_ce_loss_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Three actual kernel scopes: one complete forward row, one complete chunk program (only chunk zero writes loss), and one backward tile. Full forward sum is over vocab, not padded block. No host loss reduction/normalization is asserted.

## Conditions

- KernelReal.Flags ties source booleans to softcap/scaling nonzero tests.
- Forward: 0<vocab, vocab≤block and distinct loss/LSE regions; label domain is -100 or 0≤label<vocab.
- Chunk: 0<block, chunk*block<vocab, same distinct outputs and label domain. Backward retains arbitrary supplied LSE/upstream and no new label assumption.
- Contract exports prepared-storage ABI, views and launch geometry. KernelLaunch.forward_covered derives vocab≤block from existing next-power-of-two launch geometry; it is not an added Python guard.
- IO retains signed-int64 labels, runtime branches, target gather mask, ignored-label suppression of upstream loads, in-place gradient outputs and standard initial bounds/bindings.

Exact FP program pairs and numerical dependencies are recorded in the [fast_ce_loss contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
