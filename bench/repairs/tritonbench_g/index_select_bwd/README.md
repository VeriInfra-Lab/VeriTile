# index_select_bwd

index_select_bwd_kernel_correctness: indexIO implements i ↦ current[i]+incoming[i].

- [index_select_bwd_repaired.py](index_select_bwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete source slice of one selected row, with unit destination stride and count<=block; all active columns, source safety and exterior frame. Contract also states the complete duplicate-index gradient sum/ordered fold; this retained kernel main theorem does not pretend one slice is the whole Python loop.

## Conditions

Public conditions:

- p.stride=1
- p.count<=p.block
- Public ProgramIO input/output windows and active count mask
- FP adaptive comparison also requires p.count<=512

Contract and domain correspondence:

Contract fixes contiguous same-stride tensors, valid signed indices and original duplicate-preserving input order. The main independent target is Contract.sliceTarget; whole host loop/Tensor applications stay optional and were not added as new obligations.

Exact FP program pairs and numerical dependencies are recorded in the [index_select_bwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
