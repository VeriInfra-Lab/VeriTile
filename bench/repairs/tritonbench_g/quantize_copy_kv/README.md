# quantize_copy_kv

kernel_correct

- [quantize_copy_kv_repaired.py](quantize_copy_kv_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every token in range count, all logical heads and full width, scattered code and FP16-scale outputs, both actual address spellings.

## Conditions

width>0, heads<=height, injective dest on active count, and preserved metadataMatches signed64 channel. IO supplies allocation bounds/physical bindings and full frame. Contract power-of-two columns and headBlock launch imply coverage; no additional Python input restriction.

Exact FP program pairs and numerical dependencies are recorded in the [quantize_copy_kv contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
