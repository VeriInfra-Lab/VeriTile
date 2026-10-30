# rotary_transform_ops

Explicit conjugated/ordinary rotary prefix a*co-b*signed or a*signed+b*co, with identity coefficient fallback; suffix equals initial input.

- [rotary_transform_ops_repaired.py](rotary_transform_ops_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Original complete whole-grid read/write IO, all four signed metadata ABIs, both alias modes and all flags. Full output tensor readbacks include unchanged suffix and empty branches.

## Conditions

- Config.Valid unchanged.
- Interface.Ready binds metadata to logical configuration and retains out-of-place suffix-copy agreement.
- result masks describe valid per-batch lengths, writable masks only rotary prefix.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [rotary_transform_ops contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
