# rotary_transform

For each logical output: rotated prefix a*co-b*signed or a*signed+b*co, optional conjugation; identity coefficient fallback; unchanged input suffix.

- [rotary_transform_tile_checked.py](rotary_transform_tile_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected FP32 grid over batch/row/head/column result windows, including packed/fixed metadata, both signed widths and all original flags. Full result windows include suffix; writable frame covers prefix only.

## Conditions

- Config.Valid unchanged, including shape, metadata and address/configuration constraints.
- PublicIO.domain binds cumulative/offset metadata and the existing out-of-place suffix input agreement; inplace uses simultaneous binding.
- Empty total/half yields the existing empty schedule; result masks retain variable-length rows.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [rotary_transform contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
