# flash_attn

Normalized weighted V and base-2 log mass over all visible logical keys, preserving the exact source decimal score conversion.

- [flash_attn_scalar_abi_checked.py](flash_attn_scalar_abi_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected flash_attn_scalar_abi_checked source grid and both full O/L tensors, all logical head dimensions and visible logical keys; independent Q/K/V/O pitches retained.

## Conditions

- Real theorem retains x.shape.Valid, OutputInjective x and Logical.RealInputs x (ordinary finite logical Real inputs encoded as some).
- Public IO exposes actual source schedule, scalar register binding, layouts, masks, initial bounds/bindings and exact frame.
- Selected Contract adds the actual raw-integer scalar ABI range check; existing float payload checks remain. This review does not silently replace the decimal base-2 conversion by exact natural exponential scaling.

Exact FP program pairs and numerical dependencies are recorded in the [flash_attn contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
