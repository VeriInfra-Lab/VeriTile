# attention_fwd_triton3

Full signed-window attention row maximum and weighted denominator/numerator, seeded by INIT or entry carry. Continuation returns M/L/O; END returns M+log2(L) and O/L with L store masked.

- [attention_fwd_triton3_scalar_cast.py](attention_fwd_triton3_scalar_cast.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full ordinary grid, all original padded O/M/L output rows/channels, grouped heads, windows and masks. The END denominator output store remains masked by the public IO.

## Conditions

- Valid c retained explicitly: block geometry, dimensions, strides/output layout, separation, even-tail and machine/index safety conditions.
- INIT/END/window/complement flags, signed offset/size, no-loop versus executed all-masked distinction, actual query tail mask, six Real inputs and runtime scale binding retained. No positivity condition added to total Real division/log.

Exact FP program pairs and numerical dependencies are recorded in the [attention_fwd_triton3 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
