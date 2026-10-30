# fast_rms_layernorm

Y and saved reciprocal RMS, DX with RMS correction, and complete-row DW sum in regular/Gemma modes.

- [fast_rms_layernorm_repaired.py](fast_rms_layernorm_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete forward row grid. Backward is the complete DW feature-owner grid followed by in-place DX row grid. Empty rows retain the original empty schedule and zero-DW input requirement.

## Conditions

- Call fixes positive cols and covered tile/block settings; regular/Gemma and runtime epsilon retained.
- Visible backwardDomain is saved r equal to the independent inverse-RMS formula, plus rows=0 implies initial DW=0. It is propositionally equal to Reference.realDomain h true, not stronger.
- Prepared Real IO contains actual preparation, schedule, finite logical channels, masked bounds/bindings and exact output/frame meaning; no host result or hidden source run is assumed.
- Checked Contract metadata/checks are unchanged; no extra FP32 or host obligations added.

Exact FP program pairs and numerical dependencies are recorded in the [fast_rms_layernorm contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
