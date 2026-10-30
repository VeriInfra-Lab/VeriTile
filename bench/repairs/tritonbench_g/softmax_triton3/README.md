# softmax_triton3

exp(xs i+ms i) / sum (j : Fin n), exp(active xs j+active ms j), and its unmasked branch

- [softmax_triton3_repaired.py](softmax_triton3_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both original optional-mask branches, complete per-launch active row windows. Correct retains its selected contiguous row adapter; FP retains general original stride/mask/PID layout. Full checked grid is optional, not claimed by the selected main.

## Conditions

- 0<B
- Existing n≤B retained in both Real statements
- MaskedKernelIO₁/₂ preserve input/output bindings, active masks, safety and exterior frame
- FP: B<2^32 and bound Pair.Ports model; every raw FP32 input word remains admitted

Exact FP program pairs and numerical dependencies are recorded in the [softmax_triton3 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
