# int8_quantization

The repaired [Python source](../int8_quantization.py) and existing [bias-fold candidate](../int8_quantization.py) each have one complete 23-statement function body in [Kernels.lean](../Kernels.lean). Pointer storage and immutable constexpr arguments are supplied outside the pasted body. [source correspondence inventory](../../../../validation/source-correspondence.json) checks both copies; [candidate-provenance.json](../../../../validation/tasks/int8_quantization/candidate-provenance.json) records the single changed sign-bias expression.

[Correct.lean](../RealCorrect.lean) proves both quantized codes and block scales for every scheduled block, including invalid blocks, zero scales, active tails, access safety and the exterior memory frame. Its mathematical interpretation computes `C**-0.5 * 1.44269504` with Mathlib real powers and proves the inverse-square-root factor, without a supplied-factor premise. The real classification threshold is the original decimal `3.4028234663852886e38`.

[FPEquiv.lean](../FPEquiv.lean) compares the two complete sources with the same input/scale storage, padded lane order and compile-time primitive interpretation. Only `0.5 * where(sign, 1.0, -1.0)` changes to `where(sign, 0.5, -0.5)`; the two changed products are proved in the existing software FP32 profile. The remaining primitive operations keep their explicit bindings. The constexpr expression tree preserves each operation before runtime tensor arithmetic; the exact Real model does not claim binary64 or hardware power accuracy.

Both `.to(../tl.int64)` casts, the INT8 conversion, classification, sanitation, clamp, masks and stores remain in the source. The native signed-width address connection and host admission conditions are retained in [Contract.lean](../Contract.lean) and [Proofs/](.). No GPU run is required by this source migration.

Completion requires current-source independent replay; a body check or successful build alone is not a completion certificate. The [validation records](../../../../validation/body-migration-progress.json) record pending work separately.
