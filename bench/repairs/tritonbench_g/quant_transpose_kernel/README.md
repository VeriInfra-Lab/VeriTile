# quant_transpose_kernel

kernel_real

- [quant_transpose_kernel_repaired.py](quant_transpose_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original grouped 128x128/G8 grid with strided matrix input, scalar inverse and transposed integer-byte output, both retained integer-address spellings.

## Conditions

KernelGrid.Separate p retained. Public IntegerProgramIO byte view includes exact source execution/safety/frame and unsigned 0..255 byte observation. Contract guards positive M/N, int32 padded-index/pointer envelopes, nonnegative strides with one contiguous dimension, FP32/current CUDA/alignment.

Exact FP program pairs and numerical dependencies are recorded in the [quant_transpose_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
