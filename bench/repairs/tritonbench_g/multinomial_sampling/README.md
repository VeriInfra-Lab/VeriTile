# multinomial_sampling

multinomial_categorical_correctness

- [multinomial_sampling_repaired.py](multinomial_sampling_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Exactly one complete source program at pid, all token blocks, active rows pid*block+i<batches, source strides, signed64 labels, arbitrary typed masked-load fallback payloads.

## Conditions

SatisfiesOn preserves SeedDomain (signed64 seeds, including masked lanes) and ProbabilityDomain on active rows: tokens>0, nonnegative weights and sum=1. Public IO supplies input binding, memory bounds, actual source safety/success, typed labels and full frame. Contract checks finite FP32 probability units summing to 2^149, integer/pointer envelopes and ordinary metadata; no restriction was added.

Exact FP program pairs and numerical dependencies are recorded in the [multinomial_sampling contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
