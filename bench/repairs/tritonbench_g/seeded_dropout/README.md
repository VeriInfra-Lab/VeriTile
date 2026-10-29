# seeded_dropout

At logical flat offset i: if canonical Philox-10 randReal(seed,i,bits32)>p then x/(1-p), else zero.

- [seeded_dropout_repaired.py](seeded_dropout_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected per-program tile of 1024 lanes at pid; original active mask pid*1024+lane<n and exterior frame. It is not advertised as whole n-element host execution.

## Conditions

- n<2^31, pid<gridCount n, SeedABI for original Python i32/i64/u64 JIT branches.
- Real division is total; main theorem is valid for all real p and therefore includes the Contract range 0≤p<1 without adding/removing an input restriction.
- KernelPublic.target uses pid*1024+lane logical offset, preserving RNG position rather than a statistical model.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [seeded_dropout contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
