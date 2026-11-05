# rbe_triton_transform

The repaired [Python source](../rbe_triton_transform.py) and the existing
[mask-reuse candidate](../rbe_triton_transform.py) have their complete
function bodies in [Kernels.lean](../Kernels.lean). Each caller has 31 top-level
statements; the shared frequency helper has six, including its fixed
`DIM: tl.constexpr = 128` and tuple return. Runtime arguments, pointer dtypes
and helper signatures are supplied outside the copied bodies.

The candidate reuses exactly the input real/imaginary masks at the identical
output coordinates. Integer casts, signed token positions, frequency powers,
sine/cosine, floating operation order, barriers and stores remain unchanged.
The candidate provenance file records both Python hashes and this exact delta.

[Correct.lean](../RealCorrect.lean) states the independent dense rotation formula on
`[batch,row,column]`, including odd tails and empty shapes. Its proof connects
the copied source to successful complete-grid execution, both masked components,
output readback and exterior memory frames. [FPEquiv.lean](../FPEquiv.lean) compares
the actual two complete programs with the existing empty numerical-rule set;
mask reuse does not rewrite any floating operation. The input and native-launch
conditions remain in [Contract.lean](../Contract.lean).

[source correspondence inventory](../../../../validation/source-correspondence.json) records both complete callers
and both Python copies of the helper. Body checks and local Lean builds are
separate from independent proof acceptance. The current acceptance state is
recorded in [the migration ledger](../../../../validation/body-migration-progress.json).
