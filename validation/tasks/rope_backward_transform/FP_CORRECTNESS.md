# Finite-precision backward rotary contract

`FPCorrect.lean` exposes the original backward branch with all four fp32 Q/K
writes, exact per-operation backend results, a concrete binary32 software
reference and the complete memory frame. The theorem covers both source branches, including the original Python
launch's `BACKWARD_PASS=true` configuration. All dimensions and strides remain
symbolic. Precomputed cosine/sine inputs have the same boundary as before.

`Proofs/TypedShape.lean` proves this benchmark's full JIT body is the same source
body used by `rope_transform`, and connects the typed implementation by erased
source equality. The numerical contracts reuse the two-branch theorem, whose
backward branch computes low*cos + high*sin and high*cos - low*sin with separate
rounding after each product and add/subtract.

The exact software profile and remaining native/compiler obligations are
listed in [the shared rotary contract](../rope_transform/FP_CORRECTNESS.md).
The source-shape equality does not by itself establish numerical correctness;
the typed execution and independent rounded-memory proofs provide that layer.
