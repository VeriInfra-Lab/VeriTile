# kv_cache_copy

[kv_cache_copy.py](../kv_cache_copy.py) contains the selected repaired copy and its
existing K-source-offset candidate. Both complete JIT function bodies appear in
[Kernels.lean](../Kernels.lean), including the original nonzero-start `tl.arange`,
`tl.static_range`, signed metadata reads and both K/V loads and stores. Invocation
types and scalar bindings stay outside the quotations.
[jit-correspondence.json](../jit-correspondence.json) checks both bodies.

The optional [static range frontend](../../../../VeriTile/Triton/DSL/StaticRanges.lean)
accepts natural constexpr trip counts. It checks immutable bounds and translates
aligned arange partitions with their exact width; executable source text remains
unchanged. An independently elaborated complete typed-program equality checks
both trip count and lane/store addresses in [the regression](../../../../tests/lean/StaticRanges.lean).

[Correct.lean](../RealCorrect.lean) proves both complete logical K/V tiles are copied
through the metadata-dependent windows, with the existing output-injectivity
condition. [FPEquiv.lean](../FPEquiv.lean) proves the two complete static-loop kernels
have identical full typed memory for unrestricted payloads. Only the two scalar
K-source offsets are commuted; FP32 load behavior, every split, alias semantics,
source access bounds and exterior memory frames remain in the proof.

[IO.lean](../IO.lean), [Contract.lean](../Contract.lean) and [proof helpers](.)
record the existing interfaces, conditions and actual source execution.
[Validation status](../../../../validation/body-migration-progress.json) distinguishes
local checks from independent replay; compilation alone is not completion.
