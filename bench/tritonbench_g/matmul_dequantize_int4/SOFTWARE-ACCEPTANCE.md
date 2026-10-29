# matmul_dequantize_int4

The complete original-source software bit contract is [SoftwareCorrect.lean](SoftwareCorrect.lean). It fixes +0-seeded ascending fp32 multiply/add inside each block, separate rounded block accumulation, three original half dequantization rounding points, final half conversion and ordered scatter. Both NO_GROUPS, all original K iterations/parameters, PackedInputs and active-output injectivity remain; no finite-only or single-block restriction is added.

[CompiledSoftwareCorrect.lean](CompiledSoftwareCorrect.lean) separately proves nonfused and fused live-carry graphs. The latter explicitly reads current SSA registers in a task-local half-FMA instruction between the original shared prefix and suffix. These software graphs have different numerical results; their contracts do not establish native MMA ordering, FTZ/NaN or physical ABI equivalence.

Fifteen modules and 58 named axiom checks passed independent compilation. See the [reviewed proof](../../review-checkpoints/2026-10-15/dequant-int4-source-compiled-software-closure/README.md) and [current integration](../../review-checkpoints/2026-10-15/dequant-int4-source-compiled-software-metadata/README.md). The task now has an accepted software profile; GPU execution remains unbound. The [fixed-case capture tool](../../review-checkpoints/2026-10-15/dequant-int4-native-characterization/README.md) passed 45 independent tool tests and eight offline compiler configurations; the actual GPU attempt is NOT_RUN.

The [historical README](README.md) stays byte-for-byte bound to the frozen candidate provenance. This page records the current accepted software entries.
