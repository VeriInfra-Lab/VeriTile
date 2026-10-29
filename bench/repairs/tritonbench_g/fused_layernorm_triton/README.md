# fused_layernorm_triton

`Kernels.lean` contains complete, ordinary `triton` quotations for the original
selected row source and the related `hoisted` candidate. The first Welford
iteration is specialized, the remaining statistics and normalization iterations
are actual strided loops, and the explicitly selected binary Welford tree is
expanded by the shared typed helper. The single row axis is projected to scalar
registers for normalization. The candidate moves those three invariant
projections before the second loop and uses the same FP operations in the same
order. The matching Python candidate is
`triton_red_fused_native_layer_norm_hoisted`, with wrapper
`fused_native_layer_norm_hoisted`. It scalarizes the constant row values once and
uses one-dimensional normalization tiles. No speedup is claimed without
measurements.

`FPEquiv.lean` now proves `whole_fp`: the actual full original and candidate
programs satisfy `original ≡[[]] candidate`. Both complete row grids safely
succeed, agree on the typed FP32 Mean/Rstd and BF16 Y word observations, and
preserve their entire exterior cell frames. The input carrier admits arbitrary
`Native.Term` values, including every raw FP word; no finite-value restriction
is introduced. The instruction model, compiler FMA profile, source precision,
conversions, masks and selected ordered helper tree remain explicit. This is an
exact structural transformation and uses no experimental numerical rules.

`Correct.lean` retains the independent Real theorem `kernel_real` and its full
logical IO: mean is the sum divided by D, variance is the sum of centered squares
divided by D, inverse standard deviation is `1 / sqrt(variance + 1e-5)`, and Y is
`(x - mean) * inverse * weight + bias`. Mathlib establishes the moments formula.
Inputs are contiguous, offset-zero FP32 X `[S,D]` and weight/bias `[D]`, with
D > 0; S may be zero. Both original blocks 1024/2048, stages 1 and warps 8 are
retained. Inputs may alias each other; the three outputs are separated from
inputs and one another. A complete selected reduction tree is required for the
independent Real moments theorem.

`Proofs/ReadableStatistics.lean`, `ReadableNormalize.lean`,
`ReadableNormalizeLoop.lean` and `ReadableWhole.lean` connect the full quotations
to the retained ordered recurrences, every output column and the complete grid.
`ReadableSafety.lean` follows the actual helper branches and strided loops.
`PairIO.lean` and `PairEquivalence.lean` connect this execution, safety and both
frames to the public two-program judgment. `SourcePieces.lean` contains only
proof-side source projections, with their complete context equations checked by
reduction. `SourcePreservation.original` proves equality of all regional memory
between the readable original full grid and the previously retained assembled
source, for every primitive algebra. `LegacySource.lean` preserves that old AST
as a historical proof fixture; it is not an authoritative source entry.

The optional single-source native, Tensor and historical host entries remain in
`Misc/`. Main entries and their proof helpers do not import `Misc/`. Existing
Int64 index and typed-byte bindings are retained. The primitive model/profile
boundary is not a new hardware conformance claim.

`check_source.py` runs both actual Python ASTs on CPU arrays, checks the unchanged
wrapper guards, tails, multiple chunks, input aliases, single stores and exact
pair output agreement. These are source regressions, not numerical-rule
admission or GPU conformance experiments. `bench/tests/WelfordSourceContexts.lean`
checks exact transparent helper expansion and rejects unsupported precision,
shape, axis and target overlap.
