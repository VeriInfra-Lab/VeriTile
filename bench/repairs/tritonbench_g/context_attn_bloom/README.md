# Bloom context attention: source-connected public IO

`Correct.lean` has one public `specification`, `PublicKernel.correct_real`.
It states the shared `AlgorithmProgramIO.ImplementsOn` judgment for the actual
serial Real source grid and the independent normalized-block formula.
`Misc/NativeCorrect.lean` retains the ordinary auxiliary theorem
`PublicKernel.correct_fp32`, stating shared `ProgramIO.ObservesOn` for the actual FP32 source grid and the
existing ordered native instruction interpretation. `FPEquiv.lean` now proves an actual original/index-reuse program pair on the
same complete metadata-selected grid and native domain. The public results do not expose
raw states, source runners, returned states, or cell-by-cell postconditions.

The public data are explicit in `IO.lean`, `RealIO.lean`, and `NativeIO.lean`:
three payload regions, five signed metadata regions, finite input capacities,
metadata-dependent active output windows, repeated input labels, no private
scratch, original launch geometry/scale, and the actual shared source.
`Parameters.lean` contains only sizes, strides, block selection, and passed
scale. `Conditions.lean` retains the original shape, signed metadata, packed
sequence, output ownership/separation, and reached signed-address conditions.
`Domains.lean` also admits an empty grid explicitly: unused metadata outside
finite capacities cannot impose new conditions when no source program runs.

The Real formula retains raw exponential prefix sums, the finite `-1e8` causal
sentinel, padded lanes, and the separate guard that leaves the old accumulator
unscaled for an entirely future key block. Ideal strict causal softmax is a
different function. The source still loads/stores FP32; Real is its mathematical
interpretation with explicit dtype erasure and the established once-zero undef
convention. Local registers are preserved between the original serial launches.

The native formula is independently defined from initial input arrays and
scalar metadata. It retains every FP32 load/cast, ordered reduction, gathered
page, loop step, and live PV accumulator. Its pure observation uses the existing
SM80 m16n8k8 TF32 chains, normalizer FMA contraction, natural-exp lowering through
`0x3FB8AA3B`, reduction trees, and logical-column store owner. WordISA describes
individual primitive instructions. The auxiliary result is a source-to-native observation; the main pair compares
two actual source grids under that same observation.
Primitive instruction semantics and compiler-profile realization are model
boundaries; this cleanup adds no numerical experiment or floating rewrite law.

`Proofs/IOSourceSafety.lean`, `IOLoopSafety.lean`, and `IOAccessBounds.lean`
prove actual Structural load/store safety along evaluated states. The Real
adapter reuses `WholeSafety.complete`. Finite input bindings are connected to
both independent formulas, and the grid proofs compose actual predecessor
memories. The shared judgments include successful complete execution, all
active outputs, and the exact exterior cell frame, including arbitrary raw
tags/default reads and admitted input-region aliases.

Full old physical endpoints remain as ordinary theorems
`PhysicalKernel.correct_real` and `PhysicalKernel.correct_fp32` in
`Proofs/PhysicalReal.lean` and `Proofs/PhysicalNative.lean`. Their statements and
proof bodies are preserved apart from namespace/import movement.
`Proofs/PhysicalConnection.lean` identifies the shared execution with those
existing regional runners; the complete old flattened run, output words/bytes,
input-byte preservation, and exterior physical frames are retained.
`Misc/TensorBridge.lean` keeps all three original Tensor theorem statements and
bodies unchanged, including checked input binding, coverage, supplied output
identity, `None` return, empty calls, and device-context effects.

The public Correct/native proof helpers do not import reader entries. The optional
historical physical/Tensor closure retains one explicit frozen dependency:
`CheckedReal → CheckedBridge → Contract → Whole001.Correct`. The protected
Contract import is unchanged. CheckedReal reuses the historical
`Public.normalizedBlocks`, `Public.expected`, and `Public.expected_eq` to keep
its original Tensor target. The obsolete cross-task historical Correct import
was removed by the owner of the live `context_attn_fwd` helper.

The Python wrapper, `Contract.lean`, frozen evidence, accepted
precision/domain, and previously recorded validation status are unchanged.
This pass runs targeted Lean builds and statement/axiom/import checks only.
It does not rerun numerical experiments, independent native replay, or GPU
execution. Existing compiler-profile evidence remains scoped to its saved
profiles; no new compiler or runtime verification claim is made.

The main `PublicKernel.source_candidate_equivalence` compares the complete
original grid with a directly related integer common-subexpression elimination
candidate. Original `offs_m = start_m * BLOCK_M + arange` becomes
`offs_m = block_start_loc + arange`, reusing the row origin already computed
before it. This removes a repeated integer multiplication. No measured speedup
is claimed. Every floating operation, FP32 conversion, finite causal sentinel,
TF32/MMA/FMA graph, reduction tree, guard, metadata load, mask, gather and store
is unchanged. `Kernels.lean` now contains three full ordinary `triton` blocks:
original FP32, historical scale-first and the row-origin reuse candidate.
`Proofs/ReadableSource.lean` proves complete function equality to the previous
original and historical candidate, and binds the selected actual source call.
There is no whole-source macro or authoritative statement-list construction.

`Proofs/IndexCSE.lean` proves exact full-state execution equality for every
Algebra and arbitrary initial state, by deriving the reused row origin from
its actual assignment. It separately proves actual trace-safety equivalence.
Only proof decomposition uses take/drop. `Proofs/ProgramPair.lean` composes these
results through the original metadata-dependent empty/nonempty launch order.
The main pair retains exactly `PublicDomain.native`, arbitrary original
`WordISA`, all input words/raw tags, every native observed typed output cell,
and both exact raw exterior frames. Its narrow implementation view stores
actual invocation schedules and uses the original runner and observer.
No numerical rewrite rule, experimental dependency or deferred experiment is
introduced. The independent Real normalized-block theorem is unchanged.

The old scale-first candidate was investigated first and remains readable,
with its complete source identity proved. Its previous finite-site abstract
pair is not counted as a repaired all-word native/grid pair. The SM80 selector
reverses original QK scaling operands locally; connecting the changed complete
Expr tree to that selector would require its own contextual source proof.
The main pair above does not assume that missing program equality or hide it
as a numerical condition. It selects the directly related row-origin reuse
optimization instead, under the user's authorization for meaningful candidates.
