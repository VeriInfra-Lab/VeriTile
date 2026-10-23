Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

`FPEquiv.static_unroll_equiv` compares the complete selected source against
an actual constexpr token-loop unrolling candidate, with an empty experimental
rule set. Existing native observations remain optional ordinary theorems in
`Misc/NativeCorrect.lean`; the independent Real specification is unchanged.

# Multinomial sampling kernel

Read `Kernels.lean`, `Correct.lean`, then `FPEquiv.lean`. The source-only
module contains complete direct original/candidate DSL quotations; the earlier
whole-source macro is removed. Existing historical storage profiles remain
direct quotations; the selected Python contract uses FP32 scores,
int64 seeds/offsets/labels/output, `BLOCK=8`, `BLOCK_N=128`, and eight warps.

The public `Correct.lean` statement is
`UndefProgramIO.SatisfiesOn` for the independent inverse-CDF relation.
`Misc/NativeCorrect.lean` has one `UndefProgramIO.ImplementsOn` statement for the independent
source-precision recurrence. Its arbitrary primitive model includes every
existing `software .fp32 plans` instance; it does not assert native=Real.
`IO.lean` declares the four memory channels, original strides and output mask.
`LogicalIO.lean` separately declares the three integer masked-load fallback
channels. Their values remain arbitrary; the signed64 seed domain is exactly
the original condition, proved in both directions by `Proofs/IOUndef.lean` and
`Proofs/IOFacts.lean`. Fallback channels do not require backing memory.

The Real judgment and optional native theorem cover the complete selected source program at any PID,
including token tails, zero-token native behavior and all original aliases.
This is the original per-program scope; program/lane coverage remains a
separate helper. `Proofs/SourceSafety.lean` follows the actual prelude, complete
range loop and epilogue; declared windows are connected to every masked access.
Execution, all active outputs and the exact exterior cell frame are established.
The original raw-state and Tensor endpoints remain unchanged ordinary theorems
in `Proofs/KernelCompatibility.lean`, `Proofs/FPCompatibility.lean` and
`Misc/TensorBridge.lean`.

The source retains signed32 offset narrowing, canonical ten-round Philox,
FP32 scan and carry, carry updated with the block maximum, strict lower and
inclusive upper comparisons, first valid lane within each block, replacement
by a later matching block, and fallback to the first label if no block matches.
The selected-label pointer uses `min(valid_pos, num_tokens - 1)`, the previously
accepted repair. The candidate changes only token-loop scheduling as described below.

`TensorBridge.tensor_kernel_correctness` proves successful execution of the actual
kernel in Real arithmetic against logical Tensor probabilities, labels, seeds
and offsets, with the complete exterior typed-memory frame and selected-pointer
bounds. It derives the signed seed domain and normalization from local initial
loads and the executed guard's finite FP32 ratio observations. The independent
`TensorResult` target uses the `Spec.Result` inverse-CDF relation and the exact
Real decoding of the source's rounded FP32 Philox sample. Zero selects the first label; positive samples use the unique interval
`(prefix[j], prefix[j+1]]`. The generic helper
`multinomial_kernel_correctness` also covers arbitrary nonnegative normalized
Real rows. Neither result assumes FP exactness or
successful execution. Existence and uniqueness of the mathematical label follow from `Spec.result_exists_unique`.
No ideal continuous distribution or exact categorical probability law is claimed.

`TensorBridge.tensor_kernel_words` proves successful execution of the actual FP32
kernel, equality of the entire typed memory to the rounded recurrence over
logical Tensor values, active signed label readback, the exterior frame and
selected-pointer bounds. `TensorReference` reads logical score/label/seed/offset
words and retains signed32 RNG offset narrowing. The generic helpers
`exact_kernel_words` and `opaque_kernel_outcome` preserve the state-level result;
the latter also covers arbitrary supported primitive callbacks, with no algebraic-law premises. The finite-word profile
keeps each addition, subtraction, comparison, cast and scan/max tree. `Plans`
records the selected reduction/scan schedules; the theorem does not replace one
schedule with another. Neither result assumes the desired output or a successful
run.

The wrapper assertions are unchanged. Scores are ordinary contiguous CUDA FP32
strided tensors of rank two with a positive token dimension. Supplied labels
are ordinary contiguous int64 tensors of the same shape and device; the default
expanded labels intentionally have zero batch stride. The kernel branch requires
same-device contiguous int64 seed/offset vectors. The guards reject unresolved
negative/conjugate views and check alignment, signed32 padded indices and uint64
formed pointers, including masked lanes. Zero batches remain admitted; the
singleton branch retains the existing clone behavior and does not submit a kernel.

Every admitted row has finite nonnegative FP32 values whose exact values sum to
one. `_probability_rows` checks this by summing integer multiples of `2^-149`;
it neither renormalizes nor changes inputs. A rounded `torch.sum(...) == 1`
is insufficient: FP32 `[0.1, 0.2, 0.7]` is rejected. `KernelMemory.probability_domain`
derives the Real normalization domain from those scalar readback observations.

`KernelMemory` proves unique program/lane coverage, every padded matrix/vector
pointer bound, native signed32 product/sum exactness, and unconditional selected
label capacity under the pointer clamp. Shared Torch FP32 Tensor laws connect
score logical values, storage capacity and the actual data pointer. The small
int64 Tensor binding covers supplied and expanded labels as well as vectors;
`source_score` and `source_integer` connect actual scalar loads to logical Tensor
reads. The public Tensor corollaries compose these bindings for every input role
using one common entry-memory byte reader and the selected little-endian scalar
ABI. Overlapping inputs and the default expanded labels remain supported.
`TensorInputs.Loaded` binds only initial scalar reads; `Probabilities` binds the
finite exact ratios checked by the Python guard. `encode`, `denote` and
`wordToReal_encode` state the FP32-word-to-Real interpretation explicitly. None
of these interfaces assumes a kernel result. Active outputs remain signed64
labels for arbitrary floating primitives.

Exact normalization does not make rounded selection equal to the Real target.
For canonical `seed=0, offset=0`, the sample is
`u=0.7980929017066956` (`0x3f4c4fd1`). The admitted FP32 row
`[u-2^-24, 3*2^-26, 1-u+2^-26]` sums exactly to one. Its exact second prefix is
`u-2^-26`, but the FP32 sum rounds to `u`. With labels `[11,22,33]`, the Real
inverse CDF selects `33` and the actual CPU replay of the selected kernel selects
`22`. `Proofs/Boundary.lean` checks the canonical RNG word, exact input denotations,
normalization and the decisive rounded addition/comparison. A separate zero
sample control confirms that even a zero-weight first label remains the fallback.
These boundaries are reflected in the targets; no new input guard excludes them.

Historical returned-Tensor, allocator, launch, caller-byte and conditional
exact-to-Real results remain optional under `Misc/HostCorrect.lean` and
`Misc/HostFPEquiv.lean`. The public kernel closure excludes those host results
and `Numerics.LocalExact`. Its seven historical primitive-exactness families
are conditional correspondence hypotheses, not outstanding per-task numerical
experiments. RNG, rounding, cast and native scan-plan interpretation remain
semantic model boundaries. No numerical transformation requires validation.

Previously recorded validation: the source/guard CPU suite passed 24 rejections, 17 admitted
cases and 17 complete program executions. It checks the actual JIT body, unchanged
inputs and output guard words, signed seed extremes, offset truncation, multiple
programs, scan/block tails, default labels and the two added boundary controls.
CUDA metadata/submission are mocked; no GPU/compiler or statistical conformance
is claimed. The earlier acceptance compiled 36 task modules and audited its 20-module public
closure. This IO-format pass uses targeted builds of the new adapters, both
public entries and the unchanged Tensor consumer, plus import/source-pin audits.
It adds no GPU run, independent replay or numerical experiment. `Kernels.lean`
continues to import only the DSL.

### Launch-device admission correction (2026-10-19)

Prior candidate SHA-256: `fe754f34e7ae2fb4dc797a8b6b5fb0dc7b49275961fc0888044adaca5ebaf11b`.
The installed Triton JIT selects the current CUDA device and stream; matching
input devices alone did not select scores.device. The source submission now
uses `torch.cuda.device(scores.device)` and restores the caller device. The
one-token clone return remains a no-submission path. Native preparation refers
to this effective owning-device context. CPU controls use logical cuda:1
inputs with ambient cuda:0, verify the launch device, and verify restoration
after both successful submission and an injected submission exception. Kernel
AST, RNG/FP operations, guards and memory effects are unchanged. Prior
acceptance is preserved as evidence; changed hashes require correction.

Optional actual-Tensor applications and input/check binding helpers remain
ordinary theorems in `Misc/TensorBridge.lean`.


The Python candidate `_multinomial_sampling_unrolled_kernel` makes `num_tokens`
a constexpr and executes `tl.static_range(tl.cdiv(num_tokens, BLOCK_N))`, scaling
the same `b_idx` variable by `BLOCK_N` before the unchanged original loop body.
This exposes token blocks to compile-time unrolling and removes runtime loop
control; compile cost and code size can increase, and no speedup is measured.
Every original token block appears once, in the same order. Scores, scans,
maxima, carries, Philox, signed conversions, label clamp and masked store remain.

`Proofs/Unrolled.lean` proves the exact ceiling-count/index correspondence and
full-state execution plus trace-safety equality between the counted and strided
loops, then the complete source prefix/loop/epilogue. It includes zero tokens
and arbitrary legal block widths. `UnrolledPair` supplies a narrow concrete
invocation view over the established `UndefProgramIO`, retaining its actual
memory inputs, three fallback inputs, signed64 seed domain, outputs and frame.
It uses the shared structural interpreter directly, with no result callback.
The pair covers the original complete per-program scope and arbitrary primitive
models, including every retained software scan/max plan; no new grid or host
obligation is introduced. No numerical rule or experiment is needed.
