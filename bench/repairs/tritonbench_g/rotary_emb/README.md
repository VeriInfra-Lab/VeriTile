# rotary_emb: checked direct FP32 contract

`rotary_emb_repaired.py` retains the original in-place Q/K API and returns
`None`. The original benchmark is unchanged. No clone, snapshot, packing,
copyback, dtype conversion or new tensor allocation is introduced.

Q and K are ordinary canonical contiguous FP32 CUDA tensors `[T,H,D]`, on the
same device, with positive head counts and width, equal token count and width,
zero storage offset and no lazy flags. Their head counts may differ. Cosine
and sine are canonical contiguous `[T,C]` FP32 tables with independent row
widths. The literal Python conversion `R = int(D * partial_rotary_factor)` is
preserved; the built-in finite int/float factor is in `[0,1]`, and R must be
even with `R <= D`. Each coefficient row contains at least R columns.
All raw FP32 payloads are admitted, including NaNs and infinities.

The explicit allocation contract is narrower than arbitrary Tensor views.
For a launch, all four complete backing ranges are physically separate;
actual data pointers equal their backing starts and are aligned to four bytes.
Logical extents fit their backing allocations. The byte-size, padded signed
64-bit arithmetic and formed unsigned 64-bit pointer checks are executable.
Checking distinct storage handles or start pointers alone would miss overlapping
byte-offset storage objects, so the wrapper compares complete address ranges.
T=0 or R=0 returns before pointer checks and before launching; no unused pointer
is queried. Exact canonical strides are checked only for actual launches.

The source repair fixes independent sine strides, launches enough heads for
both Q and K, and pads/masks rotary pairs. Program IDs and lanes are widened
before address arithmetic. Original arithmetic, 4/8 warp selection, one stage
and default fusion are retained. Every unrotated Q/K column, all coefficient
bytes and all other caller bytes are preserved.

`Kernels.lean` is the complete typed Triton DSL transcription of the repaired
kernel: signed address expressions, coefficient loads, masks, and all four
stores in Q-before-K order. The earlier scalar/vector/grid/host proofs remain
under `Proofs/Model.lean` and the optional `HostReal`/`HostFP` entry points.
`Contract.lean` retains the existing executable admission checks.

The canonical `TensorBridge.rotary_emb_correct` theorem proves successful typed
execution over arbitrary Real payloads, the independent interleaved formula,
and equality of the complete memory map, including every untouched cell.
`TensorBridge.rotary_emb_checked_real_correct` constructs that source memory from
actual Tensor caller reads through an explicit total `Word → ℝ` map. Tensor
laws and checked extents derive the independent logical-element rotation,
successful complete execution, and the full memory frame. The map imposes no
finite-payload guard or FP-to-Real arithmetic law.
`TensorBridge.rotary_emb_fp32_equiv` uses that same source and proves successful
execution of the faithful historical candidate which commutes only the
integer head-block multiplication. The entire resulting state is identical;
all floating outputs also equal the independent selected native graph.
`TensorBridge.rotary_emb_checked_fp32_equiv` binds actual Torch Tensor objects via
shared `TensorLaws`, derives logical reads from their observed data pointers,
and connects typed outputs to the caller-addressed complete-grid result and
its physical byte frame. Store ownership follows from checked geometry.

The selected FP32 graphs are `fma(x0,c0,neg(mul(x1,s0)))` for even columns and
`fma(x1,c1,mul(x0,s1))` for odd columns. Loads, same-format casts, and the
selected local mul/neg/fma semantics are primitive/model boundaries. Neither
public theorem requires numerical conformance obligations or finite payloads.
Sine and cosine are precomputed input tables; this kernel evaluates neither
transcendental. Source-to-interpreter/compiler correspondence and CUDA runtime
behavior are explicit boundaries; the proof does not assert GPU-to-IEEE or
FP-to-Real conformance.

Validation: the complete Lean closure and checked public entries pass targeted
Lean 4.29 builds and standard-base axiom audits. Run `python
bench/repairs/tritonbench_g/rotary_emb/check_contract.py` for eight complete
source/host CPU storage replay paths and 23 prewrite rejections, including physical
partial-byte overlap and all unaligned pointer remainders. The original sine
stride and missing-K-head failures are reproduced. CUDA launch is mocked;
these checks do not establish GPU arithmetic conformance.

`check_interpreter.py` additionally passed ten direct Triton CPU interpreter
launches across this repair and both nopad branches. They cover independent
coefficient strides, partial and padded dimensions, unequal head counts,
GQA leaders, cached padded heads, selected cache slots, metadata, and backing
sentinels. These are actual interpreter launches; no GPU execution is claimed.

The reader entry `Correct.lean` states the shared `ProgramIO.Implements`
judgment. `FPEquiv.rotary_emb_native_equiv` compares the actual original and
retained index-commuted implementations as `original ≡[[]] indexCommuted`.
The shared `ProgramIO.Implementation` view keeps the complete public IO,
input domain, concrete schedules and native observations and reuses the
established `ProgramIO.Equivalent` proof. The former source/native conjunction
is an ordinary theorem in `Misc/NativeCorrect.lean`. `IO.lean` exposes
the actual source, the selected primitive backend, eight input channels and
four in-place output channels, explicit shape/program geometry, both coefficient
strides, and every active mask. Coefficient loads have no head mask: their
windows stay live even when a program has no active Q or K head. No private
scratch or finite-value input restriction is introduced.

The independent logical target gives each interleaved column its own cosine
and sine coefficient. The native target separately states the selected
mul/neg/fma graph; the existing address-only candidate commutes integer
head-block multiplication. Both actual source traces are proved safe, both
executions succeed, and their complete output/frame conclusions are connected
to the public channels. Primitive instructions are explicit model parameters;
there are no transformation assumptions or pending numerical experiments.

`Proofs/IOBinding.lean`, `IOResult.lean`, `SourceSafety.lean`, and `PublicIO.lean`
compose the established typed source proofs. The exact former raw-state
headlines are retained as ordinary helpers in `KernelRealResult.lean` and
`KernelNativeResult.lean`. All four `TensorBridge` theorem statements remain
unchanged, including actual-Tensor and complete-grid physical memory results;
they reuse this same typed execution foundation. The public kernel statements
retain the complete one-invocation scope of the previous raw kernel headlines.

Source-only cleanup places the retained integer-index-commuted candidate in
`Kernels.lean` as a complete ordinary FP32 `triton` quotation beside the original.
`Candidate.optimized` remains a compatibility alias; its prefix helper is only
proof decomposition. Complete function identities preserve the original and the
previous prefix-plus-tail candidate by `rfl`. All precision, casts, signed-int64
addresses, Q/K and coefficient masks, operation order, domains, bindings, channels,
frames and public theorem declarations are retained. This organization adds no
numerical, dtype, native, host or hardware requirement and does not recount the
existing proof completion.
