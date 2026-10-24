# INT4 GEMM: shared source and kernel correctness

The canonical checked source is `int4_matmul_repaired.py`, SHA256
`00132e33a59c145f856047e40fcce4a096105815c080ea0d324553ac6cf1a8ae`.
This delivery does not change that Python program, its admitted inputs, or its
36 retained autotune configuration occurrences. Historical original-benchmark
source defects and their separate repair history remain recorded separately.

The independent GEMM formula is
`sum_k A[m,k] * (Q[k,n] - Z[k/group,n]) * scale[k/group,n]`, with unsigned
nibbles Q/Z extracted from signed INT32 containers. A, scales and output use
FP32. The source dequantization, FP32 live accumulator, default TF32 dot
realization, and SPLIT_K=1 store / SPLIT_K=2 atomic-add branches are preserved.
All four loads use the same K mask. Masked operands and padded K8 instructions
are retained even for nonfinite FP32 inputs. Real semantics interpret the
independent mathematical formula; they are not exact native arithmetic.

The unchanged contract requires positive M/K/N, K and N divisible by eight,
and a positive integer group size dividing K. The original tensors are ordinary
rank-two strided CUDA tensors, on one device, with independent allocations,
zero storage offset, resolved negative/conjugate views, and no autograd.
A is canonical contiguous FP32. Packed Q/Z and FP32 scales accept canonical
strides or exactly the quantizer-returned singleton transpose-contiguous
strides. The output is canonical independent FP32. Pointers have their natural
element alignment. Existing signed32 formed-index and pre-division packed
pointer-advance guards are retained; no additional divisible-tile-K, finite-A,
finite-scale, or selected-specialization restriction has been introduced.

## Source and proof organization

- `Kernels.lean` owns the single readable `KernelSource.kernel` quotation,
  with explicit FP32/INT32 regions, actual strides, all masks and both output
  branches. The retained scalar model and generated Python AST fingerprint
  support existing proofs and the source-identity harness; that fingerprint is
  not a second executable Lean kernel.
- `Contract.lean` and `Proofs/Contract.lean` retain the exact dynamic checks and
  their admission equivalences. `TensorBridge.python_checks_iff_preconditions`
  exposes the GEMM guard correspondence.
- `Proofs/Kernel/Execution.lean` interprets the observed local
  `accumulator += dot(a,b)` fusion using actual source operands and the live
  carry. It expands a tile into fixed-K8 primitive components. Everything else
  uses shared Structural semantics; it does not identify unfused addition with
  fused accumulation or accept a complete-dot/result oracle.
- `Reference`, `SourceProof` and `Complete` prove successful execution of the
  complete quotation and its exact masked store/atomic memory effect.
  `OperandBridge` derives correspondence to the existing matrix model from
  the real guarded strides and signed packed input words.
- `GridExecution` obtains each program's accumulator by executing its quoted
  source prefix. `source_program_events` connects its lane commits to execution
  of the full source epilogue. `GridProof` permits every complete lane
  serialization and proves the dense Real result and exterior memory frame.
  Read-only inputs are snapshotted; output independence ensures their values
  remain unchanged as commits interleave. Split atomics read the destination
  at their commit point.
- `TensorResults` connects original Tensor allocations/views to these source
  inputs, final output words/bytes, data pointers and storage frames.
  `TensorBridge.matmul_kernel_correct` and `TensorBridge.matmul_kernel_native`
  retain those full Tensor applications. The wrapper supplies checked metadata
  and resets output before the kernel; allocation and wrapper return identity
  are not new proof gates.

The existing `quantize_int4` and `unpack_int4` helper proofs are reused. Quantize
retains FP32 group extrema clipped around zero, `(max-min)/15`, nearest-even
round/clamp and eight-nibble INT32 packing. Its original finite-weight and finite
strictly-positive computed-scale guards remain necessary. Unpack keeps its
explicit FP16 intermediate/output and FP32 promoted zero/scale arithmetic.
The retained actual helper Tensor/byte/frame results and older wrapper results
live in `Misc/HistoricalCorrect.lean`, `Misc/HistoricalFPEquiv.lean`, and their
existing helper modules. Their preservation does not add allocator, Torch,
autotuner or runtime implementation work to the kernel acceptance scope.

## Semantic boundaries and numerical experiments

`Native.Instructions` is a bounded primitive semantics interface: scalar integer
conversion, FP32 multiplication/atomic addition, and a fixed eight-operand
ISA component with its incoming FP32 carry. Instruction semantics, physical
register-fragment mapping and Python/Triton/Lean correspondence are implementation
boundaries, not numerical-experiment TODOs. Historical predicates named
`Correspondence` remain optional transport lemmas; the new native endpoint does
not require `actual = reference` primitive hypotheses or any finite-input law.

This delivery introduces **no numerical assumptions or deferred experiments**.
It makes no quantitative claim that TF32 native results approximate the Real
formula within a given tolerance. Such a claim would require a separately stated
error target and applicable data/domain conditions. GPU conformance, compiler
verification, primitive semantics and missing program proofs cannot be relabeled
as numerical assumptions.

The shared load/store, mask, byte-address and instruction-alignment references
are in [GpuMemoryModel.md](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts).
[torch.Tensor.data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html)
and [torch.Tensor.stride](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html)
provide the Tensor API observations. The task's retained `Direct.TensorLaws`
connects those observations to ordinary allocation bases/capacity;
`Physical.decode_encode`, `encode_decode`, and `load_store` justify four-byte
word serialization and its frame. These source/cell/byte theorems do not assert
that every backend emits scalar memory instructions or formally certify emitted
PTX. Access-size alignment and compiler fragment mapping remain scoped to the
applicable implementation profile. Triton runtime grid-product overflow and
segmented-launch compatibility remain deferred outside the kernel task.

## Validation

`check_contract.py` executes the actual canonical Python AST over CPU backing
storage with mocked CUDA allocation/launch. It covers all 36 configuration
occurrences, both helper APIs, singleton layouts, K tails, special FP32 words,
supplied output reset, and admission rejections. These checks are not GPU tests.

`check_kernel_source.py` imports the actual decorated canonical JIT kernel,
obtains real wrapper arguments and uses Triton's actual
`create_function_from_signature` binder followed by `JITFunction._pack_args`.
It asserts kernel parameter order, every scalar/config value, every packed
constexpr, and the compilation options before compiling. The five authoritative
offline SM90 compilation profiles cover canonical and quantizer-returned
singleton layouts, scalar-one M/stride/group specialization, a group-one K tail,
and both store and split-atomic branches. Three valid round-2 profiles are retained;
the two packed-singleton profiles were replaced in round 3 using outputs from the
actual `quantize_int4` AST, with `(1, 1)` qweight/scale strides and
`stride_bk`/`stride_bsk` specialized as `constexpr=1`. TTIR/TTGIR/LLIR/PTX remain
under `.veritile-cache/kernel-evidence/int4_matmul/exact-specialization-round2/`
and `exact-specialization-round3/`. Each directory has a `source-check.json`;
the task delivery's `compiler-profiles-round3.json` identifies the five selected
profiles and their exact source/config/artifact provenance.
Config 0/9 and the config-0 tail use `wgmma.m64n128k8`; config 5/14 use
`mma.m16n8k8`. Both instruction families already occur in `Native.ISA`. A separate
static audit records TTIR/TTGIR live loop carries, the PTX accumulator operands,
enabled WGMMA carry gates, fence/commit/wait ordering, and the selected epilogues.
No Lean model change or new numerical assumption was needed.

The original five compilation profiles are **not authoritative**: the former
manual argument dictionary put configuration constexprs in the wrong order, so
all five actually compiled incorrect tiles with `SPLIT_K=8`. Their artifacts
remain in `exact-specialization/`, and the rejected delivery, checker and
independent review are preserved under the task delivery's `round1/` directory.
Round 2 corrected parameter binding, but its two packed fixtures retained
canonical row stride 8 because `.T.contiguous().T` did not materialize the
intended singleton layout. That rejected delivery and review are preserved in
`round2/`. Only those two profiles were recompiled in round 3. The three unaffected
round-2 profiles and existing Lean/CPU results remain applicable.

These finite profiles are compiler evidence, not universal compilation or
runtime-success guarantees. GPU execution and independent native numerical
replay have not been performed. Targeted Lean build and source/axiom audit
receipts are recorded in the task delivery; independent acceptance is separate.

## Public IO and retained Tensor applications

`Correct.matmul_grid_real` uses the shared `PerAddressProgramIO.ImplementsOnCells`
judgment for the independent dequantized matrix product. The ordinary theorem
`FPEquiv.matmul_grid_native` in `Misc/NativeCorrect.lean` uses the same source-bound
judgment for the independent native instruction graph.
These public statements contain shape/stride geometry, tile selection, primitive
models and the original complete lane permutation; raw execution/state details live
in `Proofs/`.

`IO.lean` declares five finite typed channels: A, packed weights, scales, packed
zeros and the zero-initialized output. The output channel aliases the last input.
The quoted source retains FP32 A/scale/output and INT32 packed storage; `.real` in
the Structural memory interface is a payload carrier, not a change of source dtype.
The only value-domain conditions are signed INT32 representation and the existing
output reset. Native floating words have no finiteness restriction.

The IO catalog contains each complete `(pid, split)` CTA once. Each output's
selected order is the projection of the original lane permutation; different
outputs may retain different atomic orders. The proof establishes all-CTA safe
successful execution, source locality, preservation by omitted CTAs, source-derived
cell traces, every output and the exact exterior raw-cell frame. `Proofs/Kernel/AccessSafety.lean`
checks the actual source prelude, K loop, four masked reads and store/atomic branches.
`Proofs/Kernel/SourceGraph.lean` and its companions preserve the existing fixed-K8
live accumulator using the standard Structural executor.

`Formula.lean` defines the independent finite dense formula and native graph,
including padding, primitive conversions/multiplication, live K8 carry and atomic
order. `Proofs/IOReference.lean` proves the per-address projection equals the prior
native reference and that its Real interpretation equals the dense formula.
`Proofs/RawIO.lean` constructs finite input/capacity/output bindings for every old
arbitrary scalar state and register scratch. Singleton packed strides remain
admitted by `Raw.Geometry`; no additional canonical-stride restriction is imposed.

The exact prior raw theorem statements remain ordinary theorems in
`Misc/RawCorrect.lean` and `Misc/RawNative.lean`. Complete Tensor results and
Python check correspondence remain unchanged in `Misc/TensorBridge.lean`, using the
same established source-execution foundation. Primary entries never import the
Tensor bridge, and supporting proofs do not import public reader entries.

This format change adds no Python/kernel variant, numerical transformation law,
primitive conformance assumption or numerical/GPU experiment. Independent replay
and GPU validation retain their previously reported status.

## Implementation-pair review (2026-10-21)

The source-to-native results above are retained as ordinary theorems in
`Misc/NativeCorrect.lean`; their full names and statements remain unchanged.
`FPEquiv.fold_stride_equiv` now compares the complete original source with
`int4_matmul_fold_stride.py`. The candidate precombines constexpr `BK*S` in
three dynamic `k*BK*S` scale-address and K-mask expressions. This removes one
dynamic integer multiplication from each expression; no speedup is claimed.
Both sources are complete ordinary `triton` quotations in `Kernels.lean`.
`check_fold_stride.py` checks their exact complete delta, including unchanged
Python host helpers, resets, wrappers and all36 autotune configurations.

`Proofs/FoldStride.lean` proves exact complete state equality and access-safety
equivalence under every Structural algebra, including failure behavior. Only
natural-number multiplication associativity is used. `Proofs/FoldStrideIO.lean`
transports the complete original per-address source theorem: every CTA,
source-locality/coverage, each independent split order, both actual cell traces,
typed output cells and both exterior frames remain. The main pair has an empty
experimental rule set and retains all FP32 words and original zero-output domain.

The historical separate-dot versus carry-seeded-dot counterexample remains;
the current optimization preserves the original fixed-K8 floating graph exactly.

The independent `Correct.lean` mathematical target and its complete existing
input domain, source precision, launch coverage and frame results are retained.
No missing program connection is treated as a numerical assumption.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.
