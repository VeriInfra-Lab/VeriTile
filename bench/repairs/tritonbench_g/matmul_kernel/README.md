# matmul_kernel: complete selected kernel

Current FP entry: `FPEquiv.source_pair_equiv` compares the original complete grid with the remaining-K mask candidate as `original ≡[[]] remainingK`. The candidate and full proof are described below. `Misc/NativeCorrect.lean` retains the independent ordered native recurrence as an auxiliary theorem; no experimental numerical rule is needed for the mask transformation.

The selected source is `matmul_kernel_repaired.py`, SHA-256
`10892032b7d88b46f0bc43e0bc94176c704c6e7d7b3cb6f92cf7229ca4926a55`.
Its independent mathematical target is
`C[r,n] = Σ k<K, A[r,k] * B[k,n]` on the initial logical input tensors.
The Real theorem idealizes the floating operations and conversions; it is not
an assertion that native TF32/FP16 arithmetic equals the Real result.

`Kernels.lean` contains both unchanged Python function bodies directly in
`triton { ... }`. Input dtypes, the nine runtime integer arguments, the three
constexpr block sizes and the existing read/write ports are supplied outside
the quotations. The original `matmul_kernel_repaired.py` and existing candidate
`matmul_kernel_remaining_k.py` are unchanged. This retains both program ids,
modulo M/N loads, all six strides, K masks, the live three-argument dot, pointer
updates, FP16 narrowing, and masked output stores. The frontend supplies the
actual implicit FP32 store widening; no extra cast is written in the pasted
source and no floating conversion is assumed to be an identity.

`Proofs/SourcePreservation.lean` proves complete kernel identities, including
all statements and interface ports, with the previous typed original/candidate
for every `Config` and symbolic region. This preserves arbitrary primitive
models, successful and failing execution, complete final states and every
trace/memory observation. `HostModel.lean` retains the unchanged pure host,
configuration and instruction model; its previous location is a thin
compatibility import. Actual source no longer imports its proof directory.

`Correct.lean` displays `realIO`: the actual complete 2D source schedule,
M×K and K×N input channels, M×N output channel, original stride addresses,
and the independent full-K matrix product. Its sole specification uses the
shared `ProgramIO.Implements` judgment. `Misc/NativeCorrect.lean` displays the matching
`nativeIO` and proves the source-to-native ordered K8 coordinate target through
the same shared judgment. The native result remains a primitive-model
interpretation; it does not introduce a second optimized kernel or a new
numerical equality assumption.

Both judgments conclude safety of every actual source load/store, successful
execution of every program, every logical output, and the entire exterior
cell frame. The program schedule updates only axes 0 and 1, matching the prior
runner exactly. The `.real` channel tag is the Structural floating carrier;
FP32/TF32 operations and both final conversions remain in the source/backend.
No raw `State`, runner or existential final memory appears in the main theorem
statements. The old coordinate theorems remain in `KernelReal.launch_correct`
and `KernelNative.launch_correct`.

`Proofs/KernelInputs.lean` connects only active logical matrix coordinates to
the existing complete source results. `Proofs/KernelSafety.lean` checks the
actual typed prelude, loop loads, every actual successor context and the masked
final store with shared `StructuralSafety`. `Proofs/PublicIO.lean` equates the
shared launch schedule with the original full-grid runner and assembles the
shared typed IO. Declared channel bounds alone do not substitute for this
source trace proof.

`TensorBridge.correct` and `TensorBridge.fp_equiv` retain the complete prior
Tensor applications. `KernelEntries.real_inputs` and `native_inputs` identify
the raw initial reads with logical Tensor inputs; `c_address` identifies the
output location. `KernelEntries` applies the same `KernelReal.launch_correct`
and `KernelNative.launch_correct` proof foundation reused by the main IO entries.
`KernelResult` and `KernelReplay` retain actual Tensor readback, all four
physical output bytes, per-store replay, and the entire exterior byte frame.
Historical host proofs remain in `Misc/Legacy/HostCorrect.lean`,
`Misc/Legacy/HostFPEquiv.lean` and `README.host.md`.

## Prepared inputs and source bounds

The existing input checker admits exact rank-two, resolved FP32 CUDA matrices
A(M,K), B(K,N), C(M,N), on the same device, with positive strides and separated
rows or columns. Nonzero storage offsets and transposed/sliced layouts are
retained. C's nonempty backing storage must be independent from nonempty
inputs; read-only A/B aliases are allowed. Every power-of-two BM/BN/BK at least
16 is described, subject to ordinary successful compiler/resource legality.
The validation fixtures impose no tile-size cap on the theorem.

`KernelTensor.Prepared` connects that checked metadata to actual shared
Tensor API laws: dtype, rank/shape, device, storage identity, queried pointer,
physical index, strides and remaining storage capacity. Ordinary separation
of distinct physical allocations is recorded explicitly. These are input/API
facts, not kernel-result assumptions. `KernelBounds` links the shared DSL's
actual offset expressions to signed32 bounds and active allocation bounds,
including pointers formed after the last K iteration. M=0 or N=0 launches no
program. K=0 with nonempty output executes the epilogue and stores cast zero.

## Floating model and explicit boundary

FP32 inputs feed the live FP32 dot accumulator with Triton's default TF32
mode. The observed SM90 lowering uses fixed-K8 `mma.sync` or `wgmma` TF32
instructions. `KernelNative.Instructions.mma8` is the fixed instruction-lane
model boundary, covering its TF32 operand interpretation, internal reduction,
rounding, subnormals, signed zeros and NaNs. It consumes the original FP32
register words: no fictitious source `cvt.tf32` is introduced. Compiler opcode
selection, lane mapping, K8 grouping and instruction implementation conformance
are explicit external model boundaries.

The complete source backend expands every block into the ordered K8 primitive
calls with the live carry. The independent coordinate target spells out that
same instruction sequence using initial matrix coordinates. It retains all
padded native operations; no floating zero-product identity or reassociation
is assumed. There is no variable-size dot oracle, whole-output equality port,
finite-value restriction, or deferred source/control-flow proof.

Final conversions are concrete: direct decoded FP32-to-binary16 RNE rounding
with gradual underflow and canonical NaN, followed by exact bit-format
binary16-to-FP32 widening. Lean checks boundary vectors for ties, subnormals,
overflow and signed zero. The declared conversion profile's realization by the
compiler/GPU is an explicit boundary; conversion callbacks and assumed cast
idempotence have been removed from the public theorem.

## Evidence and proof organization

- `KernelReference`, `KernelExecution`, `KernelComplete`: full shared source,
  dynamic K-loop invariant, masked tile results and frame.
- `KernelGrid`, `KernelScalar`: complete grid coverage and ordered per-lane
  recurrence, independent of arithmetic identities.
- `KernelReal`, `KernelNative`: independent mathematical/native targets.
- `KernelTensor`, `KernelStorage`, `KernelBounds`, `KernelEntries`,
  `KernelResult`, `KernelReplay`: actual Tensor readback, exact byte interpretation, source
  bounds and public kernel entries.

The historical delivery recorded direct execution of the exact undecorated
Python AST on 52 CPU dataflow/access configurations, 90 programs and 12
empty cases, with tail blocks, K=0, transposes, positive strided views, nonzero
offsets and read-only input aliases. The original hard-coded/unmasked source
reproduces an active-load out-of-bounds failure. These CPU checks use exactly
representable integer inputs and check the explicit FP16/FP32 storage casts;
they do not establish GPU TF32 numerical equivalence.

Six actual installed Triton runtime binder/packer cases compile offline for
SM90, including scalar dimensions 1 and unit strides specialized to constexpr
1, K=0, tails, transposed views and a larger tile. The resulting TTIR/TTGIR/PTX
are pinned. This is actual binder specialization plus offline compilation,
not a manual all-i64 signature, a CUDA launch or a GPU numerical experiment.

Optional proof layout (2026-10-21): `Misc/NativeCorrect.lean`, `Misc/TensorBridge.lean`, `Misc/Whole.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Main declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.
Historical wrapper readers remain in `Misc/Legacy/HostCorrect.lean` and `Misc/Legacy/HostFPEquiv.lean`. Their host results use `Legacy.HostCorrect.correct` and `Legacy.HostFPEquiv.fp_equiv`, preserving the current `MatmulKernel.correct` and `MatmulKernel.fp_equiv` names. All task `Misc/` entries, including `CandidateCorrect.lean`, are checked together with the current primary entries; see the [namespace-fix validation](../../../review-checkpoints/2026-10-22/misc-import-fix/README.md).
The unchanged `Public.Admitted` structure is shared through `Proofs/Admitted.lean`; `Misc/Whole.lean` retains the optional host claims and theorem applications.

`FPEquiv.source_pair_equiv` compares the complete selected source with
`matmul_kernel_remaining_k.py`: both load masks change from
`offs_k + k * BLOCK_SIZE_K < K` to `offs_k < K - k * BLOCK_SIZE_K`.
This strength reduction moves the block offset out of the per-lane vector
addition; no measured speedup is claimed. Both sources have complete ordinary
quotes in `Kernels.lean`. All pointers, ordered live dots, FP32 loads, FP16
narrowing, FP32 storage widening, validation and wrapper branches are unchanged.

The main theorem uses the shared complete `ProgramIO` syntax. Under the original
logical input bindings and IO bounds, both actual full two-dimensional grids
are safe and succeed, complete observed output cells agree and both exterior
memory frames hold. `Proofs/Pair.lean` lifts an exact natural tail-predicate
identity through both loads, the whole K loop and every invocation. Missing
register failures are preserved too. `Proofs/PairIO.lean` supplies actual
success, safety and frames from the retained full source proofs. No floating
rewrite, experimental numerical assumption or finite-word restriction is used.

The existing signed32 input guard also bounds the new remaining-length
subtraction: in reached iterations `k * BK < K`, so it is nonnegative and
bounded by `K`. Empty output launches and K=0 epilogue behavior are retained.
`Misc/CandidateCorrect.lean` additionally transports the actual ordered physical
byte replay through the complete candidate source. Independent Real/native and
Tensor entries retain their full domains and statements.


The current correspondence manifest checks both complete function bodies,
all source parameter names, dtype metadata and call bindings. Mutation checks
reject splitting the live dot, changing ceil-division loop coverage, deleting
the FP16 narrowing, inserting a source store cast or weakening the candidate
K-tail mask. The original and candidate complete AST identities and all 47
task modules were replayed after migration. Historical CPU/compiler fixtures
are historical evidence; this run adds no GPU or new numerical experiment.
