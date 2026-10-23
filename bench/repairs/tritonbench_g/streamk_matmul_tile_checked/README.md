# streamk_matmul_tile_checked

Independent candidate for the frozen `streamk_matmul` static-domain diagnosis.
The six JIT kernel bodies are unchanged. The wrapper adds two source-static
checks, only when M and N are both nonzero:

- `BLOCK_K >= 16`, including runtime K = 0;
- each of `(BLOCK_M, BLOCK_K)`, `(BLOCK_K, BLOCK_N)` and `(BLOCK_M, BLOCK_N)`
  has at most 2^20 elements.

The existing power-of-two profile, tensor, integer and pointer contracts remain
in force. Empty outputs preserve the parent admission domain and launch no
kernel. These conditions describe the Triton frontend's dot and block types;
they are not GPU resource limits. Grid-product compatibility is deferred under
the project-wide scope rule.

`Contract.lean` is the public input domain; `Proofs/Contract.lean` proves that the
pure checker accepts exactly that domain. Python type checks also reject values
outside the typed Lean API (for example, bool where an integer is required).

`Kernels.lean` contains readable quotations of all six current source bodies.
`SourceSyntax.lean` supports this task's finite syntax subset;
`SourceReader.lean` extracts statement fields, and `Execution.lean` evaluates
those fields. Cast spelling and tuple shape syntax are normalized; ACC_TYPE is
FP32 under the public guard. Unsupported schemas are rejected. The interpreter
has no successful-output callback and does not invoke the old complete kernel
controller. Proof reuse must go through a proved source projection.

The Real target is the independent full-K matrix product. Native semantics must
retain fixed TF32 MMA operations, FP32 accumulation and scratch, ordered FP32
atomic reductions, and the original full-tile FP16 round trip before FP32 store.
No equivalence that rearranges native additions is asserted. Scalar IEEE/cast,
fixed MMA, Tensor storage, compiler and runtime correspondences are model
boundaries, not numerical assumptions awaiting experiments.

The existing accepted source/domain/precision and correctness status is retained.
The Real entry in `Correct.lean` and ordinary native theorem in
`Misc/NativeCorrect.lean` expose shared IO judgments; `FPEquiv.lean` compares
the original source with the constexpr tile-area candidate. The targeted build record is separate from historical acceptance
and independent replay. This pass runs no GPU or numerical experiments and adds
no numerical rewrite assumption or deferred experiment.

## Proof and model boundaries

The canonical executor represents ordered launches and a row-major lane trace.
`Proofs/Schedule.lean` reuses the prior proof for arbitrary first/full write
permutations and reducer CTA interleavings that retain each CTA's program order;
the physical and value executions agree with the canonical source result.
The reducer's unmasked scratch loads, including inactive output lanes, are
covered separately by `Reader.all_scratch_loads`.

Pointer updates use their extracted affine recurrence. `affine_pointer_loop`
proves its equality to repeated source updates; `mac_addressA/B` and
`full_addressA/B` prove actual operand offsets. The public input contract derives
the retained ABI bounds and signed physical operand conformance. FP32 byte
stores and current-value atomics reuse the existing verified memory model.

The frontend is task local. Its decoding and execution projection theorems cover
this complete current quotation; this is not a universal Triton compiler proof.
The native interpretation is the pinned SM80 fixed-MMA profile, not a claim that
all CUDA hardware/compiler configurations have the same instruction graph.
Primitive IEEE/half/MMA/atomic semantics, compiler instruction selection and
Tensor API representation are explicit model boundaries. No operand-finiteness
input condition or empirical primitive-arithmetic validation has been added.

The output/scratch memory bindings are ordinary kernel resource contracts:
actual output Tensor identity/shape/FP32 storage, aligned representable writable
windows and separation from input storage. Input tensors may alias each other
and have nonzero storage offsets. Initial output and scratch contents are
arbitrary and merely bound to the caller's initial memory; no output equation or
successful run occurs in these preconditions. Allocation/dispatch are not part
of the new theorem.

## Evidence and references

The frozen parent diagnostic and its exact installed-compiler excerpts are
[stream-K diagnosis](../../../../.veritile-cache/kernel-evidence/streamk_matmul/diagnosis.md)
and [frontend rules](../../../../.veritile-cache/kernel-evidence/streamk_matmul/frontend-rules.json).
The latter pins Triton 3.7.1 `nvidia.compiler.min_dot_size` (line 19),
`language.semantic.dot` (line 1490), `language.core.block_type` (line 702), and
`_utils.validate_block_shape` (line 62). These justify the static checks.

The candidate's [actual-binder controls](../../../../.veritile-cache/kernel-evidence/streamk_matmul_tile_checked/candidate-binder.json)
record all runtime ABI arguments and offline SM80 artifacts. Six compilations
pass, five illegal static inputs reject before launch, and both empty-output
controls preserve admission. [K=0 controls](../../../../.veritile-cache/kernel-evidence/streamk_matmul_tile_checked/static-precheck.json)
show why BK=8 still fails in the unchanged parent when K=0. These finite controls
are not a universal compiler-success or GPU-execution claim.

[Shared FP32 Tensor laws](../../../../VeriTile/Torch/FP32.lean) cite the Torch
`data_ptr`, `storage_offset`, `is_contiguous` and negative-view APIs.
The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) specifies
memory address/alignment, FP32 atomics, half conversions and
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. The candidate opcode
[inventory](../../../../.veritile-cache/kernel-evidence/streamk_matmul_tile_checked/native-profile.json)
checks the selected MMA, the explicit full-tile FP16 round trip, and FP32 atomic
addition in all six actual-binder samples. Hardware interpretation remains a
model boundary; it is not registered as a numerical experiment.

`Correct.streamk_kernel_real` states that `EventIO.realIO` implements the
independent full-K matrix product. The ordinary `FPEquiv.streamk_kernel_native`
theorem in `Misc/NativeCorrect.lean` states that
`EventIO.nativeIO` implements the independent ordered instruction formula,
observing both each FP32 output word and its four bytes. These use the shared
`Concurrent.EventProgramIO.Implements` judgment. The judgment requires successful
source decoding, completed fixed execution, safety of the same complete trace,
all outputs, and an exact exterior frame. It is not an alias for a private
raw-state correctness proposition.

`EventIO.lean` names immutable A/B channels, arbitrary initial C/S logical arrays,
the full M×N output, and private scratch writable only when stream tiles are
active. Empty M/N and runtime K=0 remain covered. Initial scratch remains an
arbitrary finite logical array even when inactive; it has no new physical
binding requirement. The raw interface retains `Source.Parameters`, rather than
adding the Tensor wrapper's stricter static checks. `EventOriginalDomain` proves
that every original finite state, including arbitrary exterior values, has a
public input binding. `EventCapacity.capacity` provides a concrete valid
capacity witness, so access safety cannot hold merely because no bounds fit.

`ClosedSource.lean` reifies the actual quotation decoder using closed scalar
expressions. The shared executor accepts no store-result callback or task
runner. `SourceReads.lean` also records every active source A/B load and every
producer-enabled scratch load, including lanes with masked C stores/atomics.
`SourceReadsSafety` proves those actual masks/pointers safe. `SourceTape` proves
successful decoding and exact whole-call projection; `EventValues`, `EventFrame`
and `EventImplementation` connect it to the public IO judgments. Atomics still
read the current output and scratch values. The full-tile FP16 round trip and
the fixed K8 MMA/ordered FP32 atomic reference are unchanged.

`Proofs/KernelResults.lean` retains the complete original raw Real/native
statements, including physical output values, every byte and the physical frame.
`Proofs/PhysicalRefinement.lean` connects that same complete shared execution to
both original physical runners. A/B are not newly bound to a physical heap in
the raw interface; physical input correspondence remains in the Tensor
application. No arbitrary address relocation theorem is claimed. The existing
`Proofs/Schedule.lean` continues to cover all allowed first/full permutations
and per-CTA reduction interleavings.

`TensorBridge.kernel_correct` and `TensorBridge.kernel_native` retain the complete
original Tensor statements and source text, including metadata/checks, physical
input correspondence, output observations and frames. They use the same original
source execution and independent reference foundations. The public reader import
closures do not import the Tensor bridge or another reader.


The 2026-10-21 implementation pair adds the complete callable
`streamk_matmul_area.py` and readable `Kernels.area_mac_loop` /
`Kernels.area_reduce_streamk` quotations. The four other complete helper/stage
quotations are shared unchanged. Both scratch slot expressions replace
`program * BLOCK_M * BLOCK_N` (or `pid`) with
`program * (BLOCK_M * BLOCK_N)`, folding the constexpr area before the live
index multiplication. This removes one live scalar multiplication per address
calculation; hardware performance has not been measured.

`FPEquiv.tile_area_equiv` compares the actual full source programs with an
empty experimental rule set. `Proofs/AreaEquivalence.lean` connects the actual
compiled candidates through slot evaluation, source events and the complete
decoded tape, including every active read-only event. `ProgramView.lean`
retains source data and the original shared decoder; its structural relation
requires both successful completed safe runs, equal full final states,
observed words/bytes and exact exterior frames. The original input domain,
empty M/N, runtime K=0, arbitrary initial C/scratch, private active scratch,
K8 MMA order, full-tile FP16 round trip, atomic order, masks and all three
stages remain covered. No floating algebra or numerical assumption is used.
Original independent Real and optional native/Tensor declarations are retained.
The complete Python AST check allows only these two address expressions and
candidate entry names. Its integer checks are not a numerical experiment.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.
