# Packed FP4 to BF16: kernel proof

The selected source is `fp4_to_bf16_repaired.py` (SHA256
`70f10f7ab112fd50f29511e24c8dac26374d51e3ea8bb6dc769e899bdc6d22be`).
This revision preserves its Python, every input guard and the existing test.
The original benchmark's signed-int32 indexing defect remains separately
recorded; no correctness claim is transferred to that original source.

## Independent target and precision

Each packed uint8 byte produces two E2M1 values, **high nibble first**. For a
nibble with sign s, exponent e and one mantissa bit m, the magnitude is m/2
when e=0 and `(1+m/2)*2^(e-1)` otherwise; s chooses the sign. The last Tensor
dimension doubles. `KernelResult.mathematical` states this Real formula using
logical input bytes, independently of kernel execution or output memory.
`nativeTarget` uses the independent BF16 format table and distinguishes both
zero signs. Every nibble and all 256 packed bytes are admitted.
For example, `0x21` becomes `(1, 0.5)` and `0x80` becomes `(-0, +0)`.

The actual conversion performs uint8 bit manipulation, constructs a signed32
payload, bitcasts to FP32 and then converts to BF16. All E2M1 values are exactly
representable in both formats. Existing kernel-checked `Bits.all_nibbles`,
`integer_payload` and full conversion execution proofs are reused. There are
**zero numerical assumptions or numerical experiment deferrals**.

## Source and public entries

`Kernels.lean` pastes the selected Python helper and calling kernel in full:
`packedToBf16` has the helper's 22 statements (including its docstring), and
`body` has the caller's 13 statements. `jit-correspondence.json` checks both
bodies. Casts to signed64, local `BLOCK_SIZE_OUT: tl.constexpr`, uint8
arithmetic, helper assignments, `tl.where`, bitcast and BF16 conversion remain
in their original order. Input/output dtypes and the wrapper's format constants
are supplied by `sourceSignature` and its invocation outside the pasted body.
The helper's unannotated parameters remain unannotated.

`Kernels.source` selects this actual caller. `Proofs/SourceLoad.lean`,
`SourceArithmetic.lean` and `SourceContract.lean` prove its complete execution
and memory result, including every helper argument/result copy. `SourceSafety`
checks its actual masked loads and stores. The historical conversion's
independent E2M1/BF16 tables and finite signed64 address certificates are reused.
`Launch.program` binds the actual count/program-ID ABI;
`Launch.complete_transfer` connects complete schedules through successful
execution and equality of the entire memory. The new helper registers may
differ from the historical inlined implementation, so no equality of full
register states or old/new source ASTs is asserted.

- `TensorBridge.fp4_to_bf16_kernel_real` proves complete successful source execution
  and the independent Real formula at every actual output Tensor coordinate.
- `TensorBridge.fp4_to_bf16_kernel_native` proves the exact corresponding BF16 word
  against the independent format graph, including negative zero.
- Both entries include the complete tagged-memory frame, exterior physical
  byte frame and preservation of the complete input backing allocation.
- `TensorBridge.native_lane_refinement` states the exact padded native masks using
  a bound derived from the actual output allocation.
- `TensorBridge.python_checks_iff_preconditions` retains exact guard equivalence.

These are kernel results. Historical wrapper/protocol statements remain in
`Proofs/HistoricalCorrect.lean` and `Proofs/HistoricalFPEquiv.lean`; public
entries do not invoke `observe`, `runChecked` or allocation/error protocols.
The reused original `Host`/`HostSchedule` helper files describe kernel grids,
register reset and shared memory, not a new whole-wrapper proof obligation.

## Actual Tensor and memory binding

`KernelBinding.PackedTensor` describes the actual input object, metadata,
backing storage identity, byte capacity, storage offset, physical logical
indices and observed data pointer. The admitted no-negative and contiguity
checks derive direct packed-byte reads. `Bf16Tensor` describes the actual
output with two-byte element addressing, dtype/layout, shape, physical
indices and pointer laws. `Binding` connects both objects to the same launch,
doubled last dimension, device and fresh backing allocations. Nonzero input
and output storage offsets are supported; empty windows impose no unused
pointer/offset bound.

`Binding.storage` derives the existing finite-address certificate from those
actual allocations. BF16 constructor alignment and signed host storage-size
limits are ordinary output-allocation facts, not added user assertions.
`Bytes.represent` represents every initial heap, including arbitrary
`torch.empty` output bits, and `initial_realization` proves its roundtrip.
`Bytes.physical_store` already proves that each active source BF16 store
commutes with the two-byte representation. The final result reads the output
through its actual Tensor coordinates and this proved memory connection.
Memory-model/API rationale is in `documents/GpuMemoryModel.md` and the shared
`VeriTile/Torch` Tensor bindings; Torch, allocator and compiler implementations
are platform boundaries rather than new task-local proof projects.

The unchanged guards require Tensor, strided layout, uint8, nonzero rank,
CUDA, contiguity and no lazy negative flag. There is no count cutoff or payload
restriction in the guards. Successful allocation supplies finite storage;
`gridSafe` is the existing launch representability condition. Unsupported
launches may fail in the runtime and are not claimed to return a correct
Tensor. Grid-product overflow/segmented launch work is not expanded here.

## Validation

`bench/tests/test_packed_fp4_body.py` checks the two complete bodies and
rejects mutations of casts, bitcasts, nibble order, operation order, masks,
helper parameter annotations and external invocation metadata. Lean checks
cover the actual execution, full IO, independent Real target, exact BF16 word
and complete schedule comparison. Their replay and axiom-footprint results are
recorded with the source snapshot; a body check alone is not a proof result.

The unchanged `test_revision.py` retains its historical CPU interpreter and
mock-host checks. This migration does not report a new interpreter/GPU run:
Torch and Triton are absent in the current verification environment. The
standalone trust comparator is also unavailable because host isolation cannot
be established (Landlock ABI 4 versus required 5; no systemd user isolation).
These unavailable checks are recorded separately from Lean results.

`Misc/TensorBridge.lean` contains the optional Tensor/check correspondence theorems;
`Correct.lean` and `FPEquiv.lean` retain the primary kernel specifications.

The main `Correct.fp4_to_bf16_real` entry uses shared `ProgramIO.ImplementsOn`.
`FPEquiv.fp4_to_bf16_schedule_equiv` compares two actual complete CTA orders as
`ScheduleIO.original h a ≡[[]] ScheduleIO.candidate h b`. This retains the
schedule-reordering comparison already proved in
`Proofs/HistoricalFPEquiv.complete_fp_equivalence`; it does not introduce a new
conversion kernel. `ScheduleIO.Equivalent` requires both source traces to be
safe, both full schedules to succeed, every BF16 output to agree and both
exterior frames to hold on the same `ByteDomain`. `Proofs/ScheduleEquivalence.lean`
reuses the existing complete IO implementation proof for each order. The old
`FPEquiv.fp4_to_bf16_native` declaration is an ordinary auxiliary theorem in
`Misc/NativeCorrect.lean`, imported explicitly by optional consumers.

`IO.lean` exposes the complete actual finite-ABI
launch schedule, packed uint8 input window, BF16 output window, source precision,
register reset and grid metadata. `ByteDomain` is exactly the unsigned-byte
range, covering all 256 input values. The existing finite storage, region
separation and grid-representability conditions remain explicit.

The mathematical target independently decodes each high/low nibble as E2M1;
its observation is `Option Real`, and the proved output is `some` of that
formula. Thus finiteness remains a conclusion, with no default-value fallback.
The native target separately retains the exact BF16 bits, including both zero
signs. Schedule reordering changes no arithmetic or numerical assumptions.

`Proofs/SourceSafety.lean` checks both actual masks and all intervening pure
source operations. `LogicalResult.lean` reuses the established raw source-memory
result with logical byte reads; `LogicalGrid.lean` retains the entire arbitrary
program permutation, and `PublicIO.lean` connects safety, successful execution,
outputs and the full exterior cell frame. Former raw kernel statements and the
native-mask consequence are ordinary helpers in `KernelRealResult.lean` and
`KernelNativeResult.lean`. The complete `Misc/TensorBridge.lean` source and theorem
statements are unchanged; they reuse the same source execution foundation and
existing physical-store/Tensor binding proofs.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
