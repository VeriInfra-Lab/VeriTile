# ReLU on signed strided buffers, FP32

The kernel computes the independent formula `out[i] = max(0, input[i])` in
Real semantics. Its FP32 instruction semantics use strict ordered comparison:
positive nonzero words are retained; negative words, both zero signs and all
NaNs produce positive zero. All FP32 payloads are admitted.

The checked Python source adds assertions before the existing wrapper body.
The original source, `StridedBuffer`, tile/warp heuristics, scalar helper and
both Python kernel branches are unchanged. The original Python SHA-256 is
`7d0b0a748a9fb80585ad145eb9b6d1ad3810a31a9d6de63d3df7637a728e7b08`.
There is no input copy or normalization. Wrapper execution and returned-object
identity are optional historical results, not public acceptance requirements.

## Input contract

`relu_strided_buffer_repaired.py` and `Contract.lean` require the same domain:

- Tensor or StridedBuffer with a Tensor base; view/base are FP32 on the same
  CUDA device, with strided base storage and no unresolved negative view bit.
  Input and output share a device.
- Matching rank-one shapes, integer `0 < n < 2**31`, and signed-int64 integer
  strides. Both stride signs, arbitrary bounded storage offsets, and zero
  input stride remain supported. Output stride must be nonzero.
- Four-byte-aligned backing storage, an actual pointer selecting an integral
  FP32 cell, and both endpoint cells inside the backing storage capacity.
- Disjoint accessed footprints. Separated endpoint spans take the constant-time
  path; overlapping spans use exact arithmetic-progression membership.
  Disjoint interleaved views of the same storage are admitted.

The overlap fallback uses constant auxiliary space and at most `n` membership
checks. The domain has not been narrowed to contiguous or separate-storage
operands. Empty input was already rejected by this checked revision; the
original wrapper's empty-input tile-division failure is retained separately.
No finiteness assertion is added, including on inactive lanes.

`Proofs/Contract.lean` proves check success iff this contract, including exact
signed-footprint membership and the endpoint-to-all-cells bound.
`Proofs/Memory.lean` connects metadata to signed cell addresses, actual checked
pointers, and region/allocation provenance. `AllocationAPI` supplies ordinary
storage/provenance and metadata facts; it supplies no output or execution fact.
The finite signed index and byte-address lemmas in the original task's
`Proofs/WholeAccess.lean` remain available and reused by the retained optional
host results. Shared memory-model rationale and load/store references are in
[`documents/GpuMemoryModel.md`](../../../../documents/GpuMemoryModel.md).

## Canonical source and public proofs

`Kernels.lean` contains four complete ordinary `triton` quotations: the two
original branches and their existing candidates, which constant-fold the unused
integer `num_tiles0` calculation. No floating operation is changed. The source
readability cleanup removes the whole-source `reluSignedSource%` macro and its
statement splicing; the campaign preservation check proves all four complete
functions and both dispatch functions equal to their old expansions by `rfl`.
Names, programs, input domains, precisions and existing proof statements remain
unchanged. This adds no candidate or acceptance requirement.

The DSL explicitly lowers each rank-one block pointer to its signed cell
formula `base + logical_index * stride`, with the original logical boundary
mask. Conversion to a natural memory index occurs only after the complete
signed sum. `DTYPE` is fixed FP32; the two pointer-element casts and the zero
conversion are explicit. The source preserves arbitrary inactive load values.
`Proofs/Source.lean` extracts the tile from this source and proves execution
and typed-memory effects using the established signed tile result.
`Proofs/Kernel.lean::signed_source_lowering` proves successful execution and
complete memory/undef equality with the historical signed block-pointer
source. Their private pointer registers may differ.

`IO.lean` exposes finite FP32 input/output channels, signed address windows,
the original capped launch metadata and both actual source variants. Its
mathematical and native word formulas are independent of execution. Each
public entry now has one specification using the appropriate shared judgment:

- `Correct.lean::relu_strided_buffer_kernel_correctness` uses
  `ProgramIO.Implements` for the complete grid and independent `max(0,x)`
  formula, with actual source safety, successful execution and the complete
  exterior frame. Disjoint input footprints are consequently preserved.
- `FPEquiv.lean::relu_strided_buffer_folded_equiv` uses standard `≡[[]]`
  with `FP.BackendMemory`: both complete FP32 sources succeed and agree on
  every typed memory cell for arbitrary source parameters and initial state.
  Empty extents, zero tiles, either branch and arbitrary signed views retain
  the original equivalence scope. The source-level equivalence does not need
  the valid-view premises used to establish the mathematical IO application.
- `Proofs/NativeGridIO.lean::relu_strided_buffer_grid_equiv` retains the
  full-grid native `ProgramIO.Implements` and `ProgramIO.Equivalent` result,
  including both actual source traces' safety, independent FP32 classification
  and exact exterior frame under the original signed-view conditions.

`Proofs/TileSafety.lean` checks actual signed load/store addresses;
`Proofs/SourceSafety.lean` connects them to the prologue and every grid-stride
loop predecessor. `Proofs/PublicIO.lean` reuses
the existing complete source-grid result and scalar classification to assemble
the IO judgments. Exact former raw statements, including the arbitrary-parameter
single-kernel folded equivalence, remain ordinary theorems in
`Proofs/KernelRealResult.lean` and `Proofs/KernelNativeResult.lean`.
`Misc/TensorBridge.lean` retains all existing Tensor/check statements and proofs;
only its two imports follow the relocated helper files.
`Proofs/SourceEquivalence.lean` reuses the exact original successful source
equivalence. The shared backend is explicit because the typed FP32 comparison
is provided by that instruction model; the generic profiled backend would not
guarantee that comparison is supported.

The existing 512-element tile and 65536-CTA cap are retained. Kernel coverage,
signed readback and memory-frame proofs are reused. The public entries contain
no wrapper execution, allocation, launch callback or returned-Tensor claim.
`Proofs/HostReal.lean` and `Proofs/HostFP32.lean` retain the former checked-host
entries as optional results; no helper imports either public entry.

## Numerical and validation status

There are **no numerical transformation assumptions or deferred experiments**.
The candidate changes only discrete integer computation. FP32 zero construction,
strict comparison and same-format casts use explicit software instruction
semantics; the former zero-literal interface is discharged by
`WholeFP32.zero_literal_exact`, not assigned to a future experiment.

The previous source/contract validation is retained as historical evidence:
its Python harness passed 12 raw-word output/frame
cases, one direct loop-AST case, 25 rejection cases, 23,814 independently
computed signed-footprint cases, and three metadata-only launch boundaries.
The cases cover both reverse directions, zero input stride, interleaved shared
storage, offsets, tail masks, singleton extreme signed64 strides, NaNs,
infinities, signed zeros and subnormals. CUDA guard/launch are mocked; no GPU
execution or native compiler/hardware conformance is claimed. No independent
numerical replay is needed for this exact discrete candidate.

The 2026-10-20 format pass runs targeted Lean builds, statement-surface,
formula-independence and standard-base axiom audits. Build receipts/logs are
in `.veritile-cache/public-io-format-20261020/` under the batch04 ReLU entries.
It does not rerun Python numerical checks or GPU experiments.

`Misc/TensorBridge.lean` holds the optional Tensor/check correspondence helpers;
`Correct.lean` and `FPEquiv.lean` retain the mathematical and native kernel entries.


The mathematical main interface accepts raw signed views (region, base and stride), extent/capacities, active bounds and footprint separation. The source-equivalence main interface accepts arbitrary source parameters without those view conditions. `Proofs/RawKernel.lean` derives their canonical launch from these fields. The full original checked view/allocation applications are retained in `Misc/TensorBridge.lean` under `TensorBridge`, using the same proved grid execution with the original guard-to-memory correspondence.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.
