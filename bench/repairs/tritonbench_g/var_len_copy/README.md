# Variable-length FP32 copy: kernel contracts and source repair

The original benchmark has a tail-mask defect. With one segment of length 257,
start 0 and block size 256, its second iteration uses `lane < length` and writes
indices 256 through 511. Indices 257 through 511 are outside the requested
segment; a 257-element backing allocation also makes those accesses out of
bounds. `check_contract.py` reproduces the overwrite using a larger sentinel
buffer. The original Python and its historical proofs remain unchanged.

The separate repair tests `i + lane < length` and widens metadata, program ID
and lane indexing to signed int64 before arithmetic. The selected payload
precision is FP32, and all three metadata arrays are signed int32. There is no
floating arithmetic, dtype cross-product or payload conversion.

## Mathematical target and scope

For program `p`, write `L = max(length[p], 0)`. For every `0 <= j < L`,

```
after[new_start[p] + j] = initial_source[old_start[p] + j]
```

Every other memory cell is unchanged by this program. Nonpositive lengths
perform no writes and impose no conditions on their unused signed starts.
The theorem covers the complete dynamic loop, including partial final blocks;
it is not a one-chunk theorem. Proof scope is the kernel. The wrapper supplies
assertions; wrapper execution, allocation and returned-object identity are not
required acceptance results.

## Input contract and memory meaning

The existing Python assertions are unchanged: five rank-one, contiguous,
strided CUDA tensors on one device, no unresolved negative bit, FP32 source and
output, equally shaped signed-int32 metadata arrays, metadata count below
`2**31`, and four-byte alignment for nonempty pointers. Output backing spans
must be disjoint from all read buffers. Read buffers may alias each other.
Positive-length source and destination intervals must fit their payload
buffers, and destination intervals for different programs must be disjoint.
Contiguous storage-offset views are admitted. Only metadata is read back for
validation; payload buffers are passed directly without copies.

`Contract.lean` declares those assertions and the independent `KernelCopy`
formula. `TensorBridge.python_checks_iff` proves admission equivalence.
`TensorBinding` and `prepared_direct` in `Proofs/Contract.lean` connect actual
pointers, capacities and signed metadata cells to the finite memory model.
`logicalState`, `logicalState_binding` and `logicalState_payload` map those
same four-byte contiguous addresses to the kernel's logical regions. Real
payloads remain arbitrary mathematical values, separate from the signed
metadata decoder. No output property is assumed in the initial binding.

## Canonical source and public proofs

`Kernels.lean` contains the readable typed Triton DSL. Both proof entries use
that source directly. The source and candidate differ only in the source
address expression:

```
source + old_start + i + old_offset
source + i + old_start + old_offset
```

Both Python kernels are present, with identical repaired masks and casts.
This is the existing address-commutation candidate, not an identity comparison.

- `Correct.varLenCopyIO` visibly declares the signed metadata channels, FP32
  source's Real projection, complete input/output segment windows, masks and
  absence of scratch. `segmentDomain` constrains signed length and active starts;
  `copyFormula` independently states that each output lane equals its input lane.
- `Correct.var_len_copy_kernel_real` uses the shared
  `UKernelIO.RegionalImplementsOn3D` relation. It proves complete source execution,
  all output lanes, actual access safety from the declared bounds, and the exact
  raw-cell frame outside the destination segment. Nonpositive lengths still read
  all three metadata cells but leave their unused signed starts unconstrained.
- `TensorBridge.var_len_copy_real` and `TensorBridge.kernel_trace_safe` retain
  the complete Tensor-bound results as ordinary theorems. They reuse the same
  existing execution/safety foundations as the public IO theorem.
- `FPEquiv.source_equivalence` proves both actual sources succeed and agree on
  every typed memory cell. Its numerical assumption list is empty.
- `FPExecution.kernel_results` in `Proofs/KernelFPResults.lean` retains the exact
  destination frame for both successful executions as an ordinary helper. Arbitrary payloads, aliases in the abstract
  equivalence relation, and arbitrary masked-load values are admitted. No
  finiteness premise is imposed on inactive lanes or payloads.

`Proofs/KernelReal*` adapts the existing signed-loop execution/safety proof to
the corrected mask; shared loop arithmetic and memory lemmas are reused.
`Proofs/KernelFP*` proves the actual source transformation, successful execution
and frame over opaque floating values. It assumes no primitive conformance or
floating algebraic law.

The former finite event interpretation is retained in `Proofs/Model.lean`, not
as a second kernel AST. `Index64`, `Native` and `Conformance` retain the finite
signed masks, terminal-loop/increment bounds, pointer stages and nonwrapping
active addresses. `EventCorrect` retains the whole-request Real event result;
`HostCorrect` retains the optional historical wrapper/object/byte-frame results.
Neither public entry imports the host proof. These optional results are not
substitutes for the canonical DSL kernel proofs.

## Validation and boundaries

Run:

```
python bench/repairs/tritonbench_g/var_len_copy/check_contract.py
```

For each Python source variant, this executes the actual kernel AST over CPU
backing storage: 11 direct-copy cases, 14 rejected-input cases and the original
length-257 counterexample. Lengths 0/1/255/256/257/511/512/513, negative lengths,
multiple segments, storage-offset sentinels, NaNs, infinities, signed zero and
subnormal raw FP32 words are covered. CUDA context and launch are mocked; this
is not a GPU run.

The public entries and retained optional helpers undergo targeted Lean builds
and standard-base axiom audits. There is no deferred numerical transformation
assumption and no independent numerical replay to claim. Ordinary contiguous
Tensor backing/metadata, same-format load/store semantics, and compiler/device
execution remain platform modeling boundaries; they are not per-task numerical
experiments. Shared grid-product overflow and launch compatibility stay deferred
outside this kernel task.

`Misc/TensorBridge.lean` exposes `TensorBridge.python_checks_iff` as an ordinary
admission-correspondence theorem. Each primary entry has one headline:
`Correct.var_len_copy_kernel_real` for mathematical IO correctness and
`FPEquiv.source_equivalence` for the existing standard floating program
relation. Neither imports the Tensor bridge. The former raw Real theorem is
retained unchanged in `Proofs/KernelRealExecution.lean`; `Proofs/IOExecution.lean`
connects those proved values, exact frame and source safety to the shared IO
relation without assuming an execution, safety or output conclusion.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.

The 2026-10-21 source-only cleanup replaces the whole-source selector macro
with two complete ordinary `triton` blocks directly in `Source.kernel`.
An audit-only frozen prior definition proves complete function equality by
`funext` over every source parameter, cases on the version and `rfl`. Existing
loop/body proofs still compile unchanged. FP32 copy payloads, signed int32
metadata widened to signed int64, both repaired tail masks, loop structure,
nonpositive-length behavior, alias scope and exterior frames are unchanged.
The independent Real statement and existing two-program pair retain their
names and full domains. This source readability change does not alter proof
completion or add validation obligations.

`IO.lean` holds the canonical `PublicIO.io` layout, metadata domain and logical-output packaging. The reader reuses the IO and metadata definitions and retains its source-independent mathematical formula. The source projection stays in `Proofs/SourceProjection.lean`; `Proofs/IOLayout.lean` retains its compatibility import.
