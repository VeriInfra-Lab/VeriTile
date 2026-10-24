# ReLU: kernel proof and explicit contiguous FP32 contract

The original benchmark has a source defect: `if pid == 0` suppresses stores
from every later program. With 1025 input ones and output storage initially
filled with `-12345`, output element 1024 stays `-12345` instead of becoming 1.
The original Python remains unchanged (SHA-256
`2dffef82d69f280104c25b798e23f59a1ba9d00325552b91149a1d3b50fcffb0`).
The separately named repair removes this gate. Its proof does **not** certify
the original benchmark or shrink the contract to at most 1024 elements.

## Target, precision, and input contract

The independent mathematical target is `out[i] = max(0, x[i])` for every active
lane of every kernel program. The selected source uses FP32 input, load,
comparison, selection, and output; there is no accumulation or arithmetic
rewrite. `Real` is solely the mathematical interpretation.

The wrapper checks Tensor classification, strided layout, FP32 dtype, CUDA,
default contiguity, no unresolved negative view bit, `numel < 2**31`, and a
4-byte-aligned actual pointer when nonempty. There is no finite-value guard,
input copy, view normalization, or added dtype variant. Empty/scalar tensors,
nonzero storage offsets, and contiguous singleton dimensions remain admitted.
The output buffer is a fresh FP32 allocation; its allocator implementation and
the wrapper's launch/return behavior are outside the required kernel proof.

`Contract.lean` lists the checks and matching mathematical contract in the same
order. `TensorBridge.python_checks_iff_preconditions` proves the exact admission
correspondence. `Proofs/Binding.lean` uses the shared `VeriTile.Torch.FP32` API
to connect contiguity and the negative flag with actual logical-element reads,
storage offsets, the observed data pointer, and checked alignment. The
`native_indices` theorem retains the original finite int32 index/mask proof;
its range includes the padded masked lane at `2**31 - 1`.

The alignment guard comes from scalar FP32 memory accesses, not from a demand
for vector alignment. Shared [load/store modeling references](../add_example/LOAD_STORE_AUDIT.md)
record Triton masking and PTX natural-alignment rules. Valid Tensor storage and
ordinary compiler/runtime interpretation remain modeling boundaries.

## Source and public proofs

- `Kernels.lean` contains the canonical readable typed Triton DSL. The repair
  and the existing store-inline transformation are the only two versions.
  `Correct.lean` and `FPEquiv.lean` refer to exactly `originalKernel`.
  The optional `ib`, `ob`, and `elementBytes` parameters express pointer
  coordinates: `(0, 0, 1)` is the displayed Python element addressing;
  physical helpers use byte coordinates. They are not FP operations or
  additional Python inputs. No hand-written source AST remains.
- `Correct.relu_kernel_correctness` uses the `bench/examples`-style
  `MaskedKernelIO₁ ⊨` contract: successful execution, `max 0 x` on all active
  lanes, safe masked accesses, and the exterior memory frame. Its write mask
  includes every active program, so the original gate defect is not hidden in
  a weaker specification. Inactive load values are unrestricted.
- `TensorBridge.relu_checked_kernel_correctness` applies the existing execution
  proof to a checked `Torch.FP32.Tensor`. This is an ordinary theorem;
  `relu_kernel_correctness` is the single mathematical specification.
  `N` is exactly `x.metadata.numel`;
  the initial input cells come from `read (x.dataPtr.toNat + 4*i)`.
  Shared Tensor laws and the input contract derive their equality with
  `logicalValue read x laws i`. The public result gives `max 0` of those
  logical inputs on every active lane, successful execution and the frame.
  Its `decode : Word → Real` only chooses the mathematical interpretation of
  input words; it makes no FP32-to-Real exactness or finite-value assertion.
- `FPEquiv.relu_inlined_equiv` proves the existing transformation
  `result = where(...); store(result)` to `store(where(...))` on the repair.
  Both actual typed sources execute successfully and have matching outputs
  and exterior frames. `FPObservations.full_memory_equiv` in
  `Proofs/FPObservations.lean` additionally states equality of
  every memory cell. The private `result` register may differ. This is the
  single FP equivalence specification. It proves preservation of kernel
  behavior; the independent mathematical result belongs to `Correct.lean`,
  and the optional Tensor application belongs to `Misc/TensorBridge.lean`.
- `FPObservations.original_source` in `Proofs/FPObservations.lean` binds the public FP program directly to the
  canonical source; precision tags are retained during FP execution. Only
  the Real proof uses the mathematical projection.

`Proofs/KernelReal.lean` adapts the original masked readback/frame argument to
all programs, reusing the established scatter infrastructure.
`IO.lean` contains the public FP windows, source/profile/domain data shared by
the reader and supporting proofs. `Proofs/FPObservations.lean` packages the actual
source runs, frame proofs and optional exact-source identity for the one-line
public FP composition. It does not import either reader entry.
`Proofs/Structural.lean` reuses the original scatter congruence and retains its
complete optional grid proof. Prior `Proofs/Host.lean`, `Proofs/Pointers.lean`,
`Proofs/Real.lean`, and `Proofs/FP.lean` remain optional supporting results;
neither public proof entry imports the local host proof. They have been
rebound to the same canonical DSL rather than retaining a second source AST.
`Misc/TensorBridge.lean` collects the input-check correspondence and the optional
Tensor application theorem for separate review. Neither `Correct.lean` nor
`FPEquiv.lean` imports it. The existing Lake task glob includes this module.
`Proofs/Binding.lean` contains the mathematical initial-state adapter and its
input-read lemmas, used by `Misc/TensorBridge.lean`. They describe a kernel's input region
and retain arbitrary surrounding state; they do not allocate, launch or execute
a wrapper. The checked correctness entry remains a per-program kernel result,
like the existing `MaskedKernelIO₁` entry, with the same exterior frame.
Python assertions and kernel sources are unchanged by this adapter.

## Floating-point scope

There are **no numerical transformation assumptions and no deferred numerical
experiments** for store inlining. The comparison-support requirement merely
requires the interpreter to implement FP32 comparison; equivalence does not
assume its Boolean result. FP32 load interpretation also remains opaque and
identical on both sides. The operand domain is unrestricted, including masked
lanes, NaNs, signed zeros, infinities, and subnormals.

Under ordinary ordered FP32 comparison, the source selects positive zero for
NaNs and negative nonzero values; it preserves the original word for negative
zero and other nonnegative values. The retained optional software
`reference`/`classification` lemmas prove that classification. Primitive
interpretation is a semantic boundary, not an experiment TODO. No numerical
reassociation, finite-only equivalence, native-to-IEEE experiment, or exact
FP-to-Real execution theorem is claimed.

## Focused validation

Run the checked Python source/guard and CPU storage tests:

```sh
python bench/repairs/tritonbench_g/relu_triton_kernel/check_contract.py
```

They bind the preserved original source to the gate-only repair, check all
eight assertions, cover 11 rejected and 14 admitted inputs, reproduce the
1025-element original defect, and compare the store-inline candidate over six
empty/tail/block boundaries and representative special FP32 words. They check
complete output memory, the output tail frame, and input-storage preservation.
CUDA device/allocation/launch operations are mocked; the int32 maximum case is
metadata-only. These are CPU checks, not GPU execution or numerical replay.

Targeted Lean validation:

```sh
LEAN_NUM_THREADS=2 PATH=/tmp/veritile-lean-toolchain/bin:$PATH \
  LAKE_HOME=/tmp/veritile-lean-toolchain /tmp/veritile-lean-toolchain/bin/lake build \
  bench.repairs.tritonbench_g.relu_triton_kernel.Correct \
  bench.repairs.tritonbench_g.relu_triton_kernel.FPEquiv \
  bench.repairs.tritonbench_g.relu_triton_kernel.Misc.TensorBridge \
  bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Real \
  bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.FP \
  bench.repairs.tritonbench_g.relu_triton_kernel.Proofs.Pointers
```

Public declarations are checked with `#axiomsClean`; the footprint is the
standard base (`propext`, `Quot.sound`, `Classical.choice`).
`#print_fp_assumptions relu_inlined_equiv` reports `none`.

The separation of mathematical correctness and FP equivalence has a scoped
[validation record](../../../../.veritile-cache/relu-tensor-entry-20261020/tensor-bridge-validation.json).
`Correct` contains the mathematical specification; `FPEquiv` contains kernel
equivalence results; `TensorBridge` holds the optional Tensor application.
The moved declarations and affected entries passed the targeted Lake build
and axiom audits. Existing Python source/guard and CPU storage checks remain applicable
to the unchanged Python source and assertions. No new numerical experiment,
GPU run or host proof was added.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.
