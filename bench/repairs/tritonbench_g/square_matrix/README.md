# square_matrix: contiguous FP32 kernel square

`Correct.lean` displays the source, input/output row windows and mask in
`Correct.squareIO`, followed by the independent pointwise-square specification.
`FPEquiv.lean` compares the two typed sources through the shared successful
memory-equivalence judgment. Execution, coordinate and source-precision helpers
remain in `Proofs/KernelReal.lean` and `Proofs/KernelFP.lean`.
[TensorBridge.lean](Misc/TensorBridge.lean) retains all metadata/Tensor applications.

The independent mathematical target is pointwise square:
`Y[row,col] = X[row,col] * X[row,col]`. It is not matrix multiplication.
The original source leaves the input dtype unspecified, so this task selects
FP32. Its load, multiplication and store precision remain explicit. The original
benchmark Python is unchanged.

## Input contract and source

The revised Python wrapper admits rank-two, strided, contiguous FP32 CUDA
Tensors with no unresolved negative view bit and `numel < 2**31`. A nonempty
input has a four-byte-aligned actual data pointer and
`next_power_of_2(n_cols) <= 2**20`. Python checks and `Contract.lean` have the
same named conditions and check order; `python_checks_iff_preconditions` proves
exact admission for their modeled metadata.

Both zero-row and zero-column matrices return before launch. They have no
unused pointer, stride or column-count restriction. The kernel theorems describe
nonempty launches and every valid row. `block` retains the installed helper's
six shifts; 4/8/16-warp branches are unchanged. Those host-selection definitions
live with the contract, not in the source-only `Kernels.lean`.

Nonzero storage offsets and irrelevant singleton-dimension strides remain
supported when PyTorch reports contiguity. The wrapper passes the original
Tensor directly and supplies the logical column count as both row strides.
It performs no copy, flattening or input conversion. Ordinary output allocation
supplies fresh contiguous FP32 storage on the input device. Allocation,
launch orchestration and returned Tensor identity are not kernel proof tasks.

The shared `VeriTile.Torch.FP32.TensorLaws` connects metadata to actual storage,
contiguity and `data_ptr`. `InputCells` binds every initial kernel input cell
to a read from that original storage window. The public
`square_input_address` theorem then proves that the source's actual signed-i32
row product and separately scaled pointer additions read the Tensor's logical
element. This is a memory interpretation, not a collection of unchecked Bool
flags. No finite-value input restriction is imposed: NaNs, infinities, subnormals,
signed zeros and ordinary negative values remain admitted.

`Kernels.lean` contains complete ordinary FP32 `triton` quotations for the
existing original and index-commuted candidate. Both complete function identities
against the previous shared sources are proved by `rfl`; the historical
`sourceKernel` alias remains for compatibility. The previous source reused the DSL in
[`bench/tritonbench_g/square_matrix/Kernels.lean`](../../../tritonbench_g/square_matrix/Kernels.lean).
Both proof entries reference its original FP32 source. The existing candidate
commutes only the integer `row_idx * input_row_stride` expression. Both Python
versions are displayed in `square_matrix_repaired.py`; the production wrapper
continues to call the unchanged original kernel. FP32 multiplication, negative
infinity padding, both masks and the store are identical. No new optimization,
handwritten source AST, or additional dtype family is introduced.

## Public proof scope

- `Correct.square_kernel_correctness` proves the mathematical square for every
  active lane, successful flat-memory execution, actual source access safety and
  the exterior frame through the shared `MaskedKernelIO₁` contract.
- `KernelReal.coordinates` retains the complete current-row coordinate result;
  `block_covers` connects the selected block with every logical column.
- `FPEquiv.square_index_equiv` proves that both complete typed sources succeed
  with identical entire memory. Only integer index multiplication is commuted;
  its numerical assumption list is empty.
- `KernelFP.coordinates` retains source-order FP32 multiplication for every raw
  input word and the entire typed-cell exterior frame. `KernelFP.precision`
  checks the actual product/store tags. The structural `.real` carrier is opaque
  FP32 data with explicit `.fp32` ComputeOp tags, not exact Real arithmetic.
- `TensorBridge.square_kernel_fp32` and `square_input_address` retain the
  original Tensor values and actual-pointer/logical-element connection.
  `Indices.mask_exact`, `pointer_exact` and `tensor_pointer_read` retain signed
  comparisons, nonwrapping active addresses and access bounds.

Helpers in `Proofs/KernelReal.lean` and `Proofs/KernelFP.lean` reuse applicable
per-row execution and memory proofs. Public entries do not import host runners,
allocator proofs or one another. Historical grid/host helpers remain optional;
the old Real wrapper statement is preserved in `Proofs/WrapperReal.lean`.

The mathematical formula is independent of execution. Floating equivalence
compares the actual implementations without identifying FP32 multiplication
with an exact Real square. Primitive rounding, FTZ and native instruction
interpretation are semantic/modeling boundaries. The former `LocalFP32Square`
native-to-IEEE conformance condition is therefore removed from task acceptance
and from the numerical TODO list. **There are no numerical transformation
assumptions or deferred numerical experiments for this task.** No conditions
are imposed on inactive lanes.

## Original-source diagnostics and boundaries

For `[[1,2],[3,4]].T`, the original unrestricted wrapper supplies row strides
of one and writes offsets `0,1,1,2`; output offset three is unwritten. The
existing `RealProof.original_transpose_missing` proof and CPU counterexample
are retained. The explicit revised contract rejects that unsupported
noncontiguous input; it does not certify the unrestricted original wrapper.
The original signed-i32 product also wraps at row `2**30` times stride two;
`Indices.original_overflow` records that concrete diagnostic. The checked
extent and logical row stride exclude overflow for launched rows.

Load/store and alignment references are collected in
[`documents/GpuMemoryModel.md`](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts).
The natural-alignment premise follows the documented PTX memory-access contract;
its actual pointer is checked in Python. Successful Tensor allocation and the
compiler/runtime realization of the modeled kernel remain ordinary boundaries.
The shared grid-product-overflow compatibility issue remains deferred; no
per-task workaround is added.

## Targeted validation

```bash
env PATH=/tmp/veritile-lean-toolchain/bin:$PATH \
  LAKE_HOME=/tmp/veritile-lean-toolchain LEAN_NUM_THREADS=2 \
  /tmp/veritile-lean-toolchain/bin/lake build \
  bench.repairs.tritonbench_g.square_matrix.Correct \
  bench.repairs.tritonbench_g.square_matrix.FPEquiv
python bench/repairs/tritonbench_g/square_matrix/check_contract.py
```

The targeted Python check binds the unchanged original kernel AST and the
candidate's single integer-operand swap. For each variant it checks 16 CPU
Tensor admissions, 14 rejections, metadata-only extent/block boundaries,
singleton strides beyond i32, nonzero offsets, empty inputs, all three
misalignment residues, masked tails, warp branches and special FP32 values.
Both original layout/overflow counterexamples are reproduced. CPU storage,
allocation and launch mocks validate source/layout behavior; they are not a GPU
multiplication experiment. GPU execution and independent numerical replay have
not been run; this task has no numerical law requiring replay.

The public entries contain the two specifications above. Raw coordinate and
native observations remain proved in supporting modules; the optional Tensor
applications use the same execution/value/frame foundation. No source, Python
assertion, precision, input domain or accepted proof scope changed in this
format pass.

Source-only cleanup preserves all existing precision, domain, binding, candidate
and theorem declarations. Other shared storage profiles remain unchanged; this
organization adds no proof, dtype, native, host or hardware acceptance requirement
and does not recount the existing proof completion.
