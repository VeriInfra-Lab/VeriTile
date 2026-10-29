# matrix_transpose: FP16 kernel transpose

The public entries prove the kernel. The wrapper supplies dimension assertions
and the original FP16 contiguous input/output allocations; allocation, random
sampling, launch orchestration and returned Tensor identity are outside the
required proof scope. The original benchmark Python remains unchanged.

## Mathematical target, precision and input contract

For input shape `(m,d)` and output shape `(d,m)`, the independent result is

```text
out[col,row] = matrix[row,col]
out[col*m+row] = matrix[row*d+col]
out[o] = matrix[(o % m)*d + o/m]
```

The formula reads the initial input. `Contract.lean` defines it independently
of the source and interpreter. `InputCells` connects every logical input element
to its typed FP16 memory cell; it is a storage binding, not a finiteness test.

The Python arguments are two exact built-in integers, excluding `bool` and
integer subclasses. Both dimensions must be positive powers of two with
`m*d <= 2**20`. `TensorBridge.python_checks_iff_preconditions` proves that the ordered Python
checks modeled in Lean admit exactly that contract. `contract_iff_sourceShape`
proves the equivalent independent family `m=2**a`, `d=2**b`, `a+b<=20`: all 231
shape pairs, including singleton, rectangular and maximum-size cases.

The wrapper itself creates FP16 input/output with strides `(d,1)` and `(m,1)`.
It has no caller-provided tensors, layouts, strides or dtype choices. Ordinary
successful allocation supplies valid disjoint, naturally aligned storage; no
allocation implementation is proved. The source performs only a typed FP16
load, transpose and store. There is no floating arithmetic, cast, FP32
substitution or finite-value guard. Every 16-bit payload is preserved, including
NaN payloads, signed zero, infinity and subnormals. The Real proof is the
separate mathematical interpretation of the same source.

The original unrestricted wrapper fails before producing a result for `(0,4)`
or `(3,4)`: `tl.arange` requires positive power-of-two endpoints. A tile of
shape `(2**10,2**11)` exceeds Triton's `2**20` element limit. These are explicit
frontend preconditions, not wrong-output counterexamples within the admitted
domain. All admitted lanes are active and in bounds; there is no masked tail or
empty branch. Resource/compilation failures on large tiles are runtime boundary
behavior, not a guarantee that every admitted shape runs on every GPU.

## One source and the actual equivalent variant

`Kernels.lean` contains complete ordinary FP16 `triton` quotations for the
existing original and index-commuted candidate. Their complete function identities
to the previous shared sources are proved by `rfl`; legacy `kernel`/`typedKernel`
aliases remain for compatibility. The previous implementation reused the DSL in
[`bench/tritonbench_g/matrix_transpose/Kernels.lean`](../../../tritonbench_g/matrix_transpose/Kernels.lean).
`source` selects its FP16 original; `indexCommutedSource` selects its existing
index variant. This is the sole typed source used by both public entries.
The historical untyped source is used only through a proved mathematical
projection; it is not another FP implementation. No handwritten source AST is
introduced.

Both selected definitions now use `ComputeKernel := triton { ... }` directly.
The DSL follows the intermediate pointer bindings to infer `[M]` and `[Out]`,
so no outer `ComputeKernel.mk ... (...).surfaceBody` is needed. Complete
function equality to the previous definitions, including ports, is checked by
`rfl`; see the [scoped validation](../../../review-checkpoints/2026-10-22/matrix-transpose-direct/README.md).

The variant changes only `row * matrix_stridex` to `matrix_stridex * row`.
`matrix_transpose_repaired.py` now displays both Python kernels; its production
wrapper still calls the unchanged original kernel. The FP16 loads, transpose,
stores and other indices are identical. Integer multiplication commutativity
proves this change; it is not a floating transformation or an identity-source
shortcut.

## Public proofs

- `Correct.transpose_kernel_correctness` proves successful flat-memory
  execution, every output lane and the exterior frame for any disjoint flat
  placement of the declared windows. The transposition is explicit in those
  windows and proved against the actual source.
- `KernelResults.transpose_kernel_coordinates` exposes the independent coordinate
  formula, the complete admitted shape contract, successful execution from
  arbitrary initial memory and preservation outside the output extent.
- `FPEquiv.transpose_index_equiv` proves that the original and actual existing
  index-commuted source both execute successfully and have identical complete
  typed memory. Its numerical assumption list is empty.
- `KernelResults.transpose_kernel_fp16_correct` proves the independent coordinate
  formula for all FP16 bit words from `InputCells`, with the exterior frame.
  `transpose_no_numerical_assumptions` proves independence from every floating
  primitive implementation because no such primitive is reached.
- `KernelResults.transpose_source_addresses` retains the source-order physical
  pointer proof: signed-int32 products, separate sign-extension/scaling of the
  two offsets, uint64 non-wrapping, natural two-byte alignment, and bounds for
  both bytes of every FP16 access. The shape contract establishes the index
  bounds; valid allocation bounds/alignment are explicit premises.

The main `Correct.lean` displays the source, regions, coordinate windows and mask;
`FPEquiv.lean` displays both actual typed source programs. The raw coordinate,
word-preservation and pointer corollaries are ordinary theorems in
`Proofs/KernelResults.lean`.

`Proofs/KernelReal.lean` and `Proofs/KernelFP.lean` connect these statements to
existing applicable execution proofs. They do not import public proof entries,
host runners or allocation proofs. `Proofs/Real.lean`, `Proofs/FP.lean` and
`Proofs/Host.lean` retain earlier optional host results; they are not imported
by either public entry or required for kernel acceptance.

There are **no numerical assumptions or deferred numerical experiments** for
this task. Compiler, Torch/allocator internals and GPU scheduling remain model
boundaries. Load/store and physical-address modeling references are collected
in [`documents/GpuMemoryModel.md`](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts).
Triton's [`arange` reference](https://triton-lang.org/main/python-api/generated/triton.language.arange.html)
and [`core.py`](https://github.com/triton-lang/triton/blob/main/python/triton/language/core.py)
document power-of-two endpoints and the tensor-element limit. These semantic
and frontend facts are not per-task numerical experiment obligations.

## Targeted validation

```bash
env PATH=/tmp/veritile-lean-toolchain/bin:$PATH \
  LAKE_HOME=/tmp/veritile-lean-toolchain LEAN_NUM_THREADS=2 \
  /tmp/veritile-lean-toolchain/bin/lake build \
  bench.repairs.tritonbench_g.matrix_transpose.Correct \
  bench.repairs.tritonbench_g.matrix_transpose.FPEquiv
python bench/repairs/tritonbench_g/matrix_transpose/check_contract.py
```

The two public entries and dependencies build with standard-base axiom audits;
the independent formula passes the noncircularity audit. The FP assumption
report is `none`.

The Python check verifies the original kernel and admitted wrapper AST are
unchanged, and that the candidate AST is precisely the single index-operand
swap. It checks all 231 admitted shape/metadata pairs, 16 rejection boundaries,
and both actual kernel bodies at seven representative shapes. The `256x256`
case covers all 65,536 FP16 bit patterns. Existing failure-propagation checks
remain available. CPU storage and launch mocks are used: GPU execution and
independent numerical replay have not been run, and no replay is needed for an
empty numerical assumption set.

`Misc/TensorBridge.lean` contains the optional Python-check correspondence. The
primary `Correct.lean` and `FPEquiv.lean` entries retain the kernel results and
do not import the bridge.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.

Source-only cleanup preserves the existing precision, domains, bindings,
candidates and theorem declarations. No proof scope is reopened or recounted;
no numerical, native, host or hardware validation obligation is added.

`IO.lean` holds the canonical `KernelReal.transposeIO` descriptor. The public `Correct.transposeIO` is a thin instance; the source/IO correctness lemmas remain in `Proofs/KernelReal.lean`.
