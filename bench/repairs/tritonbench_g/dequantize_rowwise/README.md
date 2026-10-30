# Rowwise dequantization: INT8 kernel proof

The selected route is **INT8 weights, FP32 row scales, FP16 output**. Every
original benchmark case explicitly creates INT8 weights (`dequantize_rowwise.py`
lines 51, 57, 63 and 69); the production wrapper explicitly allocates FP16 output
at line 30. This retains the required quantized integer type. The earlier FP32
input repair is preserved separately as optional prior scope, not silently
reclassified as an INT8 result.

## Mathematical target and source

For every row and column, the independent Real formula is

```
output[row,column] = scale[row] * signed_int8(x[row,column]) / 127
```

Zero and negative scales and the entire signed INT8 range are included. Real
is the mathematical specification; native execution keeps the two FP32
multiplications in their original left-associated order and converts their
result to FP16. Python's runtime reciprocal remains `1.0 / 127`, passed at FP32
precision. No floating arithmetic is reassociated.

`Kernels.lean` now contains the selected Python function body verbatim inside
`triton { ... }`. Both `.to(tl.int64)` calls remain. There are no `tl.region`,
`tl.param`, `$(...)` splices or extra `.to(tl.float16)` calls in that body.
The candidate changes only `pid * BLOCK_SIZE` to `BLOCK_SIZE * pid`.

`callSignature` is the outer call/verification type configuration: INT8 weights,
FP32 scales, FP16 output, the runtime FP32 reciprocal, the unused integer
`n_elements` parameter, and the two constexpr block dimensions. The generic
`with_triton_signature` frontend uses this information to type loads and apply
Python's implicit destination conversion at `tl.store`. It rejects missing
pointer metadata, duplicate parameters, metadata written inside the pasted body,
and unsupported store/alias control flow. This configuration specializes the
selected production route; it does not infer Tensor dtypes from Python text.

`body` is the callee. `source` uses the shared `withScalarArguments` call adapter
to supply `inv_127` and `n_elements` before executing that callee. FP32 argument
rounding occurs through the numerical backend at the call. This is separate
from the copied body, which reads the scalar register. The Real proof erases
precision locally and retains the existing independent Real formula.

`BodyExecution.source_memory` proves the complete new execution's memory
connection to the established natural-index reference, retaining execution
failure. `KernelCorrectness` uses that connection to prove success, outputs,
safety and the full exterior frame for the actual new call. The FP proof
compares the two complete new calls directly using signed multiplication
commutativity, with no floating numerical rewrite or experimental rule.
`NativeIndex.lane_exact` retains its independently proved fixed-width byte/index
projection under the unchanged physical-input bounds. The mathematical index
carrier and those physical bounds remain distinct; this is not a compiler/ISA
refinement theorem.

To check the pasted body and its external signature:

```bash
python3 -m bench.jit_correspondence bench/repairs/tritonbench_g/dequantize_rowwise/jit-correspondence.json
python3 -m unittest bench.tests.test_jit_correspondence
lake build bench.repairs.tritonbench_g.dequantize_rowwise.Correct \
  bench.repairs.tritonbench_g.dequantize_rowwise.FPEquiv bench.tests.CallSignature
```

This first example checks a straight-line Python body. Other operations and
control-flow spellings retain the existing DSL support boundary; this change
does not claim that arbitrary Python function bodies are already supported.

## Input contract

The wrapper checks rank-two INT8 `x` and rank-one FP32 `state_x`, strided CUDA
layout, a matching device, contiguous storage and no unresolved negative view
bits. Columns are at most `2**20` (Triton's block limit); rows are below `2**31`
(the row-program index range). Nonempty inputs require one available scale per
row and a four-byte-aligned scale pointer. INT8 loads require only byte alignment;
no four-byte restriction is imposed on INT8 inputs. FP16 output buffers use their
ordinary two-byte element geometry.

Empty tensors perform no kernel launch. Scale-count and pointer checks are
skipped in that branch. The wrapper allocates the existing FP16 output on the
input device and adds no input copies or normalization. All guards precede
allocation. `TensorBridge.python_checks_iff_preconditions` proves the Lean check function is
exactly equivalent to its public input contract; `check_contract.py` connects
that check list to the actual Python AST.

`Proofs/InputBinding.lean` uses shared Torch metadata and the existing core FP32
storage binding. Its task-local INT8 descriptor connects actual byte storage,
strides, storage offsets and data pointers to signed `BitVec 8` values, whose
range is proved. Both inputs are read from one shared byte heap, so nonzero
storage offsets and overlapping read-only views across dtypes are permitted.
No input separation is hidden in the public regional theorem. Fresh, bounded
output buffers are supplied at the kernel API boundary; allocator internals are
outside this task.

The storage-offset and pointer binding follows the core references for
[`Tensor.data_ptr`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html),
[`Tensor.storage_offset`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.storage_offset.html),
and [`Tensor.is_contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html).
Natural load/store alignment follows
[PTX addresses as operands](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#addresses-as-operands):
one byte for INT8, four for FP32 and two for FP16.

## Public correctness and floating equivalence

- `Correct.lean` declares `dequantizeIO`: the actual typed source, an INT8
  row channel, an independent one-lane FP32 scale channel, and the masked
  output channel. Its single specification uses the shared
  `MixedMaskedKernelIO₂.Implements` relation to prove actual source access
  safety, successful execution, the independent `scale * weight / 127`
  formula and the full exterior regional memory frame. The unmasked scale
  read remains explicit even for a zero-column invocation; no positivity
  or read-only input separation restriction is added.
- Optional `Misc/TensorBridge.lean` provides `TensorBridge.tensor_kernel_correct`,
  which specializes the kernel result to the admitted original caller tensors
  through the actual input binding, plus the input/check correspondence helpers.
  Wrapper execution, returned-object identity and allocation are not proof targets.
- The earlier finite flat INT8 result is retained in
  `Proofs/FlatCorrectness.lean` as `dequantize_rowwise_flat_correct`.
  That optional endpoint explicitly has the
  stronger flat allocation contract. It alone does not certify overlapping
  input views; those use the primary regional endpoint.
- `FPEquiv.lean` has one specification for the two selected complete typed
  sources, using the shared `ProfiledMemory.Program` relation. Both sources
  succeed and agree on every typed memory cell. `Proofs/FPExecution.lean`
  contains execution and the reusable arbitrary-region equivalence theorem;
  the existing integer-index rewrite supplies equality. FP32 scales may contain
  any words: there is no finite-value guard or numerical side condition.

**Numerical transformation assumptions: none.** Basic FP32 multiplication,
integer-to-FP32 conversion and FP16 conversion are modeling boundaries, not
per-task experiments. GPU execution and native conformance are reported
separately; no GPU result is claimed by the CPU source checks.

## Preserved prior scope and source defects

The previous FP32-input source and all its mathematical/structural/conditional
software-reference proofs remain under `Proofs/FP32Prior/`, in namespace
`DequantizeRowwise.FP32Prior`. Its Python is
`dequantize_rowwise_fp32_prior.py`, with `check_fp32_prior.py`. Those results keep
their original FP32 scope and assumptions. Their primitive-conformance premises
are not counted as deferred numerical experiments for the selected INT8 task.

The unchecked original source is not retroactively certified. Its pointer-only
layout interpretation gives wrong tensor values for transposed or lazy-negative
inputs. Its original signed32 row-index product can also wrap (for example
`2**20 * 2**12`). The repair explicitly rejects unsupported views and widens
index arithmetic. The shared runtime grid-product/segmented-launch question
remains deferred outside this task.

## Targeted validation

```sh
lake build bench.repairs.tritonbench_g.dequantize_rowwise.Correct \
  bench.repairs.tritonbench_g.dequantize_rowwise.FPEquiv \
  bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FlatCorrectness \
  bench.repairs.tritonbench_g.dequantize_rowwise.Misc.TensorBridge \
  bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Correct \
  bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.FPEquiv
python bench/repairs/tritonbench_g/dequantize_rowwise/check_contract.py
```

The CPU checker runs the actual repaired JIT/wrapper AST, verifies its guard list
against Lean and compares its source with the original after erasing only the
integer widenings. It covers signed INT8 extremes, empty axes, tails, byte-offset
inputs, shared input bytes, zero/negative/special FP32 scales, FP16 output and
unchanged caller storage. CUDA admission and launch are mocked, so this is not a
GPU experiment or an independent numerical replay.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
