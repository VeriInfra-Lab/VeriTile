# Current canonical JIT compilation defect

Status: **source defect pending repair**, not correctness completion or completion
with deferred numerics. The canonical source and checked input domain remain
unchanged. No repair candidate is silently substituted.

Canonical source: `int8_matmul_quantization_repaired.py`, SHA256
`2023c3fc3c849aa9a93b45bb5f672dd0d4412d175a713978dde6f70c5b164a81`.
The original benchmark is a separate source, SHA256
`af66c9bf98a319d032110ab1f28c20f8bae4c3659fb3249288135011dab802b0`.
The failing scalar widening was added in the checked canonical revision; this
report does not misattribute that edit to the original benchmark.

## Admitted counterexamples

All tensors are strided, contiguous, aligned, on the same modeled CUDA device,
without negative-view flags or storage aliases. The checker executes the actual
canonical assertion and wrapper AST, backed by real CPU tensors. Only CUDA
ownership, allocation and launch are mocked. Every dimension/address/grid check
passes for every existing configuration.

| Kernel | Inputs | First failing expression | Source line |
| --- | --- | --- | --- |
| Producer | FP32 ones `[1,32]` | `M = M.to(tl.int64)` | 109 |
| Producer | FP32 ones `[2,32]` | `stride_fpak = stride_fpak.to(tl.int64)` | 112 |
| GEMM | INT8 ones `[1,32]` and `[32,1]`, FP16 row/FP32 column scales one | `M = M.to(tl.int64)` | 227 |
| GEMM | INT8 ones `[2,32]` and `[32,2]`, FP16 row/FP32 column scales one | `stride_ak = stride_ak.to(tl.int64)` | 231 |

The independent Real targets are producer scale `1/127` and code `127`, and
GEMM output `32` at every active cell. The current canonical calls produce no
output because JIT compilation fails before launch.

The producer wrapper passes strides `[32,1,32,1,1]`; the row-major GEMM wrapper
passes `[32,1,1,N,1,1,N,1]`. Ordinary `native_specialize_impl` and
`JITFunction._pack_args` turn the unit strides into constexpr integers. Python
integer `1` has no tensor method `.to`. Using `M=N=2` avoids the dimension-one
failure and exposes the obligatory unit-stride failure. This is unrelated to
large-grid overflow or numerical accuracy.

The unchanged JIT body/signature and `@triton.jit` are compiled with the derived
native specialization. Only the host autotune decorator is omitted to compile
selected existing configurations directly. Producer configurations 3/0 and GEMM
configurations 10/27 confirm eight failures on Triton 3.7.1, CUDA SM80 warp32.
The latter GEMM configurations cover split one and split two. Two controls using
the unchanged original kernels and the same `M=2,K=32` calls compile successfully.
Those controls are compiler evidence, not original-benchmark correctness proofs.

## Replay and evidence

From the repository root:

```sh
python bench/tritonbench_g/int8_matmul_quantization/check_counterexample.py
```

The receipt is
`.veritile-cache/task-deliveries/int8_matmul_quantization/constexpr-cast/counterexample.json`.
It pins the checker and both sources, records actual arguments, signatures,
constexprs, admitted grids, all eight error traces, and both controls' compiler
artifacts. Diagnostic process exit zero means the expected counterexamples and
controls were checked successfully; it does not mean the canonical kernels
compiled. No GPU executed and no independent numerical replay ran.

Earlier `native_probe.py` profiles explicitly assigned all scalar arguments i64.
The CPU contract checker supplied int64 tensor scalar values to its interpreter.
Both bypass ordinary Python-integer specialization and therefore do not refute
this counterexample. Their results retain only their stated model/profile scope.

## Preserved work and remaining scope

The historical `Correct.lean` and `FPEquiv.lean` entries are preserved in their
declared models; no claim is made that the Lean propositions themselves are false.
New quotation-model results include full producer grid execution/output/frame,
GEMM loop/epilogue execution with actual carry and both store/atomic branches,
total signed16-to-low8 conversion, FP16 roundtrip, independent producer targets,
and an exact-integer dot-plan bridge. Their successful targeted build receipts
are retained in the delivery packet, not counted as an accepted whole problem.

The following new public connections remain incomplete and are not relabeled
as numerical assumptions: producer native/Real target specialization, full GEMM
source-to-grid recurrence, final Tensor/byte readback integration, and composition
using the stored producer INT8 values and FP16 scales. A corrected canonical
source and its source/model binding must be reviewed before completing these.

There are no new numerical assumptions or deferred experiments in this defect
delivery. Basic casts, rounding, integer wrapping/saturation and memory encoding
remain mathematical/model obligations. Global runtime grid-product handling
remains the separately deferred project item.
