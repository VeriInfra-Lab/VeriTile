# Checked-source compilation failure on admitted inputs

The current canonical revision invokes the tensor method `.to(tl.int64)` on
arguments that ordinary Triton JIT specialization turns into Python integer
constants. Legal nonempty calls fail during compilation, before any output
exists. This is a source defect in the checked revision, not a numerical
experiment obligation or a failure of the mathematical formula.

Canonical source: `int8_dequant_matmul_repaired.py`, SHA256
`d71b15843752bfa2904d3d341a49ce9017635a7a4273acf1ed4d7edb4620c44f`.
This matches `contract_revision.source/source_sha256` in the task ledger.
The source, assertions, configurations, and all existing Lean files remain
unchanged. The defect diagnosis must not be counted as correctness completion.

## Target, precision, and domain

For each output `(m,n)`, the independent mathematical formula is

```text
state_w[n] * (state_x[m] *
  sum(k=0..K-1, signed8(A[m,k]) * signed8(B[k,n])) / 16129)
+ optional_bias[n]
```

The actual source uses INT8 A/B, an INT32 dot accumulator, a runtime FP32
divisor obtained from the binary64 literal `1/(127*127)`, FP32 row/column
scales, and FP16 output. It narrows optional FP32 bias to FP16 and adds that
bias only on split zero. Split count one stores; larger split counts use FP16
atomic adds after the existing zero hook. The unused `ACC_TYPE=tl.float32`
does not change the accumulator type.

The current assertions admit contiguous strided INT8 `[M,K]` and `[K,N]`
matrices, FP32 scale vectors `[M]` and `[N]`, and optional FP32 bias `[N]` on
one CUDA device, without unresolved negative views. They retain their
alignment, alias and index/offset conditions. No new finite-input, K-bound,
overflow, or dimension restriction is added here.

## Two small admitted cases

Both cases use all-ones INT8 A/B, all-ones FP32 scales, no bias, ordinary
separate contiguous allocations, and `K=32`:

| Case | Shapes | Exact formula at every output | First failed expression |
| --- | --- | --- | --- |
| 1 | `M=N=1` | `32/16129` | `M = M.to(tl.int64)` at source line 96 |
| 2 | `M=N=2` | `32/16129` | `stride_ak = stride_ak.to(tl.int64)` at source line 100 |

The actual canonical `_check` and wrapper AST accept both cases. Their input
contents and all non-CUDA metadata are backed by real CPU tensors. CUDA owner,
allocation and launch are mocked solely to capture the exact arguments sent
by the wrapper. All 418 canonical configurations have valid small grids
`(1,SPLIT_K)` on these cases.

For both calls, the wrapper passes its existing contiguous launch strides
`(K,1,N,1,N,1)`. The ordinary native JIT specializer and `_pack_args` mark
unit-valued arguments as `constexpr`. In case 1, `M` and `N` are specialized
integers; in case 2 they are runtime i32 values, but `stride_ak` is still the
specialized integer `1`.

In each case, offline compilation of both these unchanged source configurations
raises `triton.compiler.errors.CompilationError`:

| Configuration, zero based | BM / BN / BK / split K | Warps / stages |
| --- | --- | --- |
| 18 | 16 / 32 / 32 / 1 | 2 / 2 |
| 6 | 64 / 128 / 32 / 1 | 4 / 4 |

The underlying diagnostic is identical:

```text
AttributeError("'int' object has no attribute 'to'")
```

These casts are unconditional source expressions before the K loop. Rejecting
singleton M/N would not fix the second case: the wrapper always passes
`stride_ak=1`, `stride_bn=1`, and `stride_cn=1`. The failure is unrelated to
integer overflow, floating rounding, atomic order, input layout, or grid size.

## Source binding and control

`check_counterexample.py` uses the unchanged canonical function body,
parameter annotations and `@triton.jit` decorator. It checks that the JIT's
parsed body/parameters equal those extracted from the canonical AST. Only
the outer host autotune/heuristics decorators are omitted for direct offline
compilation. Configurations and the EVEN_K value are obtained from the
canonical configuration AST and captured wrapper values. The legacy host
`triton.ops.matmul_perf_model` import is not executed, so compatibility of that
host helper is not the diagnosed failure.

The four expected failures use Triton 3.7.1 and
`GPUTarget("cuda",90,32)`, with each configuration's original compiler options.
No forced generic-i32 signature replaces the ordinary argument specialization.
Each receipt includes the complete signature, constexpr values, attributes,
configuration and saved compiler traceback.

As a targeted control, the original benchmark kernel compiles successfully for
case 2, configuration 18, using the same argument values and ordinary JIT
specialization. The original kernel lacks these scalar `.to(tl.int64)` lines.
This is a compiler control only: it neither certifies the original benchmark's
correctness nor replaces the current checked source with the original source.

## Classification and preserved work

The diagnostic process exits successfully only after observing all four exact
compiler failures, actual assertion admission, and the successful original-source
control. This means the **diagnostic passed**, not that the canonical kernel
compiled. No GPU execution, independent numerical replay, or new Lean proof is
claimed.

All 37 preexisting task files are hash-preserved. Existing ideal-Real and
abstract selected-instruction/source-plan theorems retain their stated scope.
The counterexample does not assert that these Lean propositions are false;
it identifies why their successful-compiler boundary is unavailable for these
admitted canonical calls. The source execution connection cannot be completed
by declaring primitive conversion/ISA laws to be numerical experiments.

This packet adds no assumption or source repair. A distinct repair must make
the integer widening valid for both runtime tensor values and specialized
scalar constants, then recheck the actual compiled accumulation behavior.
No native integer-output counterexample is claimed for a canonical kernel
that has not compiled.

Reproduce the diagnostic from the repository root:

```bash
python bench/repairs/tritonbench_g/int8_dequant_matmul/check_counterexample.py
```

The final receipt and four error traces are in
`.veritile-cache/kernel-evidence/int8_dequant_matmul/constexpr-cast/`, alongside
the original-source compiler control artifacts. The delivery manifest pins
these files and the observed diagnostic process exit.

References:

- [Triton JIT](https://triton-lang.org/main/python-api/generated/triton.jit.html):
  JIT compilation and specialization controls.
- [Triton tensor.to](https://triton-lang.org/main/python-api/generated/triton.language.tensor.to.html):
  the tensor conversion method invoked by the failing source.
- The saved signatures and tracebacks record the specific application of the
  installed Triton 3.7.1 specializer and compiler. No general compiler proof
  or experimental hardware-conformance claim is inferred.
