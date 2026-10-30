# Admitted integer matmul counterexample

The canonical checked source has a configuration-dependent integer result on
an admitted input whose exact mathematical answer fits INT32. Under the retained
Triton 3.7.1 SM90 lowering, configuration 14 loses 65,536 through intermediate
signed saturation. Configuration 4 uses wrapping WGMMA and returns the dense
answer. This is a source defect relative to the independent matmul target,
not a deferred floating-point experiment or a completed correctness proof.

The canonical file is `int8_matmul_kernel_repaired.py`, SHA256
`e95569108e080703b0bc8df8a61c862eba863cefe6fa4c6321c9ced9ca496cee`.
Its kernel and all fifteen configuration ASTs equal those in the original
benchmark. No source, guard, configuration, or existing theorem was changed
to obtain this counterexample.

The independent formula for output row `m`, column `n` is

```text
sum(f = 0..3, q = 0..K/4-1,
    truncInt8(A[m, f*(K/4)+q]) * (((uint8(B[q,n]) >> (2*f)) & 3) - 1))
```

`A` advances continuously through four fields; the packed `B` pointer resets
at each field. INT32 is the source accumulator and output precision. Whether
the target is the unbounded sum when representable or its signed low 32 bits,
the expected result below is the same.

## Witness and admission

```text
M = N = 1
K = 16,777,728 = 2^24 + 512
A: contiguous FP32 [1,K], every value -128, strides [K,1]
B: contiguous uint8 [K/4,1], every byte 15 (0x0f), strides [1,1]
unpacked B field weights: [2,2,-1,-1]
output: fresh INT32 [1,1]
```

Use ordinary aligned strided allocations on the same current CUDA device,
without unresolved negative view bits. Every FP32-to-signed8 conversion is
exact and satisfies the existing finite/truncation guard. The source signed32
offset envelopes are respectively `33,555,584`, `4,194,561`, and `514`, all
less than `2^31`. Ordinary allocation addresses also satisfy the checked
signed64 formed-byte-pointer bounds.

`K` is divisible by 512, so every one of the fifteen unchanged per-configuration
assertions `K % (4*BLOCK_SIZE_K) == 0` holds. Every configuration has grid `(1,)`.
There is no tail or grid overflow in this example. Source row/column modulo
indexing is defined because `M=N=1`, and only output `[0,0]` is stored.

`check_counterexample.py` executes the actual canonical wrapper assertion AST
against complete real CPU-backed witness contents. CUDA metadata, output
allocation and launch are mocked. This establishes assertion admission without
claiming a GPU execution; the receipt separately records these boundaries.

## Exact-source compiler evidence

The checker imports the canonical decorated `matmul_kernel.fn` and calls the
ordinary native Triton JIT specializer on the witness scalar arguments and
aligned pointer dtype tokens. Thus `M=N=1` and unit strides have their actual
constexpr specialization, while the nonunit A row stride keeps its divisible16
attribute. CPU tokens select pointer dtype/alignment only and are not executed
as device inputs. The compiler target is `GPUTarget("cuda",90,32)`; emitted PTX
targets `sm_90a`.

Both retained configurations have four warps, four stages, and one CTA:

| Configuration, zero based | BM / BN / BK / group M | Actual instruction |
| --- | --- | --- |
| 14 | 32 / 32 / 32 / 4 | `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` |
| 4 | 64 / 128 / 32 / 8 | `wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8` |

TTIR in both profiles feeds the inner-loop accumulator into `tt.dot` and yields
the result into the next iteration. Config 14 PTX uses the same accumulator
registers for input and output. Config 4 LLIR has `scale-d = i1 true`; its PTX
scale-d predicate is explicitly true. These are carry-fed instructions, not
zero-seeded tile dots followed by a separate modular `add.u32`.

The source FP32-to-signed8 cast remains `arith.fptosi` / LLVM `fptosi ... to i8`.
PTX uses `cvt.rzi.s16.f32` followed by low-byte packing; `-128` becomes `0x80`
exactly. The packed field extraction and INT8 subtraction produce exactly
`[2,2,-1,-1]`. Thus the discrepancy does not involve approximate arithmetic or
an out-of-domain cast.

## Integer calculation

Each field contains `K/128 = 131,076` K32 instruction updates per output.
The first two fields add `-8192` per update; the last two add `4096`.

| After field | Signed saturating config 14 | Wrapping config 4 |
| --- | ---: | ---: |
| 0 | -1,073,774,592 | -1,073,774,592 |
| 1 | -2,147,483,648 | 2,147,418,112 |
| 2 | -1,610,596,352 | -1,610,661,888 |
| 3 | **-1,073,709,056** | **-1,073,774,592** |

The independent dense result is `-128*(K/4)*(2+2-1-1) = -64*K = -1,073,774,592`,
which fits INT32. Update 262,144 reaches INT32 minimum exactly. There are eight
additional negative updates before the positive fields, losing `8*8192 = 65,536`
under saturation. Subsequent additions cannot restore that lost value.

The checker uses Python arbitrary-precision integers for both documented
instruction recurrences. Control sizes `K=512` and `K=2^24` yield identical
saturating, wrapping, and dense results. These are exact arithmetic checks,
not sampled floating numerical tests.

## Scope and preserved work

The actual guard execution, original/canonical kernel and config AST equality,
two offline compiler profiles, and integer recurrence checks passed. No GPU
kernel was run, and no independent numerical replay or new Lean proof is
claimed. The existing forty source/proof/document files are hash-preserved.

The prior unbounded mathematical theorem and selected-ISA finite recurrence
theorem remain valid within their stated interpretations. This counterexample
does not assert that those Lean propositions are false. It shows why proving
the selected recurrence alone cannot establish a configuration-independent
integer matmul result for the current canonical program.

The old conversion/instruction `NumericLeaves` interfaces are primitive semantic
correspondence obligations, not numerical experiments that can justify deferring
this discrepancy. This packet adds no numerical assumption and no K bound.
It makes no correction to the canonical program. A separate repair would need
to establish consistent accumulation semantics across the retained source
configurations; such a candidate is not silently substituted here.

Reproduce the targeted diagnostic from the repository root with

```bash
python bench/repairs/tritonbench_g/int8_matmul_kernel/check_counterexample.py
```

Authoritative evidence is under
`.veritile-cache/kernel-evidence/int8_matmul_kernel/canonical-launch/`:
`counterexample.json` and each configuration's TTIR, TTGIR, LLIR and PTX.
The task delivery manifest pins these artifacts and the observed successful
process exit. Earlier generic-profile diagnostic artifacts are not the final
evidence used for this classification.

Modeling references:

- [Triton dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html):
  operand and accumulator types.
- [PTX MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma):
  signed INT8 products, signed INT32 accumulation and `.satfinite` overflow.
- [PTX WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma):
  INT32 accumulation, wrapping without saturation, and scale-d accumulator use.
- [PTX conversion](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt):
  round-toward-zero float-to-integer conversion.
- [LLVM fptosi](https://llvm.org/docs/LangRef.html#fptosi-to-instruction):
  signed conversion and its representable-result domain.

These references specify the model used to interpret the saved instructions;
they are not a claim of experimental hardware conformance.
