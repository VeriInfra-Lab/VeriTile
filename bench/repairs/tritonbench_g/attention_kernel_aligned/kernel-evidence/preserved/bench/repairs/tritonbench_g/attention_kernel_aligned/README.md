# attention_kernel_aligned: checked direct-input repair

The repaired entry mutates the caller's output and returns `None`, with no
input copies or new allocations. Q/K/V/Out use their own batch/head/row/column
strides. Query and key tails, bias query reads and final stores are masked.
The original benchmark remains unchanged (SHA256
`ca0ecc71aeb5d33c1e216ec1891a726281fae909e8ee765945a3bb12414453b9`).

## Scope and intended result

The five ordinary strided CUDA tensors have one device and one FP16 or BF16
dtype, with resolved negative/conjugate views. Q/K/V/Out have equal
`[B,H,N,D]` shapes. Bias is contiguous `[B,H,N,128]`; positive unusual
singleton strides remain valid. BM, BN and D independently belong to
`{16,32,64,128}`, warps to `{4,8}`, stages to `1..4`, and `N <= 64*BN`.
The scale is a finite Python int/float in FP32 range. Its observed FP32 launch
word retains signed zero; the Real parameter is that word's decoded value.

Nonempty tensors have positive strides, all full storage intervals are pairwise
disjoint, and the sorted output-span scan admits only nonoverlapping output
coordinates. Grid Y is at most 65535. Signed-i32 argument and formed-offset
checks cover both query and key padding and final pointer advances. Actual
pointers are 2-byte aligned, and each formed byte envelope fits unsigned64.
The SM80 resource guard rejects BN=D=128/stages=4 and BM=BN=D=128/stages=3;
1004 dtype/configuration pairs remain. Empty B/H/N returns before unused
pointer, stride, storage-separation and envelope checks.

For each batch/head, define

```
s[i,j] = (scale * 1.44269504) * sum_d Q[i,d]*K[j,d]
       + bias[i,j//BN] + bias[i,64+(j%BN)%64]
out[i,d] = sum_j 2**s[i,j]*V[j,d] / sum_j 2**s[i,j]
```

This is noncausal base-two bias attention. Unlike `attention_kernel`, the
aligned source does not multiply its bias by log2(e). The two independent
Real entries prove this formula for every output, caller descriptor identity,
`None` and the full exterior memory frame, with finite Tensor storage bounds.

## Exact source precision and proof boundary

FP16 QK uses an explicit FP16 dot result and FP16 bias arithmetic. The original
explicit BF16 dot-result request is unsupported by the pinned backend; the
accepted BF16 repair accumulates QK in FP32 and then narrows to BF16 before
its explicit BF16 +0 and bias operations. Both paths scale Q in FP32 before
storage narrowing, cast probability to storage before PV, accumulate PV and
online state in FP32 and narrow the final output. Fusion remains enabled.

The public `Correct.lean` and `FPEquiv.lean` expose both precision branches.
`Kernels.lean` contains source and parameter definitions only. Helper proofs
reuse the accepted family closure. The native entries cover complete source
execution, current finite caller-byte operands, returned descriptor/`None`,
all output words and the exact complement byte frame. Reductions, operand
order, casts and seeded fixed m16n8k16 coordinates remain explicit.

`Proofs/Contract.lean` and its BF16 counterpart derive the source `Guard` and
physical placement from classified launch checks, observed pointers/full
storage intervals and ordinary Tensor capacity facts. The sorted span guard
implies output injectivity. Empty calls require no formed-address or separation
premises. The checked guards contain no output formula. The `InputABI` binds
ordinary Tensor metadata and scalar conversion; it has no execution or result
condition. These are semantic correspondences to Python observations, not a
verified Python compiler or Tensor implementation.

Ordinary compiler contracts supply integer operand/routing metadata and
resource feasibility. Runtime execution must complete without interference;
the shared deferred launcher grid-product issue remains separate. The local
numerical transport entries assume only individual scalar instructions and
fixed seeded-MMA coordinates. There is no dot/reduction/tensor/output oracle.
All input words, including NaN/Inf, stay in the structural domain. Hardware
conformance and error bounds remain deferred numerical work.

## Validation

The complete source/helper/public closure is checked with Lean 4.29.0,
standard-base axiom checks and independent-specification audits. The old
historical acceptance record is not rewritten by this local revision.

`check_contract.py` checks 1042 admissions and 43 rejections, including every
configuration in the 1024-member dtype/launch family. Eight complete FP16
Triton CPU interpreter grids cover query/key tails, independent column/row/head
pitches, actual input bytes and output-gap byte frames. It reproduces the
original independent-Out-head counterexample and confirms the repaired stores.
Host metadata is mocked to test CUDA guards; the CPU interpreter calls the
unchanged kernel body directly. BF16 arithmetic is not interpreter-validated.

`check_cuda_lowering.py` and `check_precision.py` compile and inspect eight
current-source SM80 profiles with Triton 3.7.1 and the real JIT binder. They
check both distinct QK precisions, Q/probability/output casts, storage bias
addition and FP32 PV. Reports default to `/tmp/veritile-attention-aligned-lowering`.
No CUDA device was available; these checks do not establish native arithmetic
accuracy or GPU execution conformance.
