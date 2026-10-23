# attention_kernel: checked direct-input repair

The revision fixes the original V/Out head bases, independent batch pitches,
query/key tails and bias query masks. It mutates the caller's output and returns
`None`. Inputs are passed directly; the wrapper makes no copies. The original
benchmark is unchanged (SHA256
`fbdb987fbd6fd07203c0f539018c01432c5fd617a76a93b0a4f34213f43bf9cb`).

## Contract and formula

The five ordinary strided CUDA tensors have one device and a common FP16 or
BF16 dtype, with no unresolved negative/conjugate bits. Q/K/V/Out have shape
`[B,H,N,D]`; contiguous bias has shape `[B,H,N,128]`. D, BM and BN are in
`{16,32,64,128}`, warps in `{4,8}`, stages in `1..4`, and
`ceil(N/BN) <= 128`. The Python scale is a finite int/float in FP32 range.

For a nonempty call, Q/K/V/Out have inner stride 1, row stride D, head stride
at least N*D and batch stride at least H*headStride. Batch/head pitches may
differ between tensors. The checked signed-i32 envelope covers source
arguments, formed offsets and padding; grid Y is at most 65535. All actual
pointers are 2-byte aligned and their formed byte envelopes fit unsigned64.
The output's full storage interval is disjoint from each input's full storage;
read-only input aliases are admitted. Empty B/H/N returns before pointer,
alignment, stride-envelope and alias checks and leaves storage unchanged.

For each batch/head and query i, the independent Real score is

```
s[i,j] = (scale * 1.44269504) * sum_d Q[i,d] * K[j,d]
       + (bias[i,j//BN] + bias[i,64 + (j%BN)%64]) * 1.44269504
out[i,d] = sum_j 2**s[i,j] * V[j,d] / sum_j 2**s[i,j]
```

This source is noncausal. The exact decimal log2(e) approximation and the
block-dependent relative-bias indexing remain visible. `Correct.lean` proves
the formula in the Real projection, including the complete finite caller
output, `None`, address safety and the exterior storage frame. It makes no
FP-to-Real accuracy claim.

## Precision and proof boundary

`Kernels.lean` contains both storage-precision bodies. Q scaling is FP32 before
narrowing to storage; QK and PV dots accumulate in FP32. Bias addition and its
log2(e) multiplication occur at storage precision. Probability is narrowed
before PV; online maximum/normalizer/accumulator and final division are FP32,
then the output is stored at its original precision. Fusion remains enabled.

`FPEquiv.attention_native` connects the complete typed source, all grid
programs in any legal order, finite input words, caller output cells and their
frame to an explicit instruction graph. The graph retains scalar operand
order, storage casts, reductions and seeded fixed m16n8k16 instructions.
`Arguments.scaleWord` is the observed FP32 launch word: the unique scalar-scale
site uses these bits, preserving negative zero and scalar underflow. This
closes a signed-zero omission in the older family model without changing the
Python kernel. The Real target continues to erase scalar rounding.

The ordinary boundary binds classified Tensor metadata, strides, storage
identities/capacities, pointers, entry-state words and the scalar launch word
to the Python arguments. `Proofs/Contract.lean` is a semantic guard interpreter,
not a verified Python parser. Its successful checks derive the historical
input ABI and integer/address domain; they contain no output predicate.
Tensor storage bounds and separation identify actual finite caller storage.
The compiler boundary contains integer routing metadata only. Successful,
race-free runtime execution and ordinary Tensor/JIT semantics remain external.
The separately deferred shared launcher grid-product issue remains separate.

`attention_local_numerical_transport` is conditional only on scalar instruction
and fixed seeded-MMA coordinate laws. There is no variable-dot, whole-reduction,
softmax, output-equality or execution-equivalence numerical premise. NaN/Inf
input words remain in the structural domain. Hardware arithmetic conformance
and any accuracy/error budget require the deferred numerical experiments.

## Validation

The 32-module source/helper/public closure passed targeted Lean 4.29.0 builds,
standard-base axiom checks and independent-specification audits. Helpers reuse
the accepted Whole001 source/Real and Whole002 native family results.

`python bench/repairs/tritonbench_g/attention_kernel/check_contract.py` passes
55 metadata/host controls (including six signed scalar-word cases), seven
complete address grids and six original addressing/tail counterexamples.
These controls enumerate addresses and mock the launch; they do not execute
floating instructions.

`check_cuda_lowering.py` compiles eight current-source SM80 profiles with
Triton 3.7.1 using the real JIT signature binder. `check_precision.py` verifies
Q widening/narrowing, storage bias arithmetic, FP32 QK/PV, probability and output
casts in all eight artifacts. Both FP16 and BF16 are covered. The reports default
to `/tmp/veritile-attention-kernel-lowering`. These are compiler/source checks;
CUDA was unavailable and no GPU execution was performed.
