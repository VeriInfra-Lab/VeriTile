# Canonical kernel fails ordinary integer-one specialization

Canonical source SHA256:
`a5a6166c7e57e82e104805f7060e09a472a62d7be35e04ca7a9296bc5b87631e`.

Under Triton 3.7.1, the checked canonical kernel fails offline compilation for
ordinary admitted positive workloads. The native JIT binder specializes scalar
integer 1 as a constexpr Python integer, while the kernel calls tensor `.to`
on that argument. No result is produced.

| Admitted shape | Actual specialized argument | First failure |
| --- | --- | --- |
| M=N=1, K=64 | M is constexpr 1 | line 35: `M = M.to(tl.int64)` |
| M=N=2, K=64 | stride_ak is constexpr 1 | line 39: `stride_ak = stride_ak.to(tl.int64)` |

Both shapes were checked with activation `""` and `"leaky_relu"`, for four
confirmed `CompilationError` results containing
`AttributeError("'int' object has no attribute 'to'")`. All cases use the sole
canonical configuration: BM128, BN256, BK64, GROUP8, stages2, warps8. A/B are
contiguous rank-two FP32 matrices on the same CUDA device; output is FP16.
All-one inputs have the independent mathematical result 64 in every output
cell, exactly represented by FP16 bits `0x5400` in either activation case.
No dtype, shape, finiteness or positive-K restriction has been added.

`check_kernel_counterexample.py` imports the exact canonical module and retains
its actual JIT kernel/helper definitions. It executes the unchanged wrapper
with real Torch CUDA FakeTensor metadata and a captured launch, then applies
`native_specialize_impl` and `JITFunction._pack_args` to its scalar arguments
and matching aligned pointer ABI. CPU pointer handles are used only by the
offline binder; they are never sent to a device launcher. The diagnostic
orders the bound arguments by `kernel.arg_names` and asserts every resulting
constexpr against the captured argument or configured value.

A separate diagnostic compiles the same unchanged source with deliberately
forced dynamic scalar parameters. That offline control succeeds and reports
196608 shared-memory bytes. It is intentionally a different ABI from the
actual wrapper specialization and is not evidence that the canonical launch
works. It demonstrates why an all-dynamic offline profile can miss this
failure.

The diagnostic process exits zero only after verifying all four expected
canonical compile failures and the explicitly separate control. There is no
GPU execution or primitive numerical experiment. This is a current canonical
source defect, not a numerical deferral or a completed kernel correctness
result. The existing abstract Real, native instruction, source-event, Tensor
and frame proofs retain their declared interpretation and are unchanged.
Historical FP16/E4M3 scopes also remain untouched; the currently selected
wrapper explicitly admits FP32 inputs and has no FP8 dtype branch.

Evidence is in
`.veritile-cache/kernel-evidence/matmul_leakyrelu_fp8/delivery.json`.
The first diagnostic attempt had an argument-dictionary ordering bug, was
retained as non-authoritative history before freezing, and is excluded from
the final four-case evidence. The final evidence uses the corrected binder
and independently checked block/group/activation constants.
