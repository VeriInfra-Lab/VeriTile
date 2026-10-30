# Current kernel revision: source defect

The selected source is `iv_dependent_matmul_i64.py`, SHA256
`19aa1bc4e73dabdc494cbfe290114c8c7b317192194f845b9c0e4d09dec9e10b`,
as recorded in `task_progress.json` under `contract_revision`. This review leaves
that source, its guards, all historical proof files and the selection unchanged.

The mathematical target is ordinary matrix multiplication:
`C[i,j] = sum(k=0..K-1, A[i,k] * B[k,j])`. The source allocates FP32 A/B/C,
accumulates in FP32 with the default TF32 dot profile, and explicitly narrows the
final accumulator to FP16 before widening into C. The previous instruction model
keeps live-seed K8 MMA accumulation separate from a separately rounded Dot+Add
graph. This review introduces no reassociation, precision change, floating-point
experiment or numerical deferral.

The wrapper admits nonnegative integer dimensions, five pointer schedules, and
positive power-of-two BM/BN/BK values up to `2^20`, with explicit signed-control,
active-byte and grid bounds. It does not enforce the dot frontend's minimum K
tile size.

For `M=K=N=BM=BN=BK=1`, default FP32 and CUDA, every wrapper guard succeeds,
the grid has one program and the kernel reaches its dot operation. On the pinned
Triton 3.7.1 NVIDIA frontend, FP32 dot operands require tile dimensions at least
`(M,N,K)=(1,1,16)`. Actual offline SM80 compilation of the selected kernel fails
for each of the five schedules with:

```text
Input shapes should have M >= 1, N >= 1 and K >= 16
```

The source line is `accumulator += tl.dot(a, b)`. The same `M=K=N=BM=BN=1`
call with `BK=16` passes the guards and compiles successfully. Small BM/BN are
therefore not the cause of this counterexample.

`Proofs/KernelDefect.lean` proves that the checked contract accepts the exact
live counterexample for every schedule while violating that necessary frontend
condition. `Proofs/check_block_defect.py` checks 30 actual-wrapper guard prefixes
using fake integer metadata, then performs the five failing compilations and one
successful control. It allocates no tensors and executes no floating-point
operations or GPU kernels. Compiler-source hashes, diagnostics, successful-control
IR and the current Lean receipt are pinned in the delivery packet.

This is an admitted-source failure, so an additional implicit “supported block”
premise would narrow the current domain. The current whole kernel task is
classified `source_defect`; the historical conditional results remain available
within their explicit premises. No new complete kernel/Real/native/actual-Tensor
equivalence is claimed for all inputs admitted by this selected source. A future
repair must state its domain change explicitly and preserve the selected
precision. This review does not claim the minimum K bound alone is sufficient
for every other supported block/device combination.
