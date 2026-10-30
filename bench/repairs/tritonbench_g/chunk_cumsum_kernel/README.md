# chunk_cumsum_kernel

For logical output time t, the inclusive prefix sum over all k in range T with k≤t of the initial logical input.

- [chunk_cumsum_kernel_repaired.py](chunk_cumsum_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One arbitrary source row/program ID, complete dynamic time loop over all T and every masked tail; this is a row theorem, not an asserted simultaneous host-grid schedule.

## Conditions

Public conditions:

- output≠input
- Config retains its original positive source block configuration
- UKernelIO regional input/output windows cover the whole logical row; masks, source access bounds, successful complete loop, active outputs and exact regional frame remain

Contract domain source:

Contract.mathematical is the same independent inclusive prefix. Contract retains FP32, optional FP32 output, rank-3 contiguous CUDA input, row-grid and alignment conditions. The public theorem is row-parametric and adds no host/native acceptance obligation.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_cumsum_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
