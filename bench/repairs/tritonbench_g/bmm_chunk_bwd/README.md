# bmm_chunk_bwd

For each actual output lane, the finite Dout-transpose times A sum over the true shortened chunk length, plus the optional initial residual.

- [bmm_chunk_bwd_repaired.py](bmm_chunk_bwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original three-axis source grid and all nine configurations for the selected FP32 A/Dout mathematical branch; strided/tail masks and optional residual remain explicit.

## Conditions

Public conditions:

- h.aStorage=.fp32 and h.doutStorage=.fp32
- RealViews.OutputDistinct
- HostGrid.InputsSeparate
- HostGrid.SeparateWrites over PublicIO.jobs
- PreparedAlgorithmProgramIO retains actual source projection, complete schedule, all access bounds, successful execution, active outputs and exterior frame

Contract domain source:

Contract and HostContract retain the direct FP32 caller-buffer ABI, shape/stride checks and physical input metadata. PublicIO.inputMask distinguishes full Dout chunk reads from shortened residual/store masks; no new dtype or host acceptance work is introduced.

Exact FP program pairs and numerical dependencies are recorded in the [bmm_chunk_bwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
