# batched_vecmat_mult

For every (m,n), sum over all k : Fin K of realValue(decode,A[m,k])*realValue(decode,B[m,n,k]); arbitrary logical source expressions retain their original decoding.

- [batched_vecmat_mult_int64.py](batched_vecmat_mult_int64.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every M by N output of the complete selected floor grid under the explicit divisibility conditions; full logical K, not the block width or a truncated tail.

## Conditions

Public conditions:

- Whole.IntegerAgreement p
- SourceGraph.InputsSeparate p
- 0<BM and 0<BN
- BM divides M, BN divides N, BK divides K
- The observer and ProgramIO input binding, access bounds, success, all outputs and exterior frame remain unchanged

Contract domain source:

Contract classifies nonnegative builtin scalar arguments. The inherited HostPrepare successful launch conditions and the actual Python assertions K%block_k=0, M%block_m=0, N%block_n=0 supply the divisibility conditions; they remain explicit in the theorem. A zero Python divisor does not produce a successful launch.

Exact FP program pairs and numerical dependencies are recorded in the [batched_vecmat_mult contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
