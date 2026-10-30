# softmax_flaggems

Forward: mass(q[o])/sum_{n<d.N} mass(q[row,n,col]), mass(-∞)=0 and mass(x)=exp(x). Backward saved-output VJP: p[o]*(g[o]-sum_{n<d.N} p[row,n,col]*g[row,n,col]).

- [softmax_flaggems_repaired.py](softmax_flaggems_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](KernelInterface/Correct.lean)
- [FPEquiv.lean](KernelInterface/FPEquiv.lean)

## Scope

Real: actual full M×N×K grid for both distinct forward/backward results. FP: complete selected source kernel for each inner/noninner and one/multiple-tile branch, not an invented host result.

## Conditions

- Positive N,K,B,C; c.one=oneTile d.N c.B. Forward X≠Y; backward P≠Y and G≠Y.
- ForwardDomain: at least one finite logit in each logical N-column; remaining values may be -∞. This mathematical interpretation is separate from wrapper all-word admission.
- IO channels cover count=M*N*K, actual forwardJobs/backwardJobs enumerate the original grid; total extension only serves valid logical addresses.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [softmax_flaggems contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
