# rmsnorm_implementation

(xs k / sqrt ((sum j:Fin n, (xs j)^2)/n + eps)) * weights k.

- [rmsnorm_implementation_repaired.py](rmsnorm_implementation_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete selected logical row; identity singleton launch retains initial program IDs, leading strides and both source loops. Width n is arbitrary/empty; source blockSize=4096 loops cover all n.

## Conditions

Explicit conditions:

- All original batch/row/column/weight/output strides and n/eps are parameters
- OutputUnique n sok (injective active output addresses) only
- RegionalScheduledImplements oneProgram keeps original contextual input binding and physical windows, source safety, successful execution, all active outputs and exterior frame.

Contract correspondence:

Contract.mathematical is the independent finite unpadded RMS expression. IOResults.mathematical_formula proves its finite-channel equality. Singleton/empty domains permit zero output stride where OutputUnique holds; no arbitrary positive-stride assumption added.

Exact FP program pairs and numerical dependencies are recorded in the [rmsnorm_implementation contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
