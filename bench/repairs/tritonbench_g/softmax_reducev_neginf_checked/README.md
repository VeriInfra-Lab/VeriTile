# softmax_reducev

For batch b/head h/dimension d: sum_{k<counts[b]} mass(logit[b,h,k])*value[b,h,k,d] / sum_{k<counts[b]} mass(logit[b,h,k]), with mass(-∞)=0.

- [softmax_reducev_neginf_checked.py](softmax_reducev_neginf_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original workset (every batch/head) and arbitrary permutations, all gathered keys according to logical counts and typed metadata. IO extent uses maximum of counts, not source maximum. Original empty/alias behavior and all output dimensions retained.

## Conditions

- work.Perm(Host.workset c).
- PublicIO.RealDomain=LengthsMatch ∧ Normalizable: typed lengths equal counts; each active row has at least one finite logit and finite gathered values.
- Input/result masks, typed starts/pages/capacities and actual invocation bindings remain in IO.lean.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [softmax_reducev contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
