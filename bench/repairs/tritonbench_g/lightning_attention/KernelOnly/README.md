# lightning_attention

Independent complete causal forward and backward VJP formulas; LogicalDerivatives remains conjoined.

- [lightning_attention_repaired.py](../lightning_attention_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete forward grid and genuine intra→inter backward dataflow. All three gradient channels remain derivatives of the independent scalar loss through the retained LogicalDerivatives conjunct.

## Conditions

- The unchanged Host.Contract geometry remains public on both Real specifications.
- IO keeps actual source safety/success, typed outputs, scratch/dataflow and exact exterior frames.
- No finite-only input restriction, numerical rule, hardware conformance or new optional host requirement is introduced.

Exact FP program pairs and numerical dependencies are recorded in the [lightning_attention contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
