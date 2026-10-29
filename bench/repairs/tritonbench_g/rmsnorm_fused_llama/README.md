# rmsnorm_fused_llama

xs row col * sqrt ((sum k:Fin columns, (xs row k)^2)/columns + epsilon)^(-1) * weights col.

- [rmsnorm_fused_llama_repaired.py](rmsnorm_fused_llama_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete row schedule for either selected source or index candidate. Real allows arbitrary/empty row and column dimensions and multiple source blocks.

## Conditions

Explicit conditions:

- candidate:Bool, named X/Y/W regions, rows/columns, epsilon
- X≠Y and W≠Y; X/W aliases allowed
- Fixed source block 16384 is positive
- RegionalScheduledImplements explicitly uses IO.schedule rows rows and retains contextual input/output windows, safety, success, every output and exterior frame.

Contract correspondence:

Wrapper Contract retains positive columns≤16384 and original tensor/epsilon predicates. The Real kernel theorem intentionally proves a broader mathematical domain with both source loops; no accepted wrapper input is dropped. FP main retains its explicit positive/limit conditions.

Exact FP program pairs and numerical dependencies are recorded in the [rmsnorm_fused_llama contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
