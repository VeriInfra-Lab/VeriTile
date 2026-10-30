# token_softmax_bloom

fun xs job i => exp(xs.values job i) / sum (j : Fin(xs.length job)), exp(xs.at job j.val)

- [token_softmax_bloom_repaired.py](token_softmax_bloom_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every submitted segment/head in original work schedule, with full logical segment coverage under realDomain.Cover. Repeated identical PID launches remain admitted by original Unique; no new full external host-grid application.

## Conditions

- 0<p.width
- Grid.Separate p
- realDomain=Unique ∧ Cover; each runtime logical segment length≤width
- Original shared token_softmax_llama Contract aliases retained
- ProgramIO preserves finite metadata/input/output channel bindings, safe successful schedule and frame

Exact FP program pairs and numerical dependencies are recorded in the [token_softmax_bloom contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
