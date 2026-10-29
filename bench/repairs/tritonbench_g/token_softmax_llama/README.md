# token_softmax_llama

Full selected segment/head softmax over Fin(lengths segment).

- [token_softmax_llama_repaired.py](token_softmax_llama_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every selected segment/head PID in work; all active lanes of its complete segment, repeated identical launches allowed. This is the established selected schedule, not an asserted host-allocation result.

## Conditions

- 0<p.width; Grid.Separate p prevents output writes from corrupting input/start/length channels
- realDomain p work = Unique p work vals ∧ Cover p work vals; Cover supplies length[r]≤p.width, Unique identifies output PID/lane events
- Shared ProgramIO public judgment supplies actual channel binding, window bounds, source safety, success, output and full exterior frame

Exact FP program pairs and numerical dependencies are recorded in the [token_softmax_llama contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
