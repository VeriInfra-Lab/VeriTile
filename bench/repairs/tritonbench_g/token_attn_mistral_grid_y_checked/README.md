# token_attn_mistral

sum n in range xs.attention(job), if n+max(xs.sequence(job)-window,0)<xs.sequence(job) then probAt xs job n * valueAt xs job n d else 0

- [token_attn_mistral_grid_y_checked.py](token_attn_mistral_grid_y_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Selected grid_y_checked revision: every submitted work job and all output dimensions. Original distinct probability and page/value masks retained, including padding/final block behavior. A submitted work list is not advertised as arbitrary full B×H coverage.

## Conditions

- 0<p.block for Real result
- Grid.Separate p and Grid.Unique p work
- Ports.Domain: actual attention metadata equals counts
- Ports.span derives finite capacity from submitted counts and block; no additional maximum-length restriction
- Contract retains original integer contents, source strides/read aliases and selected gridY guard

Exact FP program pairs and numerical dependencies are recorded in the [token_attn_mistral contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
