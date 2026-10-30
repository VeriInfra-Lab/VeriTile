# fused_rotary_embedding

fused_rotary_embedding_tile_checked.fused_rotary_real: rotaryIO implements Q/K rotation and V copy on logicalMetadataDomain.

- [fused_rotary_embedding_tile_checked.py](fused_rotary_embedding_tile_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected Q/K/V source grid including empty input, in-place Q, leader-only K/V work, both cache layouts, indirect int32 metadata and all output/frame cells.

## Conditions

Public conditions:

- Config.Valid provides even/valid geometry, cache addressing and write separation
- metadataDomain binds actual context lengths and only the selected block-table entries
- ProgramIO input memory bounds, safe actual accesses and exterior frame remain
- Original checked wrapper tile guard is retained in Contract

Contract and domain correspondence:

Contract reuses original checked configuration/metadata and adds only the already existing fullBlock bound on live work. Main metadata domain is an input-array/configuration relation; unused table entries remain unrestricted. No logical width is confused with half/full source tile width.

Exact FP program pairs and numerical dependencies are recorded in the [fused_rotary_embedding contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
