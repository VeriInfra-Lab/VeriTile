# layer_norm_ops

[SoftwareCorrect.lean](SoftwareCorrect.lean) adds nine parameterized software
value contracts for the existing fp32 stage profile: RMS/plain single-row
DX/DW(/DB), independent C1/C2 reductions, DX from supplied C1/C2, residual-add
dual stores, and supplied vector/mean/rstd stores. Each proves successful
execution, an independent initial-memory result, active output readback and
the outside-write frame. The older [FPCorrect.lean](FPCorrect.lean) contracts
remain available.

[FPClosed.lean](Proofs/FPClosed.lean) removes the arbitrary literal conversion:
the natural dimension N is converted through exact Rat and rounded to fp32
with nearest-even. For example, 16777217 rounds to 16777216; no N bound,
finite-input premise or output-equality premise is added. Every add/sub/mul/div
is concrete software fp32, and the full-block reduction uses ascending lanes
with a +0 seed. Masked W tails remain arbitrary fp32 values, including Inf/NaN;
their products are retained. DW/DB also retain the source's +0 additions.

This supports software classification for these nine stages. Mean/Rstd and
other supplied inputs remain supplied: full forward statistic producers,
multi-row backward/host aggregation and other storage dtypes are not covered.
GPU reduction order, fusion, division, constant conversion and FTZ/NaN behavior
still require native binding. See the [independently compiled closure package](../../review-checkpoints/2026-10-14/next-value-repair/layernorm-software-closure/README.md)
and [source-binding review](../../review-checkpoints/2026-10-14/next-value-repair/software-closure-source-binding/root-review.json).

- Source file: `layer_norm_ops.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.
