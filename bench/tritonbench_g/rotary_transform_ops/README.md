# rotary_transform_ops

- Source file: `rotary_transform_ops.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Concrete fp32 correctness

`FPCorrect.lean` exposes `rotary_transform_ops_fp32_correctness`, the task-specific
source projection, and a nonempty software rounding witness. This task's
selected paired-row and fixed/varlen metadata-row statements are identical to
`rotary_transform`'s corresponding selected statements. `Proofs/FPStages.lean`
proves the complete task-specific source correspondence and reuses that typed
execution and independent output specification. It does not equate the source
with itself as an output-correctness argument.

Python evidence: `rotary_transform_ops.py:61–92` contains the four explicit fp32
register casts, separate multiply/add/subtract expressions and both stores;
lines 216–218 and 225 explicitly create fp32 inputs and tables. The new profile
retains fp32 storage and all register casts. Every scalar operation uses
`FP.Software.scalarFP32` and concrete `FP.mul` / `FP.sub` / `FP.add` rounding.

All three existing selected forms are covered without narrowing their stage:
ordinary paired row, fixed-length metadata row, and variable-length metadata
row. Output values are derived from initial-memory inputs, both halves are read
back, and all cells outside both active write sets are unchanged. The only
address gate is the original nonzero output component stride. No numerical
law or output-value premise is added; input/output aliasing and masked tails
are retained. The witness starts from empty registers and a different output,
and proves that both `1 ± 2^-25` round to bits `0x3f800000`, with a masked tail
unchanged, for all three entry forms.

As before, full 2D/interleaved/offset-tensor/conjugate configurations have only
their existing lowering coverage where no value theorem was selected. The
shared unary-minus lowering `0 - x` does not implement a sign-bit flip at
positive zero, so it cannot justify the conjugate path. CUDA compiler behavior,
contraction, FTZ, NaN policy, and machine addresses require a separate native
refinement. See the sibling `rotary_transform/README.md` for the common exact
operation tree and proof decomposition.

Validation: directed Lean compilation and `#axiomsClean` passed for the helper
and public entry. Logs: `/tmp/rotary-transform-ops-{stages,public}.log`.
