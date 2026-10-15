# matmul_tma

- Source file: `matmul_tma.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The new `FPCorrect.lean` contracts cover both original constexpr output branches,
all three block pointers, the complete single-tile dot and output store. Inputs
retain fp16 storage tags; output cells are fp32 or fp16. Every result is defined
from initial memory, and off-tile cells are framed. The original arbitrary
strides, input/output aliasing and output-address injectivity are retained.
`FPShape.source_projection` checks both original algorithm projections after
floating annotation erasure. The old mathematical statements remain unchanged.

Three levels are explicit: an arbitrary typed dot/backend contract; concrete
binary16 output rounding with a supplied dot; and a completely specified
software dot (half decoding, separately rounded fp32 products/additions,
ascending K order, positive-zero seed). The latter is a software profile, not
an equality theorem about tensor-core instructions. `FPWitness` checks a legal
16×16 tile, transposed A strides, A/output aliasing, output row gaps, both output
formats, and a nonidentity half rounding example through actual source execution.

Native obligations remain: binding the exact fp16×fp16→fp32 dot instruction and
its accumulation order/precision, subnormal/NaN policies, fp16 output conversion,
block-pointer address/allocation/launch and compiler scheduling. The original
Python kernel was compiled offline for SM90 in both output formats; it lowers
to WGMMA with fp32 accumulation and an additional half conversion when requested.
There was no GPU execution. Compilation does not establish equivalence to the
sequential software dot. Review artifacts are in
`bench/review-checkpoints/2026-10-14/next-value-repair/matmul-tma/` and
`matmul-tma-native/`.
