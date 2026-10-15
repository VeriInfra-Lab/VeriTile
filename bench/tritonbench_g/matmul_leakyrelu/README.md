# matmul_leakyrelu

- Source file: `matmul_leakyrelu.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The new `FPCorrect.lean` distinguishes two full floating-point graphs. The
`FPValue` graph preserves the Python two-argument dot followed by a separately
rounded fp32 add and the source activation `0.01 * accumulator`. The `FPFused`
graph describes the observed compiler candidate: live-accumulator dot and
`accumulator * 0.01`. Both retain the complete dynamic `cdiv(K, BLOCK_K)` loop,
grouped program mapping, wrapped input rows/columns, every padded K lane, and
the genuine output mask. Backend contracts support five explicit floating
input-storage tags; the closed software instance selects fp16 inputs/output
and fp32 arithmetic. No finite-input restriction or rounding-identity law is
added. Output layout uses the original `stride_cn = 1`, `BLOCK_N ≤ stride_cm`.

Each graph has an independent initial-memory recurrence, successful execution,
active-output values, and frame preservation. The software instances decode
half inputs exactly, use ascending fp32 FMA within each dot, retain their
respective inter-block accumulation order, round decimal `1/100` to fp32,
perform ordered comparison/multiplication, and narrow the result to half.
Their value recurrences are independent of arbitrary unused callbacks. The
original exact-multiple domain `K = BLOCK_K * numKBlocks`, `BLOCK_K > 0` is
included; `FPShape.original_trip_count` establishes its original trip count.
The old public Lean source and Python are unchanged. The erasure bridge is a
source-layout check, not a floating-value equality.

Offline SM90 compilation of the actual extracted Python (AST unchanged), with
activation on/off and fp fusion on/off, finds `tt.dot` with a live accumulator
in all four configurations. Activated TTIR also puts the slope on the right.
No GPU was run. The native MMA order, precision, FTZ/NaN behavior, conversion
policy, and integer/address/grid realization remain explicit binding duties.
In particular, the original-source software proof cannot certify the fused
compiler graph. `FPGraphDifference` gives three nonempty K blocks whose
software executions produce half `+0` versus half `-1`, respectively. That
counterexample assigns neither answer to a GPU.

`FPWitness` and `FPFusedWitness` additionally exhibit nonempty multi-block
rounding, both activation branches, masked-output framing, negative-zero and
NaN activation behavior. Artifact hashes, exact declarations, complete build
logs, and compiler evidence are delivered separately for registry review;
these files do not change canonical completion statistics.
