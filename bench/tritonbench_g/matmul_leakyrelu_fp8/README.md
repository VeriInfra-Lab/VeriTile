# matmul_leakyrelu_fp8

- Source file: `matmul_leakyrelu_fp8.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The new `FPCorrect.lean` covers the complete dynamic grouped matmul stream,
including every K block and padded lane, wrapped input row/column addressing,
optional leaky ReLU, fixed half conversion, and the genuine output tail mask.
The original Python uses three-argument dot with the live fp32 accumulator.
Its tests supply half inputs, and its host allocates half output despite the
benchmark name. Backend contracts expose five explicit input-storage tags;
closed software contracts select either half or E4M3FN inputs, always with
fp32 accumulation and half output. No scale factors are invented.

The independent value recurrence starts at fp32 +0 and processes the actual
`cdiv(K, BLOCK_K)` trips. The closed software dot decodes the selected narrow
format and uses an ascending per-K fp32 FMA sequence with the live seed. The
activation retains ordered comparison, decimal 1/100 rounded to fp32, and its
rounded multiplication. Final half narrowing is concrete. The value proof is
independent of arbitrary unused callbacks; no global cast identity, finite
input assumption, or arithmetic reassociation law is required. The original
output-layout conditions `stride_cn = 1`, `BLOCK_N ≤ stride_cm` remain the only
new public layout premises. `FPShape.original_trip_count` includes both old
public exact-multiple domains; their source file and statements are unchanged.

`FPValue` preserves the source's `slope * accumulator`. `FPFused` separately
models the observed compiler candidate's `accumulator * slope`. Both use live
dotAcc, and neither backend contract assumes multiplication commutes. The
original autotune configuration (128×256×64, group 8, 8 warps, 3 stages) was
offline-compiled for SM90 with half/E4M3FN inputs, activation on/off and
fp fusion on/off. All eight builds preserve live dot and the activated TTIR puts the slope on the
right. No GPU was run. Actual tensor-core order/precision, FTZ/NaN policy,
conversion policy, typed memory, and integer/address/grid realization still
need the separately recorded native binding before claiming hardware bits.

Both graphs have successful nonempty witnesses with output framing. Half
witnesses show two-block rounding and negative/NaN/negative-zero activation
cases. E4M3FN witnesses decode 64 and its minimum subnormal, then show
`4096 + 2^-18` rounding to fp32 4096 and a finite half output. The erasure bridge
checks only the original graph's source layout. Hashes, compiler artifacts,
exact declarations, and build receipts are delivered for registry review;
canonical completion statistics are not changed by these task files.
