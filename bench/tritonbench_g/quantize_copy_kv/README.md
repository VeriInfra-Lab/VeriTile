# quantize_copy_kv

- Source file: `quantize_copy_kv.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

`FPCorrect.lean` adds two parameterized software-bit contracts for the complete
accepted per-destination quantizer. They execute the original head-only masks
and every `BLOCK_DMODEL` column, read all inputs from the initial memory, and
specify both output scatters and the full memory frame. Arbitrary source/destination
aliasing is preserved; the readback hypotheses are the original distinct output
regions, injective output/scale addresses, and nonempty reduction width.

The profile uses binary16 input and scale payloads, true half absolute value and
half/single conversion, fp32 division, and signed-byte conversion toward zero.
`FPMaxProfile` implements default maxnum: a numeric operand wins over one NaN.
Its ascending reduction order, canonical all-NaN result, and signed-zero policy
are explicit software choices. It does not reuse the old encoded-rank reducer
on arbitrary NaN inputs. A kernel-checked counterexample records their difference.

The first contract covers every input payload with an arbitrary byte-valued
policy for undefined integer conversion. The second strengthens exactly the
active, defined conversions and proves independence from that policy; inactive
heads need no finite-input or padding hypothesis. `FPWitness` checks a nonempty
run with positive and negative values, genuine half-scale rounding, and an
inactive head whose conversion is undefined.

These are software execution/value proofs, not GPU conformance results.
Compiler/device bindings remain for maxnum reduction scheduling and NaN/zero
bits, division/FTZ/rounding configuration, half casts, and defined signed-int8
conversion. No native result is claimed for undefined conversion. `FPShape`
checks the unchanged load/address prefix and both store tails against the old
source; the old mathematical producer and its public statements are preserved
separately. Per-program contracts do not establish cross-program race freedom.
