# quant_transpose_kernel

- Source file: `quant_transpose_kernel.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Typed pre-rounding contract

`FPCorrect.quant_transpose_scaled_store_fp32_correctness` covers exactly the previously accepted scaled-store stage. The source uses fp32 loads and two separately rounded multiplies, with exact fp32 constant127; the independent target is `mul(cfg, 127, mul(cfg, input, absmax_inv))`. The proof establishes successful execution, active-lane transposed readback, and preservation of every other tagged memory cell. Dimensions and strides remain symbolic, with the original output-address injectivity condition.

The software configuration controls rounding and denormal handling; `FP.Config.ieeeValue` selects nearest-even without flushing. The result does not claim native GPU conformance, the original grouped launch schedule, host absmax computation, or final `llrint` / int8 quantization. `Proofs/TypedShape.lean` proves exact dtype-erased correspondence with the existing scaled-store stage at its actual source constant127.
