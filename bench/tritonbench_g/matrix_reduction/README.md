# matrix_reduction

- Source file: `matrix_reduction.py`
- Corpus: TritonBench-G v1
- Status: DSL port compiles; see `../completion_audit.md` and `../proof_blockers.md` for audit and proof status.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

## Typed bit contract

`FPCorrect.matrix_reduction_machine_correctness` proves successful execution of the complete row-max source and an independent maximum-value/memory-frame specification for both fp16 and fp32. It covers symbolic matrix dimensions and strides with nonempty rows and non-NaN inputs; infinities, subnormals, and signed zeros are included. Output words are selected original input words, and equal signed-zero values may retain a selected input's sign.

The backend carrier holds actual binary32 encodings. Half storage words are encoded as16-bit values in memory, then widened by exponent rebiasing and significand normalization; the reduction runs on the resulting32-bit encodings. The store narrows a selected widened word back to its exact original16 bits. `Proofs/BinaryConversion.lean` proves round-trip, comparison-order, sign, infinity, and exact finite-value preservation from the format fields. These are kernel-checked algebraic proofs. Image-domain narrowing is sufficient because this kernel introduces no new arithmetic values; arbitrary binary32-to-half rounding is outside this contract.

This is a source interpreter result. Compiler/device refinement, including the native unindexed-max signed-zero choice and machine address/index arithmetic, remains a separate binding obligation. `fp_source_evidence.json` records source provenance and scope. The public contract uses actual32-bit intermediate encodings.
