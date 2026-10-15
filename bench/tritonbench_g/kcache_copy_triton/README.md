# kcache_copy_triton

The accepted `n_tokens = 1`, parameterized splitX copy now has an explicit raw-bit software contract in [RawBitCorrect.lean](RawBitCorrect.lean): `five_format_raw_bit_correctness`. It covers fp16, bf16, fp32, f8e4 and f8e5 with their actual payload widths.

[RawBitMemory](Proofs/RawBitMemory.lean) defines independent tagged words and a lossless encoding. [RawBitCopy](Proofs/RawBitCopy.lean) proves that the actual full source finishes with exactly the encoding of an independent initial-memory scatter. The result includes per-lane readback, the complete off-write frame and canonical output representation. No numerical callback, finiteness condition or assumed output equality is left in the public statement.

All previous Nat parameters and alias cases remain, including zero width, zero strides and the model's division/modulo-zero behavior. Initial registers, undef and unused launch dimensions are arbitrary. [Witnesses](Proofs/RawBitWitness.lean) check a nonempty overlapping copy with NaN payload and negative zero, zero-width success and the limits of the reverse encoding law.

Actual fp32 loads use `.real` with a BitVec32 payload; the separate Structural `.fp32` tag is preserved in the full memory model. The old float64 annotation also uses `.real`. Its arbitrary-carrier and BitVec64 representation results are retained separately; explicit fp64 loads remain unsupported by the interpreter, so no native64 profile is claimed.

The earlier [opaque-carrier contract](FPCorrect.lean), `TypedCopy.source_projection` and [original Python](kcache_copy_triton.py) remain unchanged. The projection recovers the selected n1/splitX source without dropping metadata lookup or address expressions. The new reference uses the initial memory even when input, output and metadata regions overlap.

This is a typed, word-addressed software model. Multi-token execution, byte packing, alignment, pointer overflow, machine int32 metadata, cross-program concurrency and GPU/compiler memory correspondence remain outside this proof. There is no GPU experiment in this update.

Four new modules were independently compiled and 29 named axiom checks passed. See the [reviewed evidence and replay](../../review-checkpoints/2026-10-15/kcache-raw-bit-closure/README.md) and [current acceptance](../../current_source_acceptance.md). Existing `kcache_copy_triton.+` registration includes the new modules; no explicit duplicate registration is needed.
