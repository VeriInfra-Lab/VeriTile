# kcache_copy_triton

The current multi-token extension is [SignedCorrect.lean](SignedCorrect.lean):
`all_tokens_five_format_raw_bit_correctness`. It proves the complete registered
signed source for arbitrary `n_tokens`, all three program coordinates, and both
legacy and split-x cache layouts. The earlier n1 theorems below remain intact.

[SignedCopy](Proofs/SignedCopy.lean) proves exact type-erased equality with
`copy_to_kcache_seqlen_n_signed_kernel`, successful execution, complete memory
agreement with an independent initial-memory scatter, per-lane readback, and
framing. It retains the negative token shift, signed truncating division and
remainder, and the source's explicit uint64 address casts. The token-shift
identity is `token % n_tokens - n_tokens`; it is not Nat subtraction.

[SignedRawBitCopy](Proofs/SignedRawBitCopy.lean) transfers that result through
the existing lossless raw-word encoding. All fp16, bf16, fp32, f8e4 and f8e5
payloads are covered. The complete output memory is independent of every
floating callback, initial registers and undef values. No numerical assumption
is required for this new copy theorem.

[DiscreteCorrect.lean](DiscreteCorrect.lean) adds
`all_tokens_discrete_correctness` for int8/16/32/64, uint8/16/32/64 and bool
data. [SignedDiscreteCopy](Proofs/SignedDiscreteCopy.lean) instantiates the same
complete signed source and proves exact copying of every finite payload,
including the entire output memory and off-write frame. Signed words use their
two's-complement integer interpretation, unsigned words their natural-number
interpretation, and bool its one-bit interpretation. These are data-storage
profiles; the theorem does not add a native metadata-width or address-ABI claim.
This extension also requires no numerical callback assumption.

[SignedWitness](Proofs/SignedWitness.lean) checks an n=2 launch across a block-table
boundary, with a nonzero sequence/split coordinate and source/cache aliasing.
It preserves a NaN payload and negative zero at the independently computed
cache offsets. The old-layout corollaries set the split program coordinate to
zero; the general theorem retains every split-x coordinate and stride.

This closes the previous multi-token *single-program signed-model* gap. It does
not yet establish the entire Python wrapper: host launch composition, every
unrestricted caller dtype, concrete int32 metadata/overflow, byte addresses,
and cross-program collision/scheduling still require explicit binding. The
signed transcription and its uint64 address materialization are the source
boundary; no Python/compiler refinement is inferred from type erasure.

The accepted `n_tokens = 1`, parameterized splitX copy now has an explicit raw-bit software contract in [RawBitCorrect.lean](RawBitCorrect.lean): `five_format_raw_bit_correctness`. It covers fp16, bf16, fp32, f8e4 and f8e5 with their actual payload widths.

[RawBitMemory](Proofs/RawBitMemory.lean) defines independent tagged words and a lossless encoding. [RawBitCopy](Proofs/RawBitCopy.lean) proves that the actual full source finishes with exactly the encoding of an independent initial-memory scatter. The result includes per-lane readback, the complete off-write frame and canonical output representation. No numerical callback, finiteness condition or assumed output equality is left in the public statement.

All previous Nat parameters and alias cases remain, including zero width, zero strides and the model's division/modulo-zero behavior. Initial registers, undef and unused launch dimensions are arbitrary. [Witnesses](Proofs/RawBitWitness.lean) check a nonempty overlapping copy with NaN payload and negative zero, zero-width success and the limits of the reverse encoding law.

Actual fp32 loads use `.real` with a BitVec32 payload; the separate Structural `.fp32` tag is preserved in the full memory model. The old float64 annotation also uses `.real`. Its arbitrary-carrier and BitVec64 representation results are retained separately; explicit fp64 loads remain unsupported by the interpreter, so no native64 profile is claimed.

The earlier [opaque-carrier contract](FPCorrect.lean), `TypedCopy.source_projection` and [original Python](kcache_copy_triton.py) remain unchanged. The projection recovers the selected n1/splitX source without dropping metadata lookup or address expressions. The new reference uses the initial memory even when input, output and metadata regions overlap.

The historical n1 result is a typed, word-addressed software model. Byte packing, alignment, pointer overflow, machine int32 metadata, cross-program concurrency and GPU/compiler memory correspondence remain outside these proofs. There is no GPU experiment in this update.

Four new modules were independently compiled and 29 named axiom checks passed. See the [reviewed evidence and replay](../../review-checkpoints/2026-10-15/kcache-raw-bit-closure/README.md) and [current acceptance](../../current_source_acceptance.md). Existing `kcache_copy_triton.+` registration includes the new modules; no explicit duplicate registration is needed.
