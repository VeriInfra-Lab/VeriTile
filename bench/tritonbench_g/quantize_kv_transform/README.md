# quantize_kv_transform

The new machine contract is the original accepted transform stage for both
supplied input formats, fp16 and fp32, with signed-int8 output and fp16 scales.
The stage keeps symbolic heads, dimensions, block sizes, strides, destination
indices and both original masks. It includes abs, reduction, scale division,
actual fp16 scale conversion, widening that rounded scale for the quotient,
checked signed-eight-bit conversion, and both stores. The complete load/address/
mask prefix projects to the prior accepted source for each input format.

`Proofs/MixedBits.lean` uses explicit 16-bit storage encodings and actual
32-bit intermediate encodings. Widening reuses the binary16-to-binary32 bit
conversion already proved in matrix_reduction. Narrowing decodes the binary32
value and rounds directly to the real binary16 format implemented by
`matmul_dequant_int4/Proofs/Binary16.lean`: 1 sign bit, 5 exponent bits and 10
fraction bits. It does not reuse bf16 as half and does not use the inverse-only
maximum-selection narrowing helper for general arithmetic. Subnormal, tie,
overflow, negative-zero and signaling-NaN abs vectors are kernel-checked.
Narrowing canonicalizes NaNs explicitly; native payload behavior is not inferred.

The output specification uses initial input memory and exact encoded operations.
The selected abs-transformed maximum has a source-bit membership and maximum
rank certificate. Each division rounds separately in fp32 under the stated
configuration; the half-scale rounding is a separate event. The final integer
specification gives the quotient's exact rational value, truncation toward zero
and representability in signed eight bits. Output and scale readback and a
frame cover both active write sets.

The numerical domain checks only active quantized outputs: their rounded
quotients must be finite and have representable signed-int8 truncations.
Undefined conversions remain arbitrary and cannot affect observable memory on
this domain. Padded heads and columns have no added numerical restrictions.
The reduction width is positive, active destination mappings are injective,
and the two output regions are distinct. Zero active scales can produce an
undefined 0/0 conversion, which is explicitly outside the active conversion
domain rather than silently assigned an integer result.

The software profile and source-level contract do not establish native GPU
refinement. Remaining bindings include max reduction and NaN behavior, exact
rounding/flush configuration, conversion/store lowering, and the accepted Nat
address abstraction. No GPU experiment was run. Existing mathematical claims
remain through the visible `legacy_real_int8%` projection and receive no new
FP coverage credit from that compatibility wrapper.
