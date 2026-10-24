# quantize_kv_copy

`FPCorrect.lean::quantize_kv_group_fp32_int8_correctness` covers the original
accepted grouped quantizer in the supplied-test storage profile: fp32 K,
fp32 OutScale and signed-int8 Out (`quantize_kv_copy.py`, lines 206–209 and
220–223). The complete grouped stage executes, including source/destination
indexing, masked loads, abs, maximum, rounded scale, rounded quotient, integer
conversion and both masked stores. Dimensions, group count and strides remain
symbolic; the reduction width must be positive. The load/address/mask prefix
has an exact projection to the old accepted source. The later abs and cast
nodes deliberately replace the old Real projection with the corrected typed
operations; `source_abs_node` checks the actual fp32 lowering.

The software profile carries real 32-bit encodings. Abs clears only the sign
bit, maximum selects an original abs-transformed bit pattern and proves that
its encoded comparison rank dominates every input, and each fp32 division
rounds separately under the explicit `FP.Config`. The divisor is the exact
fp32 encoding of 127. The signed-eight-bit cast decodes the quotient's exact
rational value, truncates toward zero and checks the signed range. Neither
identity casts nor Real arithmetic stand in for machine operations.

The active-lane domain is explicit: each computed fp32 quotient has a finite
rational projection and a representable signed-int8 truncation. This excludes
undefined conversions such as a zero-scale row's 0/0 at an active store; it
is not an assumption about final execution or output equality. No condition
is placed on padded groups. An arbitrary undefined-conversion policy is
universally quantified, and complete observable memory is proved independent
of it. Output addresses must be injective among active lanes, and the value
and scale output regions are distinct, as in the host allocation. Readback
and the memory frame cover both outputs.

A nonempty witness executes one active group and one padded group: source
values 127 and -63.5 produce scale 1 and int8 values 127 and -63; both inactive
output sentinels remain unchanged. These ground bit operations and all named
proofs are kernel-checked with the standard axiom base.

Native GPU conformance remains a separate binding obligation: reduction
lowering and NaN handling, division rounding/flush configuration, source store
and cast lowering, and the accepted Nat address abstraction. The maximum
interpreter has a deterministic ordering for NaN payloads, but this does not
claim the GPU follows that ordering. No GPU experiment was run. The older
mathematical statements remain visible through the explicit
`legacy_real_int8%` compatibility projection and receive no new FP credit.

```sh
lake env lean bench/tritonbench_g/quantize_kv_copy/Proofs/TypedQuantize.lean
lake env lean bench/tritonbench_g/quantize_kv_copy/Proofs/TypedShape.lean
lake env lean bench/tritonbench_g/quantize_kv_copy/Proofs/BitWitness.lean
lake env lean bench/tritonbench_g/quantize_kv_copy/FPCorrect.lean
```
