# matmul_dequant_int4

The accepted public scope is the original standalone `dequantize_kernel`
(`matmul_dequant_int4.py:137–170`), with both packed arrays stored as int32 and
scale/output stored as fp16 in the actual tests. The separate matmul kernels
and host quantizer are outside this existing stage contract.

`Proofs/FPTyped.lean` retains the complete dequantize statement list, all four
address formulas, group lookup, masks, signed int32 loads, arithmetic right
shift and 32-bit `& 15`. It materializes the implicit integer-difference-to-fp16
promotion before multiplying by the fp16 scale. `PackedInputs` records actual
32-bit initial memory payloads, and active output addresses must be distinct.
These are input/representation conditions, not additional numerical laws.

The independent result decodes the two original 32-bit bit patterns, subtracts
the unsigned nibble values as signed integers, converts that difference to
binary16, and multiplies by the initial binary16 scale. The source's arithmetic
right shift is proved equal to this unsigned nibble interpretation at every
one of the eight positions, including negative packed words. Each difference
lies in `[-15,15]`, cannot overflow int32, and has exact binary32 and binary16
encodings. The retained fp32 evaluation context for its integer-promotion and
cast is recorded explicitly; the restricted profile supports that exact
31-value domain, so this compressed half carrier introduces no double rounding.
The load/store contract additionally supplies successful execution and
preservation of every cell outside the actual active writes.

`Proofs/Binary16.lean` supplies a concrete software format with 16 physical bits,
5 exponent bits, 10 fraction bits, bias 15, direct rational target-format
rounding, gradual underflow, infinity, NaNs and signed zero. It does not reuse
bf16 or assume a globally identity real-to-fp16 cast. Its boundary vectors and
all 31 possible integer differences are checked by Lean's kernel. The native
Triton/compiler correspondence remains the common external binding boundary.

The nonempty witness uses negative packed int32 words at nibble position 7,
producing difference `-7`, scale bits `0x3555`, and rounded output bits `0xc0aa`
at address 119, with address 120 and the source word preserved. The exact
rational product differs from the stored fp16 result.
