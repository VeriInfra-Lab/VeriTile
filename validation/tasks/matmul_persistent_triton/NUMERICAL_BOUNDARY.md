# Numerical boundary

The Real endpoint is the independent full K-sum over input Tensor storage.
The native endpoint is equality of raw storage words against an independent
reference of named, fixed-size ISA instructions and scalar conversions.

`Native.allWords` admits all stored words and all instruction operands. It
does not exclude NaNs, infinities, subnormals, signed zero, scalar one,
broadcast strides, or any supported shape. `LocalPorts` contains only:

1. One scalar output/storage conversion at its actual word.
2. One named fixed-size ISA operation at its actual two operand fragments and
   incoming accumulator.

There is no whole-dot, matrix-product, whole-output, host, allocator, or
complete-kernel numerical equality port. No local primitive numerical
experiment was performed.

The SM90 artifacts select
`wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16` for FP16 and
`wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3` for E4M3FN. A source
BK64/BK128 block expands into four ordered K fragments. `DotLayout` records
the named instruction and compiler-selected fragment order independently of
any numerical result. It covers fixed MMA variants as well; the endpoint is
parametric in that explicit selection.

The shared DSL's masked `other=0.0` first converts an FP32 positive-zero
literal to the input storage type. The operational proof retains that scalar
cast. `Native.Reference.positiveZero` states the independent reference's
scalar positive-zero conversion; `Native.padding_zero` derives the native
padding fact through the same individual conversion port. This is not a
matrix-output premise or a restriction on input words.

Instruction hardware semantics and compiler instruction selection remain
explicit trust boundaries. Offline compile/IR/PTX witnesses check the
reported selection, and do not constitute a machine-code correctness proof
or a GPU numerical validation.
