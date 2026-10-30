# rbe_triton_transform

`FPCorrect.lean` supplies a **backend-value** contract for the complete existing
rotary pair surface. It retains fp16 input/output storage, fp32 frequency and
arithmetic scopes, both barriers, the distinct real/imaginary masks, actual
imaginary pointer offset `+1`, and both stores. Success, readback of an
independently defined initial-state result, and preservation of all other cells
are proved under the ordinary active-write address separation condition. The
nonempty backend witness has one pair, `DIM=128`, a nonzero token position,
input preservation, and an output sentinel.

This is not yet a closed software-bit or native-GPU result. The explicit
`Algebra` parameter must still be bound to an actual mixed-format backend.
The reference first computes the rounded/native equivalents of
`e = float32(column % DIM) / float32(DIM)`,
`power = pow(float32(theta), e)`, and
`angle = float32(row + startToken) / power`; it then computes
`fp16(fp32(real) * cos(angle) - fp32(imag) * sin(angle))` and
`fp16(fp32(real) * sin(angle) + fp32(imag) * cos(angle))` in source order.
An inactive imaginary load contributes the source's converted fp16 zero, so
odd trailing dimensions retain the actual masked-load behavior.

Remaining bindings include `libdevice.pow`, `tl.sin`, `tl.cos`, natural-index
and theta/DIM literal conversion to fp32, masked-zero construction,
fp16-to-fp32 promotion, fp32 arithmetic/division, and fp32-to-fp16 store
conversion. Existing concrete scalar add/sub/mul/div and fp16 widening results
can support that work, but this task does not yet connect them into a closed
mixed-format profile. The named `real` AST carrier inside a `some fp32` compute
scope is explicitly a backend callback tag, not a claim that real arithmetic
implements native fp32. The source bridge erases only types/materialized
implicit casts and proves equality with the complete existing statement list.
No real trigonometric identity or global identity rounding law is assumed.
