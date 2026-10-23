# Reachable composed-input INT32 saturation

Candidate source SHA-256:
`2c6ceae59a42edbb9ec315beae4bedf9a156aace6fae0c19b6f0a4cd06065047`.
The previous scalar `.to` failures are fixed; this is a separate remaining defect.

## Public input and independent target

Use `matmul_quantize_int8(fpa, b, b_scale)` with:

- `fpa`: contiguous FP32 shape `(1, 262144)`, every value `-127`;
- `b`: contiguous INT8 shape `(262144, 1)`, every value `-128`;
- `b_scale`: contiguous FP32 shape `(1,)`, value `2^-31`;
- ordinary separate storage on one CUDA device, default fresh FP16 output.

The actual composed wrapper and guards admit this input. Producer K is below
`2^20`; both producer and all candidate GEMM grids satisfy existing bounds.
The producer launch is `(1,)`, BM=1, BK=262144. Config 0 (8 warps, 2 stages)
compiles offline at this actual size. The native GEMM profile for these same
shape/stride/dtype arguments also compiles; scalar data are loaded from pointers
and are not JIT-specialized values.

The independent mathematical producer has max=127, scale=1 and stored code -127.
Thus the composed target is

```
262144 * (-127) * (-128) * 1 * 2^-31 = 127/64 = 1.984375.
```

The target is exactly representable in FP16. There is no final rounding excuse
for the result 1 below.

## Producer reachability without assuming exact division

The actual large producer PTX has `div.full.f32`, followed by
`cvt.rzi.s16.f32`, low-byte stores and FP16 scale conversion. NVIDIA's PTX
[div specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div)
states: “The maximum ulp error is 2 across the full range of inputs.”
The retrieved source and exact section text are pinned in the evidence packet.

All loaded values are exactly -127; absolute value and max are exactly 127.
For the live scale `s = div.full(127,127)`, a conservative enclosing interval is
`1-2^-21 <= s <= 1+2^-21`. It lies strictly inside the binary16 rounding cell of
1, so the scale actually stored and subsequently loaded by GEMM is exactly 1.
The live quotient is `div.full(-127,s)`. Its two-ulp error near -127 is at most
`2^-16`; the resulting interval is strictly between -128 and -126. Therefore
toward-zero signed16 conversion and low-byte storage give signed code -127 or
-126. Both possibilities below fail. No claim that `div.full` equals IEEE
round-to-nearest division, or that it must produce exactly -127, is needed.

## Actual selected GEMM and integer contradiction

Configuration 10 is BM32, BN32, BK128, group M8, SPLIT_K=1, 4 warps, 2 stages.
Its grid is `(1,1)`. The emitted instruction is

```
mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32
```

The destination registers also occur as incoming accumulator registers. There
are four K32 atoms per BK128 source step, for 2048 source steps: 8192 atoms per
output element. Every product is positive. Even if every producer byte is only
-126, the unbounded integer sum is `262144*126*128 = 4227858432`, greater than
`INT32_MAX = 2147483647`. For -127 it is `4261412864`. The signed saturation
recurrence reaches `INT32_MAX` in either case and stays there.

NVIDIA's PTX [integer MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
states that `.satfinite` limits overflow to `MIN_INT32 .. MAX_INT32`; without
that qualifier the result wraps instead. This candidate emits `.satfinite`.
The final operations are therefore:

```
float32(INT32_MAX) = 2147483648 = 2^31
float16(float32(2^31 * 1 * 2^-31)) = 1.
```

`Counterexample.lean` checks this integer recurrence and the exact FP32/FP16
conversion bits. It also proves `127/64 != 1`. The only nonexact scalar
instruction involved in producer reachability is covered by the documented
bound above, not deferred experimental validation.

Configuration 27 is the corresponding SPLIT_K=2 control. Each split has 4096
K32 atoms and its integer sum remains below INT32_MAX for both uniform codes.
Its two partials give 127/64 for code -127 or 63/32 for code -126. The packet
keeps these branches distinct. It does not assert a GPU autotuner selects any
particular configuration. A full-domain claim covering every eligible selected
configuration already fails at configuration 10; the guard still admits it.

## Status and limits

The actual admission/binder/compiler artifacts, documented instruction
semantics and symbolic arithmetic give a concrete diagnostic. There is no GPU
execution or whole-source execution theorem in this new packet. The existing
15 partial quoted-source modules remain frozen and applicable within their
stated models. In particular an ideal interpretation using unbounded integer
sums must not be reported as proof that the concrete INT32 computation avoids
saturation. Finishing a different, explicitly saturating specification would
not prove the independent full GEMM target stated here.

No source/input-domain repair for saturation is applied. No hidden K upper
bound, finite-value restriction or numerical assumption is introduced. The
new candidate remains a source-defect result, not whole-task correctness.
