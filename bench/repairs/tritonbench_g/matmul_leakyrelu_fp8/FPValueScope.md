# Floating-point scope

The selected source profile uses FP32 input storage and a supplied FP32 dot
accumulator, followed by an explicit FP16 output cast. No source input cast,
AMP branch or FP8 dispatch exists. The historical FP16 fixture is outside
the checked input contract. Every FP32 input word is admitted, including
signed zeros, subnormals, infinities and NaNs.

The independent reference reads caller coordinates from pre-call memory.
For positive K it composes every BK64 iteration in source order, with eight
fixed K8 instruction result-lane operations per block and explicit masked
zero operands. It applies the selected scalar activation, then packs half.
K=0 follows the actual `zero_()` branch and writes literal positive zero.
The reference contains no launch result or final-memory value.

Exactly four numerical correspondences remain deferred in `LocalPorts`:

- `cast`: nearest-even FP32-to-FP16 packing, including exceptional values.
- `comparison`: ordered `x >= 0`, false for NaNs and true for either zero.
- `slope`: FP32 IEEE multiplication with the source literal on the left,
  whose FP32 word is `0x3c23d70a`.
- `mma`: one result lane of the selected fixed K8 ISA profile, taking two
  `Fin 8` operand vectors and one scalar accumulator.

The selected instruction's TF32 conversion, arithmetic order, rounding,
exceptional-value and FTZ behavior require native evidence. The new public
theorem parameterizes that fixed instruction profile; its arity cannot hide
arbitrary K, a full reduction, activation vector, kernel or output. The
historical offline SM90 instruction evidence remains in the conditional
stack. No unsupported explicit TF32 input conversion is inserted into the
checked source model.

The software cast and slope reference reuse the existing reviewed half
codec and IEEE FP32 value operations. Comparison is defined by NaN, sign
and zero bits. In the leaky branch the slope event is evaluated before the
selection, including when the nonnegative arm is selected. The Real
interpretation independently proves dense matrix multiplication followed
by Real leaky-ReLU or identity. Neither the FP theorem nor the CPU controls
assert native equality to that Real target.

The complete checked call derives validation, source iteration, masks,
all grid stores, zero-fill, finite autotune replay, output identity and
successful owner restoration from local events and ordinary allocation
laws. A raw byte observer independent of typed words is connected by a
one-cell codec; a one-event frame derives caller backing preservation on
every ordinary outcome, including partial execution and context failures.
Physical input bindings and active byte-pointer bounds account for offsets,
broadcast views and trailing backing bytes.

Remaining interpretation boundaries are the manual pinned source/event
correspondence; raw Python argument/descriptor extraction and grouped
metadata checks rather than exact exception-label ordering; ordinary
Tensor identity, bounds and storage laws; allocator metadata, freshness and
caller-byte preservation; and local event memory/codec/runtime laws. The
new public path does not assume the historical whole-program
`Resident.CompilerCorrect` predicate.

The 43 CPU source/reference/frame cases and 10 CUDA FakeTensor host cases
are finite structural evidence. No native GPU arithmetic conformance was
run. The current obligations are listed in `numerical-obligations.json`.
