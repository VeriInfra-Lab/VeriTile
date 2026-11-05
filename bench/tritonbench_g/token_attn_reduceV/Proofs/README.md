# token_attn_reduceV

The selected repair is [token_attn_reduceV.py](../token_attn_reduceV.py). Both
`source` and the existing probability-address candidate `affineSource` in
[Kernels.lean](../Kernels.lean) contain complete copied function bodies. Invocation
types and runtime scalar bindings precede each `triton { ... }` quotation.
[token_attn_reduceV.py](../token_attn_reduceV.py) is the existing Python candidate;
its only change distributes the integer probability-address expression.
[source correspondence inventory](../../../../validation/source-correspondence.json) checks both complete bodies.

[IO.lean](../IO.lean) retains signed length, start, request and token words, including
inactive metadata. The mathematical input domain explicitly requires nonnegative
length/start/request words for launched rows and nonnegative token words actually
read by a nonzero block. Runtime lengths equal the arbitrary logical `counts`.
Channel padding capacity is derived from those counts and adds no length cap.
The parser and prelude keep negative integer words; admissibility belongs to the
public mathematical domain.

[Correct.lean](../RealCorrect.lean) proves the complete grid's gathered weighted sum on
named probability/value tensors:

`sum n in range(counts b), probability[b,h,n] * value[b,h,n,d]`.

The theorem retains positive block size, full grid coverage, read/output buffer
separation and unique output addresses. Its proof includes actual source success,
all source access bounds, output values and the exterior memory frame.

[FPEquiv.lean](../FPEquiv.lean) compares the complete original and existing affine
candidate grids, retaining every FP32 product, padded lane, reduction, accumulator
update and explicit output cast. Only integer address arithmetic changes; no
numerical rule is assumed. Zero-block behavior remains in the FP comparison.
[Proofs/](.) connects these specifications to the actual signed metadata
loads, signed loop counter, distinct masks and unmasked final store.

[Validation records](../../../../validation/body-migration-progress.json) distinguish
body checking, local proof checking and independent replay. Compilation alone does
not complete this task.
