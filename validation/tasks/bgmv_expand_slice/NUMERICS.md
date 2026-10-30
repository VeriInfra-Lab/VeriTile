# Native numerical boundary

The source casts FP32 inputs to FP16, forms FP16 products, reduces them in FP16,
casts the result to FP32, optionally adds old Y in FP32, and stores FP32. It keeps
the original `BLOCK_N=256`, `BLOCK_K=next_power_of_2(K)`, `SPLIT_N=N`, `EVEN_K`,
`ADD_INPUTS`, and actual `CAST_TYPE=True` branch. Masked K lanes remain explicit.

Installed Triton 3.7.1 was inspected directly. Its `_pick_sum_dtype` promotes
narrow integers and returns `None` for FP16. Its `sum` therefore reduces the FP16
input without a floating promotion. `numerical_profile.json` pins the version
and both inspected source hashes, and the checker rechecks this selection.
The retained FP16 graph matches that evidence.

`Direct.FPScalarTree.Plans` selects an input-independent finite tree for each
typed reduction output. Each plan contains every original lane once plus its
declared zero leaves. Its evaluation uses explicit scalar additions, retaining
the full BK tree and its actual dtype. There is no reduction-value callback in
the public native result.

`Direct.Native.reference` expands the independent source formula into terms,
lowers selected primitive expressions, and interprets fixed-arity instructions.
The primitive term/expression machinery is reused from FastRMS, with a separate
BGMV compiler binding and production AST pin. Only scalar literal/cast/binary
instruction interpretation remains numerical. The FP16 multiplication and
reduction cannot enter that helper's FP32 FMA cases; the final FP16-to-FP32 cast
also separates the sum from the optional addition. Structural operand selection
retains native operand order without assuming commutativity on bit patterns.

Raw FP32 X/Y loads and two-byte FP16 W loads come from original caller storage.
Int32/int64 metadata is decoded from its actual bytes. The output word is split
into four concrete bytes. Whole-result, reduction-result, successful-execution,
and Real-projection assumptions are absent from the native public theorem.

The compiler and fixed-arity numerical instructions remain ordinary external
implementation boundaries. The source/CPU checks do not claim GPU conformance
for NaNs, signed zeros, denormals, rounding, or exceptional arithmetic.
