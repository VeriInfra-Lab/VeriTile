# Source review for the complete port

The file contains two JIT functions. The wrapper calls `rms_matmul_rbe` and
does not call `rbe_triton`. The latter unconditionally calls
`get_freq_multi_tokens`, but the Python file neither defines nor imports that
name. Its frequency computation therefore remains an unresolved source
dependency. No frequency implementation or mathematical law is supplied by
this port to make that missing source appear verified.

`rms_matmul_rbe` uses two program axes, wrapped input row/column indices, all
unmasked K loads and the actual `cdiv(K, BK)` iteration count. The allocated
logical K extent alone is insufficient for a partial final block. Mathematical
formulas must include every loaded padded value while retaining the logical K
divisor. Bounds must cover those padded addresses. Neither K divisibility nor
the RMS allocation length is checked by this wrapper.

The weight profiles are fp16 and signed int8. The latter reinterprets both
weight and RMS bytes as E5M2, then converts RMS to fp16 and matrix weights via
fp32 to fp16. Every conversion is retained. The dot is an independent typed
backend operation, and the accumulator addition remains a separate operation.
The square uses native integer libdevice power with exponent two. An exact
real square lemma does not identify its floating implementation with `x*x`.

The wrapper's output storage is fp16 for fp16 weights and E5M2 for signed-byte
weights. This differs from `llama_ff_triton`, whose output is fp16. The explicit
store conversion in the Lean source represents Triton's implicit destination
conversion, including the E5M2 output case. This is a floating E5M2 memory model,
not a proof of packed output bytes in the underlying int8 tensor.

The wrapper does not restrict X dtype. The initial typed family retains all
six floating input labels as a formal model. Pinned frontend dot compatibility,
discrete input profiles and native/replay evidence still require verification.
Exact source precision now compiles for all 24 templates, including the load
labels, multiplication promotion, dot input/output types, separate carry,
native power, all conversions and final output cast. The complete non-FP8 exact-real source now has successful
finite-flat IO, actual trace safety, independent padded-input output, the complete
raw-cell frame and a nonempty finite allocation witness for both variants and
all six labels. The full FP8 exact-real path now has the same execution,
access-safety and flat-frame coverage. Its independent byte-array inputs
require actual integer storage and finite E5M2 decodes at every padded read;
the formula is a pure padded RMS-GEMM expression, never an executed result.
A complete finite byte witness gives output two and preserves both raw input
bytes and integer sentinels. Separate complete typed backend executions also
pass for fp16 X, both weight arms and both variants, retaining the actual
native-power/dot/decoder callbacks and the fp16/E5M2 output distinction.
Nonzero batch output, typed sentinels and unchanged byte inputs are proved
from those same complete runs. Their exact-real arithmetic algebra witnesses
successful executions without claiming physical IEEE rounding. In particular,
merely constructing a mixed fp32/fp16 or
fp64/fp16 dot in the formal AST does not show that Triton accepts that source.

`start_token_position`, `RBE_EPILOGUE` and `THETA` are unused by the GEMM body
and remain in the new source signature. A true rotary flag does not cause an
additional epilogue in this source. The candidate commutes only one exact
natural output-row index product; it does not alter floating arithmetic or
claim a numerical or performance improvement. Negative strides, fixed-width
index overflow, inter-program races and host composition remain separate.
