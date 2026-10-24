# INT8 rowwise dequant matmul: direct checked inputs

`int8_dequant_matmul_repaired.py` keeps the accepted complete kernel and all 418
autotune configurations. The wrapper checks supported inputs before allocation,
passes the original tensors directly, and uses canonical contiguous launch
strides `(K,1,N,1,N,1)`. There is no input materialization. The original benchmark
file is unchanged.

The API remains `int8_matmul_rowwise_dequantize(a,b,state_x,state_w,bias)`.
A and B are signed INT8 matrices. Row and column scales, and optional bias, are
FP32. The explicit output and narrowed partials are FP16; the INT32 accumulator
is preserved even though the source's unused `ACC_TYPE` constant is FP32.
The old example fixture's FP16 bias does not determine the production contract.

Inputs are strided, contiguous tensors without unresolved negative views on one
CUDA device. Shapes are `[M,K]`, `[K,N]`, `[M]`, `[N]`, and optional `[N]` bias.
Nonempty FP32 input pointers must be aligned to four bytes. Nonzero storage
offsets and arbitrary inert singleton strides are supported. Read-only INT8
inputs may alias each other; FP32 scales/bias may alias each other. Mixed
INT8/FP32 raw-storage aliases are explicitly rejected: the selected local raw
memory representation has one element width per storage. This is the declared
API/representation scope, not a deferred numerical claim.

Empty M or N returns the fresh FP16 output without any kernel launch, and does
not require bounds on unused launch dimensions or offsets. Its exact returned
strides are `[max(N,1),1]`, including N=0; launch strides remain `(N,1)` because
empty outputs skip the launch. For a nonempty
output, dimensions are below `2**61`, the worst-case grid
`ceil(M/16)*ceil(N/32)` is at most `2**31-1`, and the following checked envelopes
are below `2**63`:

- `a_offset + M*K + K + 4095` and `b_offset + (K+4095)*N + N`;
- `2*((M+255)*N + N+255)` for the fresh zero-offset output;
- `4*(x_offset+M)`, `4*(w_offset+N)`, and `4*(bias_offset+N+255)`.

`Contract.lean` states these requirements. `Proofs/Contract.lean` proves exact
dynamic admission equivalence and derives every `IndexABI` and `GridABI` field
for all 418 configurations. Envelopes include masked lanes, grouped-grid
arithmetic, byte scaling, storage offsets and the final unused pointer advance.
There is no INT32 no-overflow gate or finite/positive/nonzero-scale gate.

The complete source interpretation preserves INT32 dot plans, int32-to-FP32
conversion, the runtime FP32 divisor word `948045328`, multiplication by divisor,
row scale, then column scale, FP16 narrowing, FP32 bias-to-FP16 conversion, and
FP16 bias addition only on split zero. Bias tail loads are masked. Split count
one stores its partial; larger split counts use the selected FP16 atomic-add
schedule. Every autotune trial and final split>1 call performs its zero hook.
The binary64 divisor literal and its FP32 rounding are recorded and proved in
`Proofs/Source.lean`.

`Specification.int8_dequant_matmul_wrapper_correct` covers the complete checked
wrapper, exact returned descriptor/device, every output coordinate, and caller
storage frame. Its independent mathematical target is
`column_scale[n] * (row_scale[m] * sum_k(signed(A[m,k])*signed(B[k,n])) / 16129)
+ bias[n]`, with absent bias equal to zero. Scales and bias are arbitrary original
real-valued storage; no encoding of arbitrary reals into FP words is assumed.
`DirectPointers.original_operands` and `DirectReal.target_original` connect the
source to the original contiguous tensor views and existing storage offsets.

`FloatingSpecification.int8_dequant_matmul_wrapper_fp32` is the unconditional
structural typed INT32/FP32/FP16 wrapper theorem. Its separate reference retains
the selected fixed K32 MMA/WGMMA/scalar dot plan, fusion/seed behavior, and atomic
order; it does not replace the dot by an arbitrary FP32 or exact-real fold.
`int8_dequant_matmul_native_if_local` covers the actual compiler-bound wrapper,
both bytes of every returned FP16 value, owning-device restoration, and every
pre-allocation caller byte. It consumes `RequestLaws` and derives actual input
data-pointer identities and the independent `logicalWordReference` on original
TensorView input words. `DirectWords.word_original` is consumed inside the
public result theorem, including optional bias and zero K. `SourceCall.bound` consumes the guard-derived native
bounds. Empty outputs do not require an unused bound compiled call.

Ordinary Tensor metadata/contiguity/pointer/capacity semantics, successful fresh
allocation, local typed/raw load/store bindings, source/frontend/compiler
correspondence, and race-free successful runtime execution remain implementation
boundaries. `PublicInput` binds original native input words/bytes and the FP16
output allocation; it contains no desired output. The complete production AST
is regenerated from the current Python and checked by the CPU harness. These
structural proofs are complete for the stated boundary. Native numerical
conformance is deferred separately: fixed MMA/WGMMA saturation or wrapping,
INT32 conversion, FP32 multiplication and FTZ, FP16 narrowing/addition/atomic
behavior and packing. No universal FP-to-Real equality is claimed.

Historical accepted repaired Python SHA256:
`ad8b416c5907bd4002b84c8e17d97654a73210d1d4844ca19c30708a342ca3ae`.
Review: `2026-10-18/int8-dequant-matmul-whole001-repair-integration/repair-review.json`,
SHA256 `b3eb76ee2eec22946f7da44f335ef3d9c8f64036fcff0ed7770febb51cc8501e`.
The acceptance ledger is not changed by this contract revision.

Validation: targeted Lean compilation with standard-base axiom audits and no
`sorry` or new axiom. `python bench/repairs/tritonbench_g/int8_dequant_matmul/check_contract.py`
passes 107 complete actual-AST output/metadata/frame cases over all 98 distinct
tile/split geometries, all 418 configuration bounds/reset hooks, and 27 guard
rejections. Cases cover tails, offsets, same-dtype aliases, singleton strides,
empty/K=0 outputs, nonfinite scales, bias once, and trial/final resets. CUDA is
mocked; CPU INT32-wrap/FP16 behavior is not validation of native ISA numerics.
