# Source scope

Both Python JITs are represented. `rms_matmul_rbe_qkv` is expanded into its
three actual sequential calls, preserving per-call weight/output strides and
shared X/RMS input. The source rotary flags are True, True and False; the
callee never uses those flags, start position or theta, and performs no rotary
epilogue. Those unused parameters remain in the typed signatures.

## Source operations and storage

The source loop count is ceiling division. All X, RMS and weight loads are
unmasked, including lanes outside logical K and wrapped rows/columns of an
inactive output tile. In each iteration, the RMS moment uses the raw X value
converted to fp32 with integer-exponent `libdevice.pow(..., 2)`; multiplication
by RMS precedes the typed dot; the dot result is added to a distinct fp32 carry.
Power, dot and scalar arithmetic remain distinct backend operations.

The FP8 path loads raw signed bytes, reinterprets RMS as E5M2 then converts it
to fp16; weights additionally convert through fp32 before fp16. These are not
numerical integer-to-float conversions. Typed equivalence preserves failure
and all byte payloads. The real-valued decoded-weight contracts require actual signed integer cells
and finite decoding at every loaded address, including padding. Invalid or
nonfinite payloads have no decoded value and cannot satisfy those contracts.

Q is allocated at fp16 even with int8 weights. K/V are supplied by the caller,
checked against weight dtype and reinterpreted as E5M2 on the byte branch.
The formal typed callee therefore has a separate output-storage parameter.
E5M2 typed stores are not a proof of packed byte values in the underlying int8
PyTorch buffers. Native conversion, rounding and byte-packing remain distinct
obligations.

## Proven mathematical and memory scope

The new non-FP8 and byte contracts use exact-real floating erasure, actual dynamic
source execution, actual trace safety and finite-flat execution. Expected
values are independent initial-input sums: the normalizer uses all
`cdiv(K,BK)*BK` loaded X lanes divided by logical K plus EPS, and the GEMM
uses the same padded traversal with the pass's own RMS/weight strides.
The byte contracts relate original typed integer cells to supplied finite
byte arrays, then use the pure `PaddedRMSGemm.output` formula. The intermediate
decoded view only transforms the specified RMS/weight cells; it never mutates
the executed source memory or supplies zero for an absent decoder result.
K divisibility is not a new contract hypothesis. Bounds explicitly cover
all unmasked loads and all active output stores. A natural zero divisor has
the model's totalized mathematical meaning; native launch validity is not
inferred from it.

Output injectivity is required only on active lanes, allowing N smaller than
BN. Finite allocations must be disjoint and closed, with zero undef. Each
pass frames every raw cell outside its active footprint, including other
stored dtypes and unallocated flat addresses. QKV transports read memory
between passes with the stated ten output/read region inequalities (plus
X/RMS and X/weight separation on the byte path); the conclusion
frames the union of all three active footprints. It does not assume eight
pairwise-distinct buffers: read buffers may alias when the stated conditions
permit it.

The wrapper checks shapes/dtypes but **does not check K divisibility or add a
mask to the last K block**. Ordinary unpadded tensors with nonmultiple K can
therefore access beyond their logical allocation. The finite proof applies
only when its complete access bounds hold. Historical prose saying the host
guarantees divisibility is superseded by this review.

## Remaining admission and validation boundaries

The wrapper does not constrain X dtype. The current typed family includes six
floating labels (fp16, bf16, fp32, fp64, E4M3 and E5M2); it is a formal family,
not evidence that all mixed dot inputs pass a pinned Triton frontend. Discrete
integer/bool X inputs, negative strides, native integer widths/overflow, GPU
launch/grid scheduling and inter-program races are not established here.
These unresolved source/admission obligations must remain visible in acceptance.

The precision templates are independent AST prescriptions checked against the
complete source. They are not empirical GPU rounding evidence. Likewise the
exact-real finite witnesses do not establish IEEE behavior or native device
execution. Backend-parametric candidate equivalence is unconditional but is
not a compiler correctness theorem.

General callee/QKV byte IO and concrete finite byte readbacks are proved.
The standalone typed numeric witness executes the complete callee three times,
including each actual nonempty ceil-K loop, native integer power, dot, casts and
masked store. Both weight profiles and both variants produce Q=2, K=3, V=4
with the source's separate Q and K/V storage types. The independent target
memory is proved equal to the complete run's final memory, including masked
integer sentinels and raw input bytes. This backend uses exact real arithmetic
and identity casts; it does not establish IEEE rounding or packed E5M2 storage.
Independent comparator replay remains pending; no official passing artifact
has been produced for this port.
