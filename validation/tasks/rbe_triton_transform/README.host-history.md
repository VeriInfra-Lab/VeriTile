# Contiguous FP32 rotary pairs

The separate candidate rotates adjacent logical pairs with
`angle=(position+row)/10000^((even_column mod 128)/128)`. An odd trailing real
component has a zero imaginary partner. It returns fresh contiguous storage
with the original shape, FP32 dtype and CUDA device. The original Python is
unchanged.

## Contract and source

X is a strided contiguous rank-three FP32 CUDA tensor with no unresolved
negative flag and a four-byte-aligned nonempty data pointer. Empty dimensions,
storage offsets and noncanonical unused singleton strides are supported.
Values may include any FP32 payload. Position is a Python int in signed64;
bool follows int semantics. On a nonempty launch,
`position + 2*ceil(rows/2) - 1` must also fit signed64. This check includes the
inactive padded row's frequency calculation. `Contract.lean` states exactly
these checks, and both public entries prove `python_checks_iff_preconditions`.

The source retains the 2x1024 tile, theta=10000, frequency dimension128, both
barriers, masks, default launch/compiler settings and default fusion. It makes
no input copies or dtype/device transfers. Logical contiguous strides are
passed to the kernel, so an unused singleton stride need not be canonical.
The imaginary-neighbour address uses the corresponding column stride.

Program IDs, dimensions, strides, positions and lanes are widened before
arithmetic. Scalar `tl.cast` handles constexpr-one specialization. Active
addresses and four-byte access bounds follow from the actual allocated
storage; the logical size is at most 2^62. The checked position domain bounds
every padded frequency position. Inactive pointer expressions may wrap but
produce no memory event. There is no artificial 32-bit shape restriction.
Empty input allocates and returns without launching.

## Proofs and numerical boundary

`Kernels.lean` contains only source parameters and the complete typed FP32
kernel. Historical source execution, coverage, scalar, memory and numerical
lemmas remain under `Proofs/`. The complete source execution is linked to each
active pair, actual returned descriptor, input preservation through every
store prefix, and all four bytes of each output element.

The current public specifications are
`Correct.rbe_triton_transform_repaired_real_returns`,
`FPEquiv.rbe_triton_transform_repaired_fp32_returns`, and
`FPEquiv.rbe_triton_transform_native_addresses`. They use the same checked
metadata and ordinary successful Tensor/allocation preparation. Native index
conditions are derived from these facts. The input frame contains the entire
original backing, including offset prefixes and unused suffixes. Returned
formulas read the original input bytes. Active word addresses are exact
64-bit values, aligned and within the original/returned allocation bounds.

The Real formula quantifies independent Real input memory. The FP theorem
preserves the selected ordered local operations and all FP32 payloads. The
observed default-fusion profile uses cosine first in the fused product and
sine first in the rounded product: real is `fma(cos,real,-mul(sin,imag))`, and
imaginary is `fma(cos,imag,mul(sin,real))`. Local provenance records derive this
profile from the source expression; no multiplication commutativity or
separate/fused equality is assumed.

Local native conformance for literal/integer conversion, division, pow,
sin/cos, multiply, sign and FMA remains deferred. The preserved historical
`whole_fp` theorem additionally gives an FP-to-Real refinement under explicit
local finite/exactness assumptions. Those assumptions are absent from the
current source-equivalence specification and are not claimed as native
conformance or as properties of arbitrary FP32 inputs.

The selected compiler lowering and successful logical-grid delivery are
ordinary execution boundaries. Shared large-grid launcher compatibility
remains deferred under `documents/bench/TODO.md`; no grid segmentation or cap was added.

## Validation

All 15 local Lean modules compile with standard-base axiom audits. The actual
Python wrapper and both JIT ASTs pass 15 CPU storage cases and 14 guard
rejections. Cases include empty dimensions, signed positions through int64
endpoints, offset inputs, singleton strides of 2^62, odd/even tails and multiple
row/column blocks. Independent Torch comparisons, complete once-only output
coverage and the entire input-backing frame pass.

`check_lowering.py --output-dir /tmp/rbe-lowering` passes four Triton3.7.1
offline SM80 profiles: i32/u32/i64 scalar ABIs and all-one specialization.
The emitted memory operations are scalar `.b32` accesses, including cached
loads used by libdevice; native products are int64 and default FMA remains.
There was no GPU execution or native numerical-conformance validation.

Candidate SHA256: `9095bec159a1bceccea0b6415d198eb1efed6fd30947e79d378ff6f70f17c592`.
Original SHA256: `57ff14376b6780d4c8cca0a590189d279a2e4e1dfffd95c4be529440c1dd57af`.
