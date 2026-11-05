# uniform_sampling

Current repaired Python implementation: [uniform_sampling.py](../uniform_sampling.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Canonical Philox10 word, existing rounded uniform decoder, then the independent lower + u*(../upper−lower) affine formula.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The two complete 22-statement `@triton.jit` bodies and the three-statement
fallback helper are copied literally in `Kernels.lean`. Caller metadata supplies
FP32 argument words, signed64 seed/offset channels and the checked block size.
Every cast, modular counter addition, Philox field, affine FMA, clamped pointer,
mask and cache hint stays in the source. FP32 results store directly, without an
extra conversion that the Python program does not contain.

`Proofs/UniformDecoder.lean` connects the actual selected uint32 helper path to
the canonical decoder: signed bitcast, sign fold, integer-to-FP32 rounding and
FP32 multiplication by the rounded source scale. The full dtype-dispatch helper
is retained; Philox4x32 does not call its uint64 specialization. The public Real
formula and complete FP comparison retain their existing dimensions, launch,
ABI, output, safety and exterior-frame conditions. Independent replay of the
exact current source is required before this migration is marked complete.
