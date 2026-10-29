# uniform_sampling

Canonical Philox10 word, existing rounded uniform decoder, then the independent lower + u*(upper−lower) affine formula.

- [uniform_sampling_repaired.py](uniform_sampling_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete actual invocation, all four canonical Philox fields and B lanes with output offset (pid*4+field)*B+lane<N, clamped stores and exterior frame. Whole-output wrapper chunks/allocator/reservation orchestration are optional.

## Conditions

- Finite Denotes for lower/upper is the Real interpretation of scalar ABI words, not Python input admission
- Original nonempty N, block B=512 or 1024, ArangeSizeDomain B, launched PID and signed64 seed/offset ABI remain; generic theorem needs seedABI and positive B
- Removed redundant N≤2^34 from both main statements: KernelPublic.implements_real and equivalent have no such premise. Actual wrapper still chunks at low-word carry boundaries; existing Bounds/KernelMemory prove ≤2^34 per launch, while Contract admits total N<2^63

Exact FP program pairs and numerical dependencies are recorded in the [uniform_sampling contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
