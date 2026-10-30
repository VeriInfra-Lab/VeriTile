# fp4_to_bf16

fp4_to_bf16_real: logicalRealIO implements, on LogicalByteDomain, xs ↦ (j ↦ some (nibbleReal (logicalNibble xs j))).

- [fp4_to_bf16_repaired.py](fp4_to_bf16_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete finite-ABI CTA grid; exactly 2*count outputs, all 256 input bytes, masked final CTA, both zero signs at the native-bit endpoint, safe source accesses and exterior frame.

## Conditions

Public conditions:

- Finite.Storage count
- input != output
- grid <= 2147483647
- jobs.Perm (List.range grid)
- ProgramIO input domain ByteDomain: every discrete input <256
- ProgramIO memory binding and InBounds remain in the public judgment

Contract and domain correspondence:

Contract accepts contiguous nonnegative-view CUDA uint8 Tensor storage. The discrete byte domain is visible in the headline; Finite.Storage, signed PID/count safety and schedule/freshness remain explicit. Real observation does not introduce a finite floating input premise because there are no floating input channels.

Exact FP program pairs and numerical dependencies are recorded in the [fp4_to_bf16 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
