# fused_rwkv6_kernel

fused_rwkv6_kernel_scalar_abi_checked.rwkv_real: kernelIO implements per-key-shard partialValue and optional terminal recurrence over named tensors.

- [fused_rwkv6_kernel_scalar_abi_checked.py](fused_rwkv6_kernel_scalar_abi_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full key/value/head grid, all T steps, optional initial/final state, all key-shard partial outputs and no-launch behavior. No host key-shard sum is asserted.

## Conditions

Public conditions:

- Shape and Sizes c
- Runtime Real scale is explicit in kernelIO
- Native comparison retains ScalarBinding/raw PythonNumber and ReductionPlan
- ProgramIO physical binding, active output masks, source safety and full exterior frame

Contract and domain correspondence:

Contract keeps original Checked.Guards and branch-local integer ScalarABI range -(2^63)<=n<2^64 only when a launch occurs; floating payloads (including exceptional words) are unrestricted in the native model. Real kernel result is independent exact arithmetic, not an extra host/FP32 hardware requirement.

Exact FP program pairs and numerical dependencies are recorded in the [fused_rwkv6_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
