# fused_activation

fused_activation_kernel_correctness: activationIO implements j ↦ sigmoid(z) or max 0 z, where z = input[j]/2 + x[j] + bias[j].

- [fused_activation_repaired.py](fused_activation_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete masked source program at the public regional PID/window; all three entry-state reads, cyclic bias, both activation branches, in-place x output, access safety and exterior frame. No wrapper-wide launch claim.

## Conditions

Public conditions:

- W,N,B and activation Bool are public
- RegionalImplements retains coherent input memory bindings, active read/write bounds and output uniqueness
- Active lane is pid*B+j<N

Contract and domain correspondence:

The checked Python contract gives FP32 contiguous inputs, adequate input extent, nonempty bias on nonempty work and allocation separation. The kernel regional theorem is more general and keeps memory conditions in its judgment; no new Python guard is introduced.

Exact FP program pairs and numerical dependencies are recorded in the [fused_activation contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
