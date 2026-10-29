# context_attn_mistral

kernel_real: complete original ordinary grid implements finiteSentinelAttention on the original metadata domain.

- [context_attn_mistral_repaired.py](../context_attn_mistral_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every active output lane in the original ordinary grid, retaining original underlaunch. Key range is Formula.count * BN, including rounded blocks; gathered padded values are zero and finite sentinel weights remain in the denominator. This is not ideal masked causal softmax.

## Conditions

Conditions:

- RealInterface.Geometry c capacity
- RealInterface.Metadata c capacity as the explicit ImplementsOn domain
- Original Host.Call, dense/strided region windows, signed start/length metadata, grouped heads and output separation retained

Domain source:

KernelOnly/RealIO.lean: Geometry, Metadata, programIO; Contract.lean retains Checked.Admission.

Exact FP program pairs and numerical dependencies are recorded in the [context_attn_mistral contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
