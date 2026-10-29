# attention_llama

Finite-N exp-attention weighted value average over keys satisfying causal=false or key+SP≤query, with runtime scale and full D feature sum.

- [attention_llama_scalar_abi_checked.py](attention_llama_scalar_abi_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full ordinary query/batch/head grid with caller strides, all original FP16 outputs and both K/V encodings; actual source safety/success/frame.

## Conditions

- hD>0, hN>0, causal→SP=0, byte-branch Q/K/V/Out separation and output-layout injection remain explicit.
- Public RealLayout.domain restricts only reached byte-role E5M2 values to finite; source integer scalar ABI and original supported configuration retained.

Exact FP program pairs and numerical dependencies are recorded in the [attention_llama contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
