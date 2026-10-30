# chunk_gated_attention

Mathematics.innerWeight/innerContribution/innerState expose independent chunk gate products and logical history/final contraction, including gateK choice. prefixResult exposes the finite within-chunk cumulative sum.

- [chunk_gated_attention_repaired.py](../chunk_gated_attention_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both genuine recurrent and prefix Real entries retained, complete ordinary-grid plan lists and logical histories/final outputs, zero-padding/repeated-tail staging and exhausted autotune programs.

## Conditions

StagedLayout.Canonical; logical K<=p.K and V<=p.V; DirectRealSequence.InputSeparate and conditional history/final separation; innerDomain retained via ImplementsOn. Prefix keeps Canonical, BS in {16,32,64}, T<=p.T, S<=p.S, input/output separation and prefixDomain. KernelOnly Contract retains active checked staging and BS selection.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_gated_attention contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
