# Spec sheet — `bench/tritonbench_g/mixed_sparse_attention/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/mixed_sparse_attention/mixed_sparse_attention.py`

## Public theorem: `forward_fp16`

<details><summary>docstring</summary>

```
/-- The originally accepted forward scope includes all eight fixed block
iterations, the one column iteration for BLOCK_N≥16, and writeback. -/
```
</details>

**Statement:**
```lean
specification forward_fp16 {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (hBN : 16≤c.BN) (s : State α) (hactive : rowStart c s < sequenceLength c s) :
    ∃ final, FP.Structural.exec (backend M D) (kernel c) s = some final ∧
      final.mem = (outputMemory M c s (acceptedValues M D c s)).mem
```

**Assumptions / layout contracts:**
- `hBN : 16≤c.BN`
- `hactive : rowStart c s < sequenceLength c s`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `forward_nonempty`

**Statement:**
```lean
specification forward_nonempty {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) :
    ∃ final, FP.Structural.exec (backend M D) (kernel witnessConfig) witnessInitial = some final ∧
      final.mem = (outputMemory M witnessConfig witnessInitial
        (finalValues M D witnessConfig witnessInitial)).mem ∧
      (∃ i : TileIndex [16,16], rowStart witnessConfig (witnessInitial : State α)+i.1.val <
        sequenceLength witnessConfig (witnessInitial : State α))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `epilogue_fp16`

**Statement:**
```lean
specification epilogue_fp16 {α : Type} [Inhabited α] (M : Algebra α) (AccPre LPre : RegionName)
    (c : Config) (s : State α) :
    ∃ final, FP.Structural.exec M (epilogueSource AccPre LPre c) s = some final ∧
      final.mem = (outputMemory M c s (epilogueInputs M c s AccPre LPre)).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `epilogue_software_with_cast`

**Statement:**
```lean
specification epilogue_software_with_cast [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (toHalf : FP.Value .fp32 → FP.Value .fp32) (AccPre LPre : RegionName)
    (c : Config) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (softwareEpilogue rest cfg toHalf) (epilogueSource AccPre LPre c) s = some final ∧
      final.mem = (roundedEpilogueMemory cfg toHalf AccPre LPre c s).mem
```

**Assumptions / layout contracts:**
- `toHalf : FP.Value .fp32 → FP.Value .fp32`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `epilogue_nonempty_rounding`

**Statement:**
```lean
specification epilogue_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (toHalf : FP.Value .fp32 → FP.Value .fp32) :
    ∃ final, FP.Structural.exec (softwareEpilogue rest .ieeeValue toHalf)
      (epilogueSource "acc" "den" epilogueWitnessConfig) epilogueWitnessInitial = some final ∧
      final.mem "out" 0 = .mk .fp16 (toHalf (FP.Value.ofBits .fp32 0x3eaaaaab)) ∧
      final.mem "out" 1 = epilogueWitnessInitial.mem "out" 1 ∧
      (11184811/33554432 : Rat) ≠ 1/3
```

**Assumptions / layout contracts:**
- `toHalf : FP.Value .fp32 → FP.Value .fp32`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
