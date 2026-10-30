# Spec sheet — `bench/tritonbench_g/fused_recurrent_hgrn/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_hgrn/fused_recurrent_hgrn.py`

## Public theorem: `forward_fp32_stage`

**Statement:**
```lean
specification forward_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev X G O : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (forward BHPrev X G O t T D BD) s = some final ∧
      Post s final O t T D BD (forwardValues M s BHPrev X G t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dx_fp32_stage`

**Statement:**
```lean
specification dx_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dx DHPrev DO DX t T D BD) s = some final ∧
      Post s final DX t T D BD (dxValues M s DHPrev DO t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dx_two_fp32_stage`

**Statement:**
```lean
specification dx_two_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO G DX : RegionName) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dxTwo DHPrev DO G DX t T D BD) s = some final ∧
      Post s final DX t T D BD (dxTwoValues M s DHPrev DO G t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dg_fp32_stage`

**Statement:**
```lean
specification dg_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dg DHPrev DO O H0 DG U t T D BD) s = some final ∧
      Post s final DG t T D BD (dgValues M s DHPrev DO O H0 U t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `forward_software_fp32`

**Statement:**
```lean
specification forward_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (BHPrev X G O : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (forward BHPrev X G O t T D BD) s = some final ∧
      Post s final O t T D BD (roundedForward cfg s BHPrev X G t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dx_software_fp32`

**Statement:**
```lean
specification dx_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO DX : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dx DHPrev DO DX t T D BD) s = some final ∧
      Post s final DX t T D BD (roundedDx cfg s DHPrev DO t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dx_two_software_fp32`

**Statement:**
```lean
specification dx_two_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO G DX : RegionName) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dxTwo DHPrev DO G DX t T D BD) s = some final ∧
      Post s final DX t T D BD (roundedDxTwo cfg s DHPrev DO G t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dg_software_fp32`

**Statement:**
```lean
specification dg_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev DO O H0 DG : RegionName) (U : Bool) (t T D BD : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (dg DHPrev DO O H0 DG U t T D BD) s = some final ∧
      Post s final DG t T D BD (roundedDg cfg s DHPrev DO O H0 U t T D BD)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `forward_nonempty_rounding`

**Statement:**
```lean
specification forward_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (forward "carry" "input" "gate" "out" 0 1 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
