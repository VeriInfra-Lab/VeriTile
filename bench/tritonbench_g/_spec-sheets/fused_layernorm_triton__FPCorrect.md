# Spec sheet — `bench/tritonbench_g/fused_layernorm_triton/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/fused_layernorm_triton/fused_layernorm_triton.py`

## Public theorem: `normalize_source_fp32`

**Statement:**
```lean
specification normalize_source_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (normalizeSourceChunk rstd X W Bias mean Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (normalizeValues M s mean rstd X W Bias rnumel RBLOCK)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `normalize_software_fp32`

**Statement:**
```lean
specification normalize_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (normalizeSourceChunk rstd X W Bias mean Y rnumel RBLOCK) s = some final ∧
      Post s final Y RBLOCK (rowAddress s rnumel RBLOCK)
        (fun i => column s RBLOCK i.1 < rnumel)
        (roundedNormalize cfg s mean rstd X W Bias rnumel RBLOCK)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `normalize_nonempty_rounding`

**Statement:**
```lean
specification normalize_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (normalizeSourceChunk "rstd" "x" "weight" "bias" "mean" "out" 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `welford_helper_fp32`

<details><summary>docstring</summary>

```
/-- Both branches of the pinned helper, including its general update face. -/
```
</details>

**Statement:**
```lean
specification welford_helper_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (firstIteration : Bool) (value : Op .real []) (v : α)
    (previous : WelfordTriple (Op .real [])) (prev : WelfordTriple α)
    (hv : evalOp M (some .fp32) value s = some (fun _ => v))
    (hp : EvaluatesTriple M s previous prev) :
    EvaluatesTriple M s (reduceExpressions firstIteration value previous) (reduceStep M firstIteration v prev)
```

**Assumptions / layout contracts:**
- `hv : evalOp M (some .fp32) value s = some (fun _ => v)`
- `hp : EvaluatesTriple M s previous prev`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `welford_reduction_fp32`

**Statement:**
```lean
specification welford_reduction_fp32 {α : Type} [Inhabited α] (M : Algebra α) (equal32 : α→α→Bool)
    (X Mean Rstd : RegionName) (xnumel rnumel RBLOCK : Nat) (tree : ReductionTree RBLOCK)
    (hpositive : 0<rnumel) (hblock : rnumel≤RBLOCK) (htree : tree.Complete) (s : State α) :
    ∃ final, FP.Structural.exec (withEquality M equal32) (welfordBlock X Mean Rstd xnumel rnumel RBLOCK tree) s = some final ∧
      final.mem = (statisticsMemory M s Mean Rstd rnumel
        (treeValue M equal32 (firstLaneValues M s X rnumel RBLOCK) tree)).mem
```

**Assumptions / layout contracts:**
- `equal32 : α→α→Bool`
- `hpositive : 0<rnumel`
- `hblock : rnumel≤RBLOCK`
- `htree : tree.Complete`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `welford_software_with_rsqrt`

**Statement:**
```lean
specification welford_software_with_rsqrt [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) (X Mean Rstd : RegionName)
    (xnumel rnumel N : Nat) (tree : ReductionTree N)
    (hpositive : 0<rnumel) (hblock : rnumel≤N) (htree : tree.Complete)
    (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (withEquality (LayerNormWelfold.FPStages.fp32WithRsqrt rest cfg rsqrt32) (FP.eq cfg))
      (welfordBlock X Mean Rstd xnumel rnumel N tree) s = some final ∧
      final.mem = (roundedStatistics cfg rsqrt32 s Mean Rstd rnumel
        (roundedTree cfg (roundedFirstLanes s X rnumel N) tree)).mem
```

**Assumptions / layout contracts:**
- `rsqrt32 : FP.Value .fp32 → FP.Value .fp32`
- `hpositive : 0<rnumel`
- `hblock : rnumel≤N`
- `htree : tree.Complete`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `welford_nonempty_rounding`

**Statement:**
```lean
specification welford_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32) :
    ∃ final, FP.Structural.exec (withEquality
      (LayerNormWelfold.FPStages.fp32WithRsqrt rest .ieeeValue rsqrt32) (FP.eq .ieeeValue))
      (welfordBlock "x" "mean" "rstd" 1 2 2 pairTree) welfordInitial = some final ∧
      final.mem "mean" 0 = .mk .real welfordOne ∧
      final.mem "mean" 1 = welfordInitial.mem "mean" 1 ∧
      (1 + 1/16777216 : Rat) ≠ 1
```

**Assumptions / layout contracts:**
- `rsqrt32 : FP.Value .fp32 → FP.Value .fp32`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `welford_bit_examples`

**Statement:**
```lean
specification welford_bit_examples :
    roundedCombine .ieeeValue
      ⟨FP.Value.ofBits .fp32 0x3f800000,FP.Value.zero .fp32,welfordOne⟩
      ⟨FP.Value.ofBits .fp32 0x3f800001,FP.Value.zero .fp32,welfordOne⟩ =
      ⟨welfordOne,FP.Value.ofBits .fp32 0x28000000,FP.Value.ofBits .fp32 0x40000000⟩ ∧
    roundedCombine .ieeeValue
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,welfordOne⟩
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,welfordOne⟩ =
      ⟨FP.Value.infinity .fp32,FP.Value.zero .fp32,FP.Value.ofBits .fp32 0x40000000⟩
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
