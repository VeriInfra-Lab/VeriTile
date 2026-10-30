# Spec sheet — `bench/tritonbench_g/fused_rwkv6_kernel/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/fused_rwkv6_kernel/fused_rwkv6_kernel.py`

## Public theorem: `state_fp32_stage`

**Statement:**
```lean
specification state_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (stateStep BHPrev k v w BHOut t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (stateScatter s BHOut K V BK BV (stateValues M s BHPrev k v w t s_k_h s_v_h K V BK BV)).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `output_fp32_stage`

**Statement:**
```lean
specification output_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV scale) s = some final ∧
      Post s final o BV (outputAddress s t s_v_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s BHPrev q k v u t s_k_h s_v_h H K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `conditional_final_fp32_stage`

**Statement:**
```lean
specification conditional_final_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (BHPrev k v w ht : RegionName) (STORE_FINAL_STATE : Bool)
    (t s_k_h s_v_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (finalStep BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (if STORE_FINAL_STATE then
        stateScatter s ht K V BK BV (stateValues M s BHPrev k v w t s_k_h s_v_h K V BK BV) else s).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `state_fp32_with_exp_primitive`

**Statement:**
```lean
specification state_fp32_with_exp_primitive [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (exp32 : FP.Value .fp32 → FP.Value .fp32)
    (BHPrev k v w BHOut : RegionName) (t s_k_h s_v_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (fp32WithExp rest cfg exp32)
      (stateStep BHPrev k v w BHOut t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (stateScatter s BHOut K V BK BV
        (roundedStateWithExp cfg exp32 s BHPrev k v w t s_k_h s_v_h K V BK BV)).mem
```

**Assumptions / layout contracts:**
- `exp32 : FP.Value .fp32 → FP.Value .fp32`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `conditional_final_fp32_with_exp_primitive`

**Statement:**
```lean
specification conditional_final_fp32_with_exp_primitive [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (exp32 : FP.Value .fp32 → FP.Value .fp32)
    (BHPrev k v w ht : RegionName) (STORE_FINAL_STATE : Bool)
    (t s_k_h s_v_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (fp32WithExp rest cfg exp32)
      (finalStep BHPrev k v w ht STORE_FINAL_STATE t s_k_h s_v_h K V BK BV) s = some final ∧
      final.mem = (if STORE_FINAL_STATE then stateScatter s ht K V BK BV
        (roundedStateWithExp cfg exp32 s BHPrev k v w t s_k_h s_v_h K V BK BV) else s).mem
```

**Assumptions / layout contracts:**
- `exp32 : FP.Value .fp32 → FP.Value .fp32`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `output_software_fp32`

**Statement:**
```lean
specification output_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (BHPrev q k v u o : RegionName) (t s_k_h s_v_h B H T K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (outputStep BHPrev q k v u o t s_k_h s_v_h B H T K V BK BV 1) s = some final ∧
      Post s final o BV (outputAddress s t s_v_h B H V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedOutput cfg s BHPrev q k v u t s_k_h s_v_h H K V BK BV)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `output_nonempty_rounding`

**Statement:**
```lean
specification output_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest .ieeeValue)
      (outputStep "state" "query" "key" "value" "bias" "out" 0 1 1 1 1 1 1 1 1 2 1)
      roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
