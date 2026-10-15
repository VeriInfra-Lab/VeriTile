# Spec sheet — `bench/tritonbench_g/fused_recurrent_delta/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_delta/fused_recurrent_delta.py`

## Public theorem: `output_fp32_stage`

**Statement:**
```lean
specification output_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HNext q o : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (output HNext q o t s_qk_h s_vo_h B H K V BK BV scale) s = some final ∧
      Post s final o BV (outputAddress s t s_vo_h B H V BV)
        (fun i => valueIndex s BV i.1 < V) (outputValues M s HNext q t s_qk_h K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `vnew_fp32_stage`

**Statement:**
```lean
specification vnew_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v : RegionName) (t s_qk_h s_vo_h K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (vnew HPrev k v t s_qk_h s_vo_h K V BK BV) s = some final ∧
      Post s final v BV (valueAddress s t s_vo_h V BV)
        (fun i => valueIndex s BV i.1 < V) (vnewValues M s HPrev k v t s_qk_h s_vo_h K V BK BV)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `state_fp32_stage`

**Statement:**
```lean
specification state_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta HOut : RegionName) (headwise : Bool)
    (t s_qk_h s_vo_h T K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M
      (stateStep HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV) s = some final ∧
      final.mem = (stateScatter s HOut K V BK BV
        (stateValues M s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV)).mem
```

**Assumptions / layout contracts:**
- `headwise : Bool`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `vnew_software_fp32`

**Statement:**
```lean
specification vnew_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HPrev k v : RegionName)
    (t s_qk_h s_vo_h K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (vnew HPrev k v t s_qk_h s_vo_h K V BK BV) s = some final ∧
      Post s final v BV (valueAddress s t s_vo_h V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedResidual cfg s HPrev k v t s_qk_h s_vo_h K V BK BV)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `output_software_fp32`

**Statement:**
```lean
specification output_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HNext q o : RegionName)
    (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (output HNext q o t s_qk_h s_vo_h B H K V BK BV 1) s = some final ∧
      Post s final o BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V)
        (roundedOutputOne cfg s HNext q t s_qk_h K V BK BV)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `state_software_fp32`

**Statement:**
```lean
specification state_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (HPrev k v beta HOut : RegionName)
    (headwise : Bool) (t s_qk_h s_vo_h T K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg)
      (stateStep HPrev k v beta HOut headwise t s_qk_h s_vo_h T K V BK BV) s = some final ∧
      final.mem = (stateScatter s HOut K V BK BV
        (roundedState cfg s HPrev k v beta headwise t s_qk_h s_vo_h T K V BK BV)).mem
```

**Assumptions / layout contracts:**
- `headwise : Bool`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `output_nonempty_rounding`

**Statement:**
```lean
specification output_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest .ieeeValue)
      (output "state" "q" "out" 0 2 1 1 1 2 1 2 2 1) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dkHeadwise_fp32_stage`

**Statement:**
```lean
specification dkHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dkScalar_fp32_stage`

**Statement:**
```lean
specification dkScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dkValues M s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dkCorrection_fp32_stage`

**Statement:**
```lean
specification dkCorrection_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (correctionValues M s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dvHeadwise_fp32_stage`

**Statement:**
```lean
specification dvHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dvScalar_fp32_stage`

**Statement:**
```lean
specification dvScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (dvValues M s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dbetaHeadwise_fp32_stage`

**Statement:**
```lean
specification dbetaHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV scale) s = some final ∧
      Post s final dbeta BV (betaOutputAddress s t s_vo_h NK B H V BV) (fun i => valueIndex s BV i.1 < V) (dbetaValues M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dbetaScalar_fp32_stage`

**Statement:**
```lean
specification dbetaScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      final.mem = (s.write dbeta (scalarBetaAddress s t B H T) (.mk .real (scalarDbetaValue M s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV scale))).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dqHeadwise_fp32_stage`

**Statement:**
```lean
specification dqHeadwise_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dqScalar_fp32_stage`

**Statement:**
```lean
specification dqScalar_fp32_stage {α : Type} [Inhabited α] (M : Algebra α)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat) (scale : ℝ) (s : State α) :
    ∃ final, FP.Structural.exec M (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV scale) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (dqValues M s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV scale)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dkHeadwise_software_fp32`

**Statement:**
```lean
specification dkHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkHeadwise DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDkValues cfg s DHPrev q do_ v beta Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dkScalar_software_fp32`

**Statement:**
```lean
specification dkScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ v beta dk : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkScalar DHPrev q do_ v beta dk t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDkValues cfg s DHPrev q do_ v beta Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dkCorrection_software_fp32`

**Statement:**
```lean
specification dkCorrection_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev dv dk : RegionName) (t s_qk_h s_vo_h B H K V BK BV : Nat) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dkCorrection HPrev dv dk t s_qk_h s_vo_h B H K V BK BV) s = some final ∧
      Post s final dk BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitCorrectionValues cfg s HPrev dv dk t s_qk_h s_vo_h B H K V BK BV)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dvHeadwise_software_fp32`

**Statement:**
```lean
specification dvHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dvHeadwise DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDvValues cfg s DHPrev q do_ k beta Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dvScalar_software_fp32`

**Statement:**
```lean
specification dvScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k beta dv : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dvScalar DHPrev q do_ k beta dv t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dv BV (outputAddress s t s_vo_h B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDvValues cfg s DHPrev q do_ k beta Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dbetaHeadwise_software_fp32`

**Statement:**
```lean
specification dbetaHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h NK B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dbetaHeadwise DHPrev q do_ k v dbeta t s_qk_h s_vo_h NK B H T K V BK BV 1) s = some final ∧
      Post s final dbeta BV (betaOutputAddress s t s_vo_h NK B H V BV) (fun i => valueIndex s BV i.1 < V) (bitDbetaValues cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV (FP.Value.ofBits .fp32 0x3f800000))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dbetaScalar_software_fp32`

**Statement:**
```lean
specification dbetaScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (DHPrev q do_ k v dbeta : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dbetaScalar DHPrev q do_ k v dbeta t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      final.mem = (s.write dbeta (scalarBetaAddress s t B H T) (.mk .real (bitScalarDbetaValue cfg s DHPrev q do_ k v t s_qk_h s_vo_h K V BK BV (FP.Value.ofBits .fp32 0x3f800000)))).mem
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dqHeadwise_software_fp32`

**Statement:**
```lean
specification dqHeadwise_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dqHeadwise HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDqValues cfg s HPrev k v beta do_ Bool.true t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `dqScalar_software_fp32`

**Statement:**
```lean
specification dqScalar_software_fp32 [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (HPrev k v beta do_ dq : RegionName)
    (t s_qk_h s_vo_h B H T K V BK BV : Nat)  (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (sequentialFP32 rest cfg) (dqScalar HPrev k v beta do_ dq t s_qk_h s_vo_h B H T K V BK BV 1) s = some final ∧
      Post s final dq BK (keyOutputAddress s t s_qk_h B H K BK) (fun i => keyIndex s BK i.1 < K) (bitDqValues cfg s HPrev k v beta do_ Bool.false t s_qk_h s_vo_h T K V BK BV (FP.Value.ofBits .fp32 0x3f800000))
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
