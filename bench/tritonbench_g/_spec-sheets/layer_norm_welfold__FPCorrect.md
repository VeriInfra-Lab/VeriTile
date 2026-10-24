# Spec sheet — `bench/tritonbench_g/layer_norm_welfold/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/layer_norm_welfold/layer_norm_welfold.py`

## Public theorem: `normalize_chunk_fp32`

**Statement:**
```lean
specification normalize_chunk_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (rnumel RBLOCK : Nat) (s : State α) :
    ∃ final, FP.Structural.exec M (normalizeChunk mean rstd X W Bias Y rnumel RBLOCK) s = some final ∧
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
      (normalizeChunk mean rstd X W Bias Y rnumel RBLOCK) s = some final ∧
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
      (normalizeChunk "mean" "rstd" "x" "weight" "bias" "out" 1 2) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = roundingInitial.mem "out" 1 ∧
      (1 + 1 / 33554432 : Rat) ≠ 1
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `source_one_block_fp32`

**Statement:**
```lean
specification source_one_block_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (mean rstd X W Bias Y : RegionName) (xnumel rnumel RBLOCK : Nat) (s : State α)
    (hpos : 0 < rnumel) (hle : rnumel ≤ RBLOCK) :
    ∃ final, FP.Structural.exec M (sourceOneBlock mean rstd X W Bias Y xnumel rnumel RBLOCK) s = some final ∧
      final.mem = (sourceMemory M s mean rstd X W Bias Y rnumel RBLOCK).mem
```

**Assumptions / layout contracts:**
- `hpos : 0 < rnumel`
- `hle : rnumel ≤ RBLOCK`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `source_one_block_fp32_with_rsqrt`

**Statement:**
```lean
specification source_one_block_fp32_with_rsqrt [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (rsqrt32 : FP.Value .fp32 → FP.Value .fp32)
    (mean rstd X W Bias Y : RegionName) (xnumel rnumel RBLOCK : Nat) (s : State (FP.Value .fp32))
    (hpos : 0 < rnumel) (hle : rnumel ≤ RBLOCK) :
    ∃ final, FP.Structural.exec (fp32WithRsqrt rest cfg rsqrt32)
      (sourceOneBlock mean rstd X W Bias Y xnumel rnumel RBLOCK) s = some final ∧
      final.mem = (roundedSourceMemory cfg rsqrt32 s mean rstd X W Bias Y rnumel RBLOCK).mem
```

**Assumptions / layout contracts:**
- `rsqrt32 : FP.Value .fp32 → FP.Value .fp32`
- `hpos : 0 < rnumel`
- `hle : rnumel ≤ RBLOCK`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
