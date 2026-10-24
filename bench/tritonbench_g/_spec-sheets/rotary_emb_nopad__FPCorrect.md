# Spec sheet — `bench/tritonbench_g/rotary_emb_nopad/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/rotary_emb_nopad/rotary_emb_nopad.py`

## Public theorem: `rotary_nopad_q_fp32_correctness`

**Statement:**
```lean
specification rotary_nopad_q_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPBlocks.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPBlocks.active Bool.false p s) (FPBlocks.offset Bool.false p s)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPBlocks.source Bool.false p) s = some final ∧
      FPBlocks.Post Bool.false p s final (FPBlocks.roundedValue cfg Bool.false p s)
```

**Assumptions / layout contracts:**
- `hw : Pair.WritesApart (FPBlocks.active Bool.false p s) (FPBlocks.offset Bool.false p s)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `rotary_nopad_k_fp32_correctness`

**Statement:**
```lean
specification rotary_nopad_k_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPBlocks.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPBlocks.active Bool.true p s) (FPBlocks.offset Bool.true p s)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPBlocks.source Bool.true p) s = some final ∧
      FPBlocks.Post Bool.true p s final (FPBlocks.roundedValue cfg Bool.true p s)
```

**Assumptions / layout contracts:**
- `hw : Pair.WritesApart (FPBlocks.active Bool.true p s) (FPBlocks.offset Bool.true p s)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `rotary_nopad_v2_q_fp32_correctness`

**Statement:**
```lean
specification rotary_nopad_v2_q_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPV2.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPV2.active p s) (FPV2.outputOffset Bool.false p s)) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest cfg) (FPV2.source Bool.false p) s = some final ∧
      FPV2.Post Bool.false p s final (FPV2.roundedValue cfg p s)
```

**Assumptions / layout contracts:**
- `hw : Pair.WritesApart (FPV2.active p s) (FPV2.outputOffset Bool.false p s)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `rotary_nopad_v2_cache_fp32_correctness`

**Statement:**
```lean
specification rotary_nopad_v2_cache_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : FPV2.Parameters)
    (s : State (FP.Value .fp32)) (hw : Pair.WritesApart (FPV2.active p s) (FPV2.outputOffset Bool.true p s)) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest cfg) (FPV2.source Bool.true p) s = some final ∧
      FPV2.Post Bool.true p s final (FPV2.roundedValue cfg p s)
```

**Assumptions / layout contracts:**
- `hw : Pair.WritesApart (FPV2.active p s) (FPV2.outputOffset Bool.true p s)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
