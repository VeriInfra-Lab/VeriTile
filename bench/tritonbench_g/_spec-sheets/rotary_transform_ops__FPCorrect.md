# Spec sheet — `bench/tritonbench_g/rotary_transform_ops/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/rotary_transform_ops/rotary_transform_ops.py`

## Public theorem: `rotary_transform_ops_fp32_correctness`

**Statement:**
```lean
specification rotary_transform_ops_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (k : FPStages.Kind)
    (p : FPStages.Parameters) (s : State (FP.Value .fp32)) (hstride : p.strideOutDimension ≠ 0) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPStages.source k p) s = some final ∧
      RotaryTransform.FPRow.Post k p s final (RotaryTransform.FPRow.roundedValue cfg k p s)
```

**Assumptions / layout contracts:**
- `hstride : p.strideOutDimension ≠ 0`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
