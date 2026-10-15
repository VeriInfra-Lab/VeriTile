# Spec sheet — `bench/tritonbench_g/rotary_transform/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/rotary_transform/rotary_transform.py`

## Public theorem: `rotary_transform_fp32_correctness`

<details><summary>docstring</summary>

```
/-- Concrete binary32, separately rounded arithmetic for the complete already
selected paired row and both fixed/varlen metadata branches. -/
```
</details>

**Statement:**
```lean
specification rotary_transform_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (k : FPRow.Kind)
    (p : FPRow.Parameters) (s : State (FP.Value .fp32)) (hstride : p.strideOutDimension ≠ 0) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (FPRow.source k p) s = some final ∧
      FPRow.Post k p s final (FPRow.roundedValue cfg k p s)
```

**Assumptions / layout contracts:**
- `hstride : p.strideOutDimension ≠ 0`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
