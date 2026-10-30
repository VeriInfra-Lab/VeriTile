# Spec sheet — `bench/tritonbench_g/rope_embedding/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/rope_embedding/rope_embedding.py`

## Public theorem: `rope_embedding_fp32_correctness`

<details><summary>docstring</summary>

```
/-- All four accepted half-store stages, including actual backward fneg,
register casts, initial-state numerical reference, success, readback and frame. -/
```
</details>

**Statement:**
```lean
specification rope_embedding_fp32_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (backward second : Bool)
    (p : FPHalves.Parameters) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (FPHalves.source backward second p) s = some final ∧
      FPHalves.Post second p s final (FPHalves.roundedValue cfg backward second p s)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
