# Spec sheet — `bench/tritonbench_g/rope_transform/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/rope_transform/rope_transform.py`

## Public theorem: `rope_fp32_correctness`

**Statement:**
```lean
specification rope_fp32_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec M (kernel .fp32 .fp32 p) s = some t ∧
      t.mem = referenceMemory M .fp32 .fp32 p s ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `rope_software_fp32`

<details><summary>docstring</summary>

```
/-- Every output uses separately rounded binary32 products and add/subtract.
The independent reference contains no callback from the surrounding backend. -/
```
</details>

**Statement:**
```lean
specification rope_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (p : Parameters) (s : State (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (FP.Software.scalarFP32 rest cfg)
      (kernel .fp32 .fp32 p) s = some t ∧
      t.mem = roundedMemory cfg p s ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
