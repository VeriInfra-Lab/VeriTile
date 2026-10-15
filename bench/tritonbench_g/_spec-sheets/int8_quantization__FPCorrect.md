# Spec sheet — `bench/tritonbench_g/int8_quantization/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/int8_quantization/int8_quantization.py`

## Public theorem: `per_block_scaled_fp32`

<details><summary>docstring</summary>

```
/-- Both outputs of the accepted supplied-scale, pre-rounding stage. -/
```
</details>

**Statement:**
```lean
specification per_block_scaled_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (hregions : p.out ≠ p.scaleOut) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ i, row p s i < p.rows → t.mem p.out (offset p s i) = .mk .real (value M p s i)) ∧
      t.mem p.scaleOut (scaleOffset p s) = .mk .real (scale M p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hregions : p.out ≠ p.scaleOut`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `per_block_scaled_k_software`

<details><summary>docstring</summary>

```
/-- Concrete per-operation fp32 arithmetic for the K preScale=1 profile. -/
```
</details>

**Statement:**
```lean
specification per_block_scaled_k_software [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (hregions : p.out ≠ p.scaleOut) (hscale : p.preScale = 1) :
    ∃ t, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (kernel p) s = some t ∧
      (∀ i, row p s i < p.rows → t.mem p.out (offset p s i) = .mk .real (roundedValue cfg p s i)) ∧
      t.mem p.scaleOut (scaleOffset p s) = .mk .real ((s.mem p.suppliedScale (scaleOffset p s)).read .real) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hregions : p.out ≠ p.scaleOut`
- `hscale : p.preScale = 1`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
