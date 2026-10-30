# Spec sheet — `bench/tritonbench_g/quantize_global/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/quantize_global/quantize_global.py`

## Public theorem: `quantize_global_scaled_fp32`

<details><summary>docstring</summary>

```
/-- Actual binary32 arithmetic for the accepted pre-rounding stage. The
unmasked supplied scale load, tail mask, primitive order and frame are kept. -/
```
</details>

**Statement:**
```lean
specification quantize_global_scaled_fp32 (rest : Algebra Word) (cfg : FP.Config)
    (s : State Word) (X AbsmaxInv Out : RegionName) (N B : Nat) :
    ∃ t, FP.Structural.exec (backend rest cfg) (source X AbsmaxInv Out N B) s = some t ∧
      (∀ i, TypedScaled.offset s B i < N → t.mem Out (TypedScaled.offset s B i) = .mk .real (value cfg s X AbsmaxInv B i)) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i, TypedScaled.offset s B i < N → o ≠ TypedScaled.offset s B i) → t.mem r o = s.mem r o)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
