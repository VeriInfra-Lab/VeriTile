# Spec sheet — `bench/tritonbench_g/rowwise_quantization_triton/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/rowwise_quantization_triton/rowwise_quantization_triton.py`

## Public theorem: `rowwise_scaled_typed_correctness`

<details><summary>docstring</summary>

```
/-- The accepted pre-llrint stage: typed floating operations, independent
initial-memory values, both stores, and a complete memory frame. -/
```
</details>

**Statement:**
```lean
specification rowwise_scaled_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) (hregions : p.out ≠ p.maxOut) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ i : TileIndex [p.width], i.1.val < p.columns →
        t.mem p.out (TypedScaled.offset p s i) = .mk .real (scaledValue M p s i)) ∧
      t.mem p.maxOut (s.pids 0) = .mk .fp16 (maximumStored M p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hwidth : 0 < p.width`
- `hregions : p.out ≠ p.maxOut`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `rowwise_scaled_software_correctness`

<details><summary>docstring</summary>

```
/-- Explicit binary32 values for the row and direct binary16 rounding for
its maximum. The encoded maximum's tie/NaN profile is part of this model. -/
```
</details>

**Statement:**
```lean
specification rowwise_scaled_software_correctness [Inhabited Word]
    (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hwidth : 0 < p.width) (hregions : p.out ≠ p.maxOut) :
    ∃ t, FP.Structural.exec (bitBackend rest cfg) (kernel p) s = some t ∧
      (∀ i : TileIndex [p.width], i.1.val < p.columns →
        t.mem p.out (TypedScaled.offset p s i) = .mk .real (bitValue cfg p s i)) ∧
      t.mem p.maxOut (s.pids 0) = .mk .fp16 (bitMaximumStored cfg p s) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hwidth : 0 < p.width`
- `hregions : p.out ≠ p.maxOut`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
