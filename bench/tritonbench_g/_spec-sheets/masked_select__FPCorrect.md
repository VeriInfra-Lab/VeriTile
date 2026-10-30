# Spec sheet — `bench/tritonbench_g/masked_select/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/masked_select/masked_select.py`

## Public theorem: `masked_select_typed_correctness`

<details><summary>docstring</summary>

```
/-- Actual typed selection-copy with successful execution, independent input
readback and frame. Only active output addresses must be collision-free. -/
```
</details>

**Statement:**
```lean
specification masked_select_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (d : Storage) (p : Parameters) (s : State α)
    (hinj : ∀ a b, TypedCopy.active p s a → TypedCopy.active p s b →
      outputOffset p s a = outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p s i →
        t.mem p.output (outputOffset p s i) =
          .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.width], TypedCopy.active p s i →
        o ≠ outputOffset p s i) → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hinj : ∀ a b, TypedCopy.active p s a → TypedCopy.active p s b →
      outputOffset p s a = outputOffset p s b → a = b`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
