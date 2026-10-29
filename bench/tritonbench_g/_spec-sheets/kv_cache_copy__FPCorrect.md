# Spec sheet — `bench/tritonbench_g/kv_cache_copy/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/kv_cache_copy/kv_cache_copy.py`

## Public theorem: `kv_cache_copy_typed_correctness`

<details><summary>docstring</summary>

```
/-- Typed execution of the accepted x-block, with two independent input
snapshots. Hypotheses are the existing active-address/caches conditions;
there is no floating numerical law or assumed output value. -/
```
</details>

**Statement:**
```lean
specification kv_cache_copy_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (kd vd : Storage) (p : TypedCopy.Parameters) (s : State α)
    (hcache : p.KCache ≠ p.VCache)
    (hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b →
      vOutputOffset p s a = vOutputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (TypedCopy.source kd vd p) s = some t ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i →
        t.mem p.KCache (outputOffset p.toParameters s i) =
          .mk kd.tag ((s.mem p.K (inputOffset p.toParameters s i)).read kd.tag)) ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i →
        t.mem p.VCache (vOutputOffset p s i) =
          .mk vd.tag ((s.mem p.V (vInputOffset p s i)).read vd.tag)) ∧
      (∀ r o,
        (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o ≠ outputOffset p.toParameters s i) →
        (r ≠ p.VCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o ≠ vOutputOffset p s i) →
        t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hcache : p.KCache ≠ p.VCache`
- `hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b →
      vOutputOffset p s a = vOutputOffset p s b → a = b`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
