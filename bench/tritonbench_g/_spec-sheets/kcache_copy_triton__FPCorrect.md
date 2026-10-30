# Spec sheet — `bench/tritonbench_g/kcache_copy_triton/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/kcache_copy_triton/kcache_copy_triton.py`

## Public theorem: `kcache_copy_typed_correctness`

<details><summary>docstring</summary>

```
/-- Successful raw typed copy for the accepted n_tokens=1 split-x source.
All source metadata/address parameters remain explicit in `Parameters`.
No floating arithmetic, rounding, or finite-value law is assumed. -/
```
</details>

**Statement:**
```lean
specification kcache_copy_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (d : Storage) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], t.mem p.KCache (outputOffset p s i) =
        .mk d.tag ((s.mem p.K (inputOffset p s i)).read d.tag)) ∧
      (∀ r o, (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p s i) →
        t.mem r o = s.mem r o)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
