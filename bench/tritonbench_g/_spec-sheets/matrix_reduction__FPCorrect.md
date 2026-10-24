# Spec sheet — `bench/tritonbench_g/matrix_reduction/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/matrix_reduction/matrix_reduction.py`

## Public theorem: `matrix_reduction_machine_correctness`

<details><summary>docstring</summary>

```
/-- Full source execution in binary16/binary32 encodings with actual fp32
intermediate registers. Independent input-bit maximum, non-NaN and frame
properties; exact half widening/narrowing and order preservation are proved.
No hardware/compiler refinement is inferred from this source-side theorem. -/
```
</details>

**Statement:**
```lean
specification matrix_reduction_machine_correctness
    (storage : Storage) (rest : Algebra Word)
    (s : State Word) (X Y : RegionName) (sm sn sy M N : Nat) (hN : 0 < N)
    (xs : Fin M → Fin N → Bits storage)
    (hinput : ∀ i j, s.mem X (i.val * sm + j.val * sn) =
      .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage (xs i j))))
    (hnotNaN : ∀ i j, isNaN storage (xs i j) = Bool.false) :
    ∃ t, FP.Structural.exec (backend rest) (kernel storage X Y sm sn sy M N) s = some t ∧
      (∀ i : Fin M, ∃ v : Bits storage,
        t.mem Y i.val = .mk (channel storage).toTileDType (ofFloat (channel storage) (encode storage v)) ∧
        IsMaximum storage (xs i) v ∧ isNaN storage v = Bool.false) ∧
      (∀ r o, r ≠ Y ∨ M ≤ o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hN : 0 < N`
- `xs : Fin M → Fin N → Bits storage`
- `hnotNaN : ∀ i j, isNaN storage (xs i j) = Bool.false`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
