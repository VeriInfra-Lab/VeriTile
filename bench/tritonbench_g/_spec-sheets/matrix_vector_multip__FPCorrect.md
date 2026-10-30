# Spec sheet — `bench/tritonbench_g/matrix_vector_multip/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/matrix_vector_multip/matrix_vector_multip.py`

## Public theorem: `mv_one_block_typed_correctness`

<details><summary>docstring</summary>

```
/-- Full original looping kernel in the one-block profile, including zero
columns. Every cast, initial +0 add and reduction remains in the value graph. -/
```
</details>

**Statement:**
```lean
specification mv_one_block_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (d : FloatDType) (p : Parameters) (s : State α)
    (hw : 0 < p.blockColumns) (hcols : p.columns ≤ p.blockColumns)
    (hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut)) :
    ∃ t, FP.Structural.exec M (kernel d p) s = some t ∧
      t.mem = (reference M d p s).mem ∧
      (∀ i : TileIndex [p.blockRows,1], rowIndex p s i.1 < p.rows →
        t.mem p.out (outputOffset p s i) = .mk d.toTileDType (ofFloat d (outputValues M d p s i))) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hw : 0 < p.blockColumns`
- `hcols : p.columns ≤ p.blockColumns`
- `hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `mv_one_block_software_fp32`

<details><summary>docstring</summary>

```
/-- Concrete binary32 scalar arithmetic and explicitly sequential software
reduction. The native kernel tree must be bound separately. -/
```
</details>

**Statement:**
```lean
specification mv_one_block_software_fp32 [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (p : Parameters)
    (s : State (FP.Value .fp32)) (hw : 0 < p.blockColumns) (hcols : p.columns ≤ p.blockColumns)
    (hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut)) :
    ∃ t, FP.Structural.exec (backend rest cfg) (kernel .fp32 p) s = some t ∧
      (∀ i : TileIndex [p.blockRows,1], rowIndex p s i.1 < p.rows →
        t.mem p.out (outputOffset p s i) = .mk .fp32 (roundedOutput cfg p s i)) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hw : 0 < p.blockColumns`
- `hcols : p.columns ≤ p.blockColumns`
- `hinj : Function.Injective (fun i : Fin p.blockRows => rowIndex p s i * p.strideOut)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
