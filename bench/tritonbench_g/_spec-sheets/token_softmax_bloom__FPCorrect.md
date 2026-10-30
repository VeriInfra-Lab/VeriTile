# Spec sheet — `bench/tritonbench_g/token_softmax_bloom/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/token_softmax_bloom/token_softmax_bloom.py`

## Public theorem: `token_softmax_typed_correctness`

<details><summary>docstring</summary>

```
/-- Full row computation and masked writeback, with independent initial-state
value formula. Every primitive retains its precision and storage conversion.
The native max/exp/sum implementation is a backend binding obligation. -/
```
</details>

**Statement:**
```lean
specification token_softmax_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (d : FloatDType) (p : Parameters) (s : State α)
    (hwidth : 0 < p.width) (hstride : p.outputTokenStride ≠ 0) :
    ∃ t, FP.Structural.exec M (kernel d p) s = some t ∧
      t.mem = (reference M d p s).mem ∧
      (∀ i : TileIndex [p.width], i.1.val < count p s →
        t.mem p.output (outputOffset p s i) =
          .mk d.toTileDType (ofFloat d (outputValues M d p s i))) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hwidth : 0 < p.width`
- `hstride : p.outputTokenStride ≠ 0`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
