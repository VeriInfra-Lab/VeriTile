# Spec sheet — `bench/tritonbench_g/logsumexp_fwd/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/logsumexp_fwd/logsumexp_fwd.py`

## Public theorem: `row_block_fp32`

**Statement:**
```lean
specification row_block_fp32 {α : Type} [Inhabited α] (M : Algebra α)
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) (s : State α) (hB : 0 < B) :
    ∃ final, FP.Structural.exec M (rowBlock x z D B HAS_SCALE scale) s = some final ∧
      final.mem = (s.write z (outputAddress s D B)
        (.mk .real (rowValue M s x D B HAS_SCALE scale))).mem
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `row_block_fp32_with_primitives`

**Statement:**
```lean
specification row_block_fp32_with_primitives [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (primitives : Primitives)
    (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) (s : State (FP.Value .fp32)) (hB : 0 < B) :
    ∃ final, FP.Structural.exec (fp32WithPrimitives rest cfg primitives)
      (rowBlock x z D B HAS_SCALE 1) s = some final ∧
      final.mem = (s.write z (outputAddress s D B)
        (.mk .real (roundedRow cfg primitives s x D B HAS_SCALE))).mem
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `row_block_nonempty`

**Statement:**
```lean
specification row_block_nonempty [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (primitives : Primitives)
    (s : State (FP.Value .fp32)) (hrow : s.pids 0 = 0) (hblock : s.pids 1 = 0) :
    ∃ final, FP.Structural.exec (fp32WithPrimitives rest cfg primitives)
      (rowBlock "x" "z" 1 2 Bool.false 1) s = some final ∧
      final.mem "z" 0 = .mk .real (roundedRow cfg primitives s "x" 1 2 Bool.false) ∧
      (∀ (R : RegionName) (o : Nat), R ≠ "z" ∨ o ≠ 0 → final.mem R o = s.mem R o)
```

**Assumptions / layout contracts:**
- `hrow : s.pids 0 = 0`
- `hblock : s.pids 1 = 0`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
