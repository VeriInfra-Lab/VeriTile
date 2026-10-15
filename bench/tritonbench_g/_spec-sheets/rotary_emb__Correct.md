# Spec sheet — `bench/tritonbench_g/rotary_emb/Correct.lean`

**Python source:** `bench/tritonbench_g/rotary_emb/rotary_emb.py`

## Public theorem: `rotary_emb_correctness`

<details><summary>docstring</summary>

```
/-- Independent real contracts for original and candidate, with successful
execution, the complete pair formula, source-mask safety and the full frame.
-/
```
</details>

**Statement:**
```lean
specification rotary_emb_correctness (p : Parameters)
    (q k cos sin : FullSource.Storage) (Q K Cos Sin : RegionName) :
    Spec.Real (sourceIO p q k cos sin Bool.false Q K Cos Sin ⊨
      rotarySpec p Q K Cos Sin) ∧
    Spec.Real (sourceIO p q k cos sin Bool.true Q K Cos Sin ⊨
      rotarySpec p Q K Cos Sin)
```

**Closed-form spec defs (transitive):** `rotarySpec`, `rotaryPairSpec`

<details><summary><code>rotarySpec</code></summary>

```lean
noncomputable def rotarySpec (p : Parameters) (Q K Cos Sin : RegionName) :
    Formula p × Formula p :=
  (rotaryPairSpec p Q Cos Sin p.qAddress, rotaryPairSpec p K Cos Sin p.kAddress)
```
</details>

<details><summary><code>rotaryPairSpec</code></summary>

```
/-- Interleaved pair rotation from the initial data and trigonometric memory.
Odd and even results deliberately read their own trigonometric coordinates.
-/
```
```lean
noncomputable def rotaryPairSpec (p : Parameters) (X Cos Sin : RegionName)
    (offset : BlockState → Bool → p.Lane → Nat) : Formula p := fun s odd i =>
  let x0 := s.readMem X (offset s Bool.false i)
  let x1 := s.readMem X (offset s Bool.true i)
  let c := s.readMem Cos (p.trigAddress s odd i)
  let sn := s.readMem Sin (p.trigAddress s odd i)
  if odd then x0 * sn + x1 * c else x0 * c - x1 * sn
```
</details>
