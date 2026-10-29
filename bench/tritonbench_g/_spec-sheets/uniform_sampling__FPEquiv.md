# Spec sheet — `bench/tritonbench_g/uniform_sampling/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/uniform_sampling/uniform_sampling.py`

## Public theorem: `uniform_sampling_equiv`

**Statement:**
```lean
specification uniform_sampling_equiv (storage : Storage) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    original storage y N seed offset lower upper B hB ≡[assumptions]
      indexCommuted storage y N seed offset lower upper B hB
```

**Assumptions / layout contracts:**
- `hB : SourceDomain.LegalBlock B`

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`, `source`

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage Bool.false y N seed offset lower upper B hB
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage Bool.true y N seed offset lower upper B hB
  domain := []
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (storage : Storage) (swapped : Bool) : sourceType :=
  match storage, swapped with
  | .fp32, Bool.false => uniformSource%[tl.float32, original]
  | .fp32, Bool.true => uniformSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => uniformSource%[tl.float16, original]
  | .fp16, Bool.true => uniformSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => uniformSource%[tl.bfloat16, original]
  | .bf16, Bool.true => uniformSource%[tl.bfloat16, indexCommuted]
  | .fp64, Bool.false => uniformSource%[tl.float64, original]
  | .fp64, Bool.true => uniformSource%[tl.float64, indexCommuted]
  | .f8e4, Bool.false => uniformSource%[tl.float8e4nv, original]
  | .f8e4, Bool.true => uniformSource%[tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => uniformSource%[tl.float8e5, original]
  | .f8e5, Bool.true => uniformSource%[tl.float8e5, indexCommuted]

#axiomsClean arange_domain
```
</details>
