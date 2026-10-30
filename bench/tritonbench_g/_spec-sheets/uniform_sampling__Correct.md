# Spec sheet — `bench/tritonbench_g/uniform_sampling/Correct.lean`

**Python source:** `bench/tritonbench_g/uniform_sampling/uniform_sampling.py`

## Public theorem: `uniform_sampling_correctness`

**Statement:**
```lean
specification uniform_sampling_correctness (storage : Storage) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        Random.IntegerType.i64.inRange seed = Bool.true →
        Random.IntegerType.i64.inRange offset = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (original storage y N seed offset lower upper B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected seed offset lower upper B u)) ∧
      RealIO.Implements (original storage y N seed offset lower upper B hB).toAlgKernel
        y N B seed offset lower upper)
```

**Assumptions / layout contracts:**
- `hB : SourceDomain.LegalBlock B`

**Closed-form spec defs (transitive):** `original`, `source`, `indexCommuted`

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : ComputeKernel :=
  (source storage Bool.false y N seed offset lower upper B hB).eraseDType
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

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : ComputeKernel :=
  (source storage Bool.true y N seed offset lower upper B hB).eraseDType
```
</details>

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness (storage : Storage) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        Random.IntegerType.i64.inRange seed = Bool.true →
        Random.IntegerType.i64.inRange offset = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (indexCommuted storage y N seed offset lower upper B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected seed offset lower upper B u)) ∧
      RealIO.Implements (indexCommuted storage y N seed offset lower upper B hB).toAlgKernel
        y N B seed offset lower upper)
```

**Assumptions / layout contracts:**
- `hB : SourceDomain.LegalBlock B`

**Closed-form spec defs (transitive):** `indexCommuted`, `source`, `original`

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : ComputeKernel :=
  (source storage Bool.true y N seed offset lower upper B hB).eraseDType
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

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (y : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : ℝ) (B : Nat) (hB : SourceDomain.LegalBlock B) : ComputeKernel :=
  (source storage Bool.false y N seed offset lower upper B hB).eraseDType
```
</details>
