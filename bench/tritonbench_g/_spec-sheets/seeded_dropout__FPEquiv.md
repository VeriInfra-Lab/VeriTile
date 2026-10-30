# Spec sheet — `bench/tritonbench_g/seeded_dropout/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/seeded_dropout/seeded_dropout.py`

## Public theorem: `seeded_dropout_equiv`

**Statement:**
```lean
specification seeded_dropout_equiv (storage : Storage) (kind : Random.IntegerType)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    original storage kind x y N p seed B hB ≡[assumptions]
      indexCommuted storage kind x y N p seed B hB
```

**Assumptions / layout contracts:**
- `hB : SourceDomain.LegalBlock B`

**Closed-form spec defs (transitive):** `SeedCarrier`, `original`, `assumptions`, `indexCommuted`, `source`

<details><summary><code>SeedCarrier</code></summary>

```lean
def SeedCarrier : Random.IntegerType → Type
  | .i32 | .i64 => Int
  | .u32 | .u64 => Nat
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage kind Bool.false x y N p seed B hB
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
def indexCommuted (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) : Program where
  kernel := source storage kind Bool.true x y N p seed B hB
  domain := []
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (storage : Storage) (seedKind : Random.IntegerType) (swapped : Bool) :
    sourceType seedKind :=
  match storage, seedKind, swapped with
  | .fp32, .i32, Bool.false => seededDropoutSource%[tl.float32, tl.int32, original]
  | .fp32, .i32, Bool.true => seededDropoutSource%[tl.float32, tl.int32, indexCommuted]
  | .fp32, .i64, Bool.false => seededDropoutSource%[tl.float32, tl.int64, original]
  | .fp32, .i64, Bool.true => seededDropoutSource%[tl.float32, tl.int64, indexCommuted]
  | .fp32, .u32, Bool.false => seededDropoutSource%[tl.float32, tl.uint32, original]
  | .fp32, .u32, Bool.true => seededDropoutSource%[tl.float32, tl.uint32, indexCommuted]
  | .fp32, .u64, Bool.false => seededDropoutSource%[tl.float32, tl.uint64, original]
  | .fp32, .u64, Bool.true => seededDropoutSource%[tl.float32, tl.uint64, indexCommuted]
  | .fp16, .i32, Bool.false => seededDropoutSource%[tl.float16, tl.int32, original]
  | .fp16, .i32, Bool.true => seededDropoutSource%[tl.float16, tl.int32, indexCommuted]
  | .fp16, .i64, Bool.false => seededDropoutSource%[tl.float16, tl.int64, original]
  | .fp16, .i64, Bool.true => seededDropoutSource%[tl.float16, tl.int64, indexCommuted]
  | .fp16, .u32, Bool.false => seededDropoutSource%[tl.float16, tl.uint32, original]
  | .fp16, .u32, Bool.true => seededDropoutSource%[tl.float16, tl.uint32, indexCommuted]
  | .fp16, .u64, Bool.false => seededDropoutSource%[tl.float16, tl.uint64, original]
  | .fp16, .u64, Bool.true => seededDropoutSource%[tl.float16, tl.uint64, indexCommuted]
  | .bf16, .i32, Bool.false => seededDropoutSource%[tl.bfloat16, tl.int32, original]
  | .bf16, .i32, Bool.true => seededDropoutSource%[tl.bfloat16, tl.int32, indexCommuted]
  | .bf16, .i64, Bool.false => seededDropoutSource%[tl.bfloat16, tl.int64, original]
  | .bf16, .i64, Bool.true => seededDropoutSource%[tl.bfloat16, tl.int64, indexCommuted]
  | .bf16, .u32, Bool.false => seededDropoutSource%[tl.bfloat16, tl.uint32, original]
  | .bf16, .u32, Bool.true => seededDropoutSource%[tl.bfloat16, tl.uint32, indexCommuted]
  | .bf16, .u64, Bool.false => seededDropoutSource%[tl.bfloat16, tl.uint64, original]
  | .bf16, .u64, Bool.true => seededDropoutSource%[tl.bfloat16, tl.uint64, indexCommuted]
  | .fp64, .i32, Bool.false => seededDropoutSource%[tl.float64, tl.int32, original]
  | .fp64, .i32, Bool.true => seededDropoutSource%[tl.float64, tl.int32, indexCommuted]
  | .fp64, .i64, Bool.false => seededDropoutSource%[tl.float64, tl.int64, original]
  | .fp64, .i64, Bool.true => seededDropoutSource%[tl.float64, tl.int64, indexCommuted]
  | .fp64, .u32, Bool.false => seededDropoutSource%[tl.float64, tl.uint32, original]
  | .fp64, .u32, Bool.true => seededDropoutSource%[tl.float64, tl.uint32, indexCommuted]
  | .fp64, .u64, Bool.false => seededDropoutSource%[tl.float64, tl.uint64, original]
  | .fp64, .u64, Bool.true => seededDropoutSource%[tl.float64, tl.uint64, indexCommuted]
  | .f8e4, .i32, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.int32, original]
  | .f8e4, .i32, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e4, .i64, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.int64, original]
  | .f8e4, .i64, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e4, .u32, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.uint32, original]
  | .f8e4, .u32, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.uint32, indexCommuted]
  | .f8e4, .u64, Bool.false => seededDropoutSource%[tl.float8e4nv, tl.uint64, original]
  | .f8e4, .u64, Bool.true => seededDropoutSource%[tl.float8e4nv, tl.uint64, indexCommuted]
  | .f8e5, .i32, Bool.false => seededDropoutSource%[tl.float8e5, tl.int32, original]
  | .f8e5, .i32, Bool.true => seededDropoutSource%[tl.float8e5, tl.int32, indexCommuted]
  | .f8e5, .i64, Bool.false => seededDropoutSource%[tl.float8e5, tl.int64, original]
  | .f8e5, .i64, Bool.true => seededDropoutSource%[tl.float8e5, tl.int64, indexCommuted]
  | .f8e5, .u32, Bool.false => seededDropoutSource%[tl.float8e5, tl.uint32, original]
  | .f8e5, .u32, Bool.true => seededDropoutSource%[tl.float8e5, tl.uint32, indexCommuted]
  | .f8e5, .u64, Bool.false => seededDropoutSource%[tl.float8e5, tl.uint64, original]
  | .f8e5, .u64, Bool.true => seededDropoutSource%[tl.float8e5, tl.uint64, indexCommuted]
```
</details>
