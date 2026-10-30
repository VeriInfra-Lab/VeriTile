# Spec sheet — `bench/tritonbench_g/seeded_dropout/Correct.lean`

**Python source:** `bench/tritonbench_g/seeded_dropout/seeded_dropout.py`

## Public theorem: `seeded_dropout_correctness`

**Statement:**
```lean
specification seeded_dropout_correctness (storage : Storage) (kind : Random.IntegerType)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        kind.inRange (seedInt kind seed) = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (original storage kind x y N p seed B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected x p (seedInt kind seed) B u)) ∧
      RealIO.Implements (original storage kind x y N p seed B hB).toAlgKernel
        kind x y N p seed B)
```

**Assumptions / layout contracts:**
- `hB : SourceDomain.LegalBlock B`

**Closed-form spec defs (transitive):** `SeedCarrier`, `seedInt`, `original`, `indexCommuted`, `source`

<details><summary><code>SeedCarrier</code></summary>

```lean
def SeedCarrier : Random.IntegerType → Type
  | .i32 | .i64 => Int
  | .u32 | .u64 => Nat
```
</details>

<details><summary><code>seedInt</code></summary>

```lean
def seedInt (kind : Random.IntegerType) (seed : SeedCarrier kind) : Int :=
  match kind, seed with
  | .i32, s | .i64, s => s
  | .u32, s | .u64, s => Int.ofNat s

syntax "seededDropoutSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(seededDropoutSource%[$storage:tritonDType, $seedDType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let seedTy : TSyntax `term ←
      match seedDType with
      | `(tritonDType| tl.int32) | `(tritonDType| tl.int64) => `(Int)
      | `(tritonDType| tl.uint32) | `(tritonDType| tl.uint64) => `(Nat)
      | _ => Macro.throwError "expected a supported source integer seed type"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| block_start = BLOCK_SIZE_arg * pid)
      else `(tritonStmt| block_start = pid * BLOCK_SIZE_arg)
    `(fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : $seedTy) (BLOCK_SIZE : Nat) (hBlock : SourceDomain.LegalBlock BLOCK_SIZE) =>
      ComputeKernel.mk [x_ptr] [output_ptr] (triton {
  tl.region x_ptr = $storage, output_ptr = $storage
  seed_arg = $((seed : $seedTy))
  tl.source_integer seed_arg = $seedDType
  BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE (RandomDomain.arange_domain hBlock))
  pid = tl.program_id(axis=0)
  $blockStart
  offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  random = tl.rand(seed_arg, offsets)
  x_keep = random > $(p)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, (output).to($storage:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    ComputeKernel := (source storage kind Bool.false x y N p seed B hB).eraseDType
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    ComputeKernel := (source storage kind Bool.true x y N p seed B hB).eraseDType
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

## Public theorem: `index_commuted_correctness`

**Statement:**
```lean
specification index_commuted_correctness (storage : Storage) (kind : Random.IntegerType)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    Spec.Real (
      (∀ u : BlockState, N < 2^31 → u.pid < SourceDomain.grid N B →
        kind.inRange (seedInt kind seed) = Bool.true →
        ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
          (indexCommuted storage kind x y N p seed B hB) u
          (RealSurface.writes y N B u.pid) (RealSurface.expected x p (seedInt kind seed) B u)) ∧
      RealIO.Implements (indexCommuted storage kind x y N p seed B hB).toAlgKernel
        kind x y N p seed B)
```

**Assumptions / layout contracts:**
- `hB : SourceDomain.LegalBlock B`

**Closed-form spec defs (transitive):** `SeedCarrier`, `seedInt`, `indexCommuted`, `original`, `source`

<details><summary><code>SeedCarrier</code></summary>

```lean
def SeedCarrier : Random.IntegerType → Type
  | .i32 | .i64 => Int
  | .u32 | .u64 => Nat
```
</details>

<details><summary><code>seedInt</code></summary>

```lean
def seedInt (kind : Random.IntegerType) (seed : SeedCarrier kind) : Int :=
  match kind, seed with
  | .i32, s | .i64, s => s
  | .u32, s | .u64, s => Int.ofNat s

syntax "seededDropoutSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(seededDropoutSource%[$storage:tritonDType, $seedDType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let seedTy : TSyntax `term ←
      match seedDType with
      | `(tritonDType| tl.int32) | `(tritonDType| tl.int64) => `(Int)
      | `(tritonDType| tl.uint32) | `(tritonDType| tl.uint64) => `(Nat)
      | _ => Macro.throwError "expected a supported source integer seed type"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| block_start = BLOCK_SIZE_arg * pid)
      else `(tritonStmt| block_start = pid * BLOCK_SIZE_arg)
    `(fun (x_ptr output_ptr : RegionName) (n_elements : Nat) (p : ℝ)
      (seed : $seedTy) (BLOCK_SIZE : Nat) (hBlock : SourceDomain.LegalBlock BLOCK_SIZE) =>
      ComputeKernel.mk [x_ptr] [output_ptr] (triton {
  tl.region x_ptr = $storage, output_ptr = $storage
  seed_arg = $((seed : $seedTy))
  tl.source_integer seed_arg = $seedDType
  BLOCK_SIZE_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK_SIZE (RandomDomain.arange_domain hBlock))
  pid = tl.program_id(axis=0)
  $blockStart
  offsets = block_start + tl.arange(0, BLOCK_SIZE_arg)
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  random = tl.rand(seed_arg, offsets)
  x_keep = random > $(p)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, (output).to($storage:tritonDType), mask=mask)
}).surfaceBody)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    ComputeKernel := (source storage kind Bool.true x y N p seed B hB).eraseDType
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (hB : SourceDomain.LegalBlock B) :
    ComputeKernel := (source storage kind Bool.false x y N p seed B hB).eraseDType
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
