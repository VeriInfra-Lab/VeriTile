# Spec sheet — `bench/tritonbench_g/pow_scalar_tensor/Correct.lean`

**Python source:** `bench/tritonbench_g/pow_scalar_tensor/pow_scalar_tensor.py`

## Public theorem: `floating_one_tile_correct`

<details><summary>docstring</summary>

```
/-- Full actual one-tile source correctness with a dimension-general finite
allocation, partial tiles and independent natural input/output strides. -/
```
</details>

**Statement:**
```lean
specification floating_one_tile_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin B → ℝ)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hinput : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0) B i * sx) = xs i)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X)
    (hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y) :
    Spec.Real (Contracts.ExactIO (TypedSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B)
      alloc s Y base (fun i => taskIndex (s.pids 0) B i * sy)
      (fun i => taskIndex (s.pids 0) B i < N) xs)
```

**Assumptions / layout contracts:**
- `xs : Fin B → ℝ`
- `hd : alloc.Disjoint`
- `hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0`
- `hy : Y ∈ alloc.regions`
- `hstride : 0 < sy`
- `hu : s.undef = fun _ _ => 0`
- `hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X`
- `hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y`

**Closed-form spec defs (transitive):** `kernel`, `fp32Kernel`, `fp64Kernel`, `fp16Kernel`, `bf16Kernel`, `f8e4Kernel`, `f8e5Kernel`

<details><summary><code>kernel</code></summary>

```
/-- Both complete branch bodies with unspecified boundary padding, explicit
base fp32 conversion and destination conversion. Candidate commutes only the
natural tile-offset product; task count and the unused cdiv remain present.
The template resolves Python's `out0_bptr.type.element_ty` to the declared
destination label before expanding `.to`; this is source metadata resolution,
not an additional cast or a change to Python's conversion semantics. -/
```
```lean
def kernel (output : FloatingOutput) (oneTile candidate : Bool) : Source :=
  match output with
  | .fp32 => fp32Kernel oneTile candidate
  | .fp64 => fp64Kernel oneTile candidate
  | .fp16 => fp16Kernel oneTile candidate
  | .bf16 => bf16Kernel oneTile candidate
  | .f8e4 => f8e4Kernel oneTile candidate
  | .f8e5 => f8e5Kernel oneTile candidate
```
</details>

<details><summary><code>fp32Kernel</code></summary>

```lean
def fp32Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float32] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float32] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float32] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float32] 0 1
```
</details>

<details><summary><code>fp64Kernel</code></summary>

```lean
def fp64Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float64] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float64] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float64] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float64] 0 1
```
</details>

<details><summary><code>fp16Kernel</code></summary>

```lean
def fp16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float16] 0 1
```
</details>

<details><summary><code>bf16Kernel</code></summary>

```lean
def bf16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.bfloat16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.bfloat16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.bfloat16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.bfloat16] 0 1
```
</details>

<details><summary><code>f8e4Kernel</code></summary>

```lean
def f8e4Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e4nv] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e4nv] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e4nv] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e4nv] 0 1
```
</details>

<details><summary><code>f8e5Kernel</code></summary>

```lean
def f8e5Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e5] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e5] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e5] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e5] 0 1
```
</details>

## Public theorem: `floating_grid_correct`

<details><summary>docstring</summary>

```
/-- Full actual grid-stride source correctness, including all dynamic
iterations and a partial final tile, against independent exponent arrays. -/
```
</details>

**Statement:**
```lean
specification floating_grid_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin tiles → Fin B → ℝ)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0)
    (hinput : ∀ j i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      s.readMem X (taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx) = xs j i)
    (hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx < alloc.extent X)
    (hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sy < alloc.extent Y) :
    Spec.Real (Contracts.ExactIO (TypedSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B)
      alloc s Y base
      (fun p : Fin tiles × Fin B => taskIndex (s.pids 0 + p.1.val * s.numPids 0) B p.2 * sy)
      (fun p => taskIndex (s.pids 0 + p.1.val * s.numPids 0) B p.2 < N)
      (fun p => xs p.1 p.2))
```

**Assumptions / layout contracts:**
- `xs : Fin tiles → Fin B → ℝ`
- `hd : alloc.Disjoint`
- `hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0`
- `hy : Y ∈ alloc.regions`
- `hstride : 0 < sy`
- `hu : s.undef = fun _ _ => 0`
- `hdisj : X ≠ Y`
- `hgrid : 0 < s.numPids 0`
- `hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sx < alloc.extent X`
- `hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0 + j.val * s.numPids 0) B i < N →
      taskIndex (s.pids 0 + j.val * s.numPids 0) B i * sy < alloc.extent Y`

**Closed-form spec defs (transitive):** `kernel`, `fp32Kernel`, `fp64Kernel`, `fp16Kernel`, `bf16Kernel`, `f8e4Kernel`, `f8e5Kernel`

<details><summary><code>kernel</code></summary>

```
/-- Both complete branch bodies with unspecified boundary padding, explicit
base fp32 conversion and destination conversion. Candidate commutes only the
natural tile-offset product; task count and the unused cdiv remain present.
The template resolves Python's `out0_bptr.type.element_ty` to the declared
destination label before expanding `.to`; this is source metadata resolution,
not an additional cast or a change to Python's conversion semantics. -/
```
```lean
def kernel (output : FloatingOutput) (oneTile candidate : Bool) : Source :=
  match output with
  | .fp32 => fp32Kernel oneTile candidate
  | .fp64 => fp64Kernel oneTile candidate
  | .fp16 => fp16Kernel oneTile candidate
  | .bf16 => bf16Kernel oneTile candidate
  | .f8e4 => f8e4Kernel oneTile candidate
  | .f8e5 => f8e5Kernel oneTile candidate
```
</details>

<details><summary><code>fp32Kernel</code></summary>

```lean
def fp32Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float32] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float32] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float32] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float32] 0 1
```
</details>

<details><summary><code>fp64Kernel</code></summary>

```lean
def fp64Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float64] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float64] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float64] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float64] 0 1
```
</details>

<details><summary><code>fp16Kernel</code></summary>

```lean
def fp16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float16] 0 1
```
</details>

<details><summary><code>bf16Kernel</code></summary>

```lean
def bf16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.bfloat16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.bfloat16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.bfloat16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.bfloat16] 0 1
```
</details>

<details><summary><code>f8e4Kernel</code></summary>

```lean
def f8e4Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e4nv] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e4nv] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e4nv] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e4nv] 0 1
```
</details>

<details><summary><code>f8e5Kernel</code></summary>

```lean
def f8e5Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e5] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e5] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e5] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e5] 0 1
```
</details>

## Public theorem: `integer_one_tile_correct`

<details><summary>docstring</summary>

```
/-- Complete int32 single-tile source with arbitrary dimensions, actual trace safety,
finite-flat execution, independent signed-integer zpow and a full raw-memory frame. -/
```
</details>

**Statement:**
```lean
specification integer_one_tile_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin B → Int)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hinput : ∀ i, taskIndex (s.pids 0) B i < N →
      s.readMemTyped .int X (taskIndex (s.pids 0) B i * sx) = some (xs i))
    (hvalid : ∀ i, taskIndex (s.pids 0) B i < N → Int32Bits.inRange (xs i) = Bool.true)
    (hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X)
    (hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y) :
    Spec.Real (IntegerContracts.ExactIO (TypedIntegerSource.kernel output Bool.true candidate base X Y sx sy N tasks tiles B)
      alloc s Y base (fun i => taskIndex (s.pids 0) B i * sy)
      (fun i => taskIndex (s.pids 0) B i < N) xs)
```

**Assumptions / layout contracts:**
- `xs : Fin B → Int`
- `hd : alloc.Disjoint`
- `hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0`
- `hy : Y ∈ alloc.regions`
- `hstride : 0 < sy`
- `hu : s.undef = fun _ _ => 0`
- `hvalid : ∀ i, taskIndex (s.pids 0) B i < N → Int32Bits.inRange (xs i) = Bool.true`
- `hin : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sx < alloc.extent X`
- `hout : ∀ i, taskIndex (s.pids 0) B i < N → taskIndex (s.pids 0) B i * sy < alloc.extent Y`

**Closed-form spec defs (transitive):** `kernel`, `fp32Kernel`, `fp64Kernel`, `fp16Kernel`, `bf16Kernel`, `f8e4Kernel`, `f8e5Kernel`

<details><summary><code>kernel</code></summary>

```
/-- Both complete branch bodies with unspecified boundary padding, explicit
base fp32 conversion and destination conversion. Candidate commutes only the
natural tile-offset product; task count and the unused cdiv remain present.
The template resolves Python's `out0_bptr.type.element_ty` to the declared
destination label before expanding `.to`; this is source metadata resolution,
not an additional cast or a change to Python's conversion semantics. -/
```
```lean
def kernel (output : FloatingOutput) (oneTile candidate : Bool) : Source :=
  match output with
  | .fp32 => fp32Kernel oneTile candidate
  | .fp64 => fp64Kernel oneTile candidate
  | .fp16 => fp16Kernel oneTile candidate
  | .bf16 => bf16Kernel oneTile candidate
  | .f8e4 => f8e4Kernel oneTile candidate
  | .f8e5 => f8e5Kernel oneTile candidate
```
</details>

<details><summary><code>fp32Kernel</code></summary>

```lean
def fp32Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float32] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float32] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float32] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float32] 0 1
```
</details>

<details><summary><code>fp64Kernel</code></summary>

```lean
def fp64Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float64] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float64] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float64] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float64] 0 1
```
</details>

<details><summary><code>fp16Kernel</code></summary>

```lean
def fp16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float16] 0 1
```
</details>

<details><summary><code>bf16Kernel</code></summary>

```lean
def bf16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.bfloat16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.bfloat16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.bfloat16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.bfloat16] 0 1
```
</details>

<details><summary><code>f8e4Kernel</code></summary>

```lean
def f8e4Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e4nv] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e4nv] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e4nv] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e4nv] 0 1
```
</details>

<details><summary><code>f8e5Kernel</code></summary>

```lean
def f8e5Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e5] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e5] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e5] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e5] 0 1
```
</details>

## Public theorem: `integer_grid_correct`

<details><summary>docstring</summary>

```
/-- Complete int32 grid-stride source with arbitrary dimensions, actual trace safety,
finite-flat execution, independent signed-integer zpow and a full raw-memory frame. -/
```
</details>

**Statement:**
```lean
specification integer_grid_correct (output : TypedSource.FloatingOutput) (candidate : Bool)
    (alloc : FlatAlloc) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat)
    (s : BlockState) (xs : Fin tiles → Fin B → Int)
    (hd : alloc.Disjoint) (hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0)
    (hy : Y ∈ alloc.regions) (hstride : 0 < sy) (hu : s.undef = fun _ _ => 0)
    (hdisj : X ≠ Y) (hgrid : 0 < s.numPids 0)
    (hinput : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      s.readMemTyped .int X (taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx) = some (xs j i))
    (hvalid : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      Int32Bits.inRange (xs j i) = Bool.true)
    (hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx < alloc.extent X)
    (hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sy < alloc.extent Y) :
    Spec.Real (IntegerContracts.ExactIO (TypedIntegerSource.kernel output Bool.false candidate base X Y sx sy N tasks tiles B)
      alloc s Y base
      (fun p : Fin tiles × Fin B => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 * sy)
      (fun p => taskIndex (s.pids 0+p.1.val*s.numPids 0) B p.2 < N)
      (fun p => xs p.1 p.2))
```

**Assumptions / layout contracts:**
- `xs : Fin tiles → Fin B → Int`
- `hd : alloc.Disjoint`
- `hcov : ∀ r, r ∉ alloc.regions → alloc.extent r = 0`
- `hy : Y ∈ alloc.regions`
- `hstride : 0 < sy`
- `hu : s.undef = fun _ _ => 0`
- `hdisj : X ≠ Y`
- `hgrid : 0 < s.numPids 0`
- `hvalid : ∀ j i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      Int32Bits.inRange (xs j i) = Bool.true`
- `hin : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sx < alloc.extent X`
- `hout : ∀ (j : Fin tiles) i, taskIndex (s.pids 0+j.val*s.numPids 0) B i < N →
      taskIndex (s.pids 0+j.val*s.numPids 0) B i * sy < alloc.extent Y`

**Closed-form spec defs (transitive):** `kernel`, `fp32Kernel`, `fp64Kernel`, `fp16Kernel`, `bf16Kernel`, `f8e4Kernel`, `f8e5Kernel`

<details><summary><code>kernel</code></summary>

```
/-- Both complete branch bodies with unspecified boundary padding, explicit
base fp32 conversion and destination conversion. Candidate commutes only the
natural tile-offset product; task count and the unused cdiv remain present.
The template resolves Python's `out0_bptr.type.element_ty` to the declared
destination label before expanding `.to`; this is source metadata resolution,
not an additional cast or a change to Python's conversion semantics. -/
```
```lean
def kernel (output : FloatingOutput) (oneTile candidate : Bool) : Source :=
  match output with
  | .fp32 => fp32Kernel oneTile candidate
  | .fp64 => fp64Kernel oneTile candidate
  | .fp16 => fp16Kernel oneTile candidate
  | .bf16 => bf16Kernel oneTile candidate
  | .f8e4 => f8e4Kernel oneTile candidate
  | .f8e5 => f8e5Kernel oneTile candidate
```
</details>

<details><summary><code>fp32Kernel</code></summary>

```lean
def fp32Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float32] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float32] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float32] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float32] 0 1
```
</details>

<details><summary><code>fp64Kernel</code></summary>

```lean
def fp64Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float64] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float64] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float64] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float64] 0 1
```
</details>

<details><summary><code>fp16Kernel</code></summary>

```lean
def fp16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float16] 0 1
```
</details>

<details><summary><code>bf16Kernel</code></summary>

```lean
def bf16Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.bfloat16] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.bfloat16] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.bfloat16] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.bfloat16] 0 1
```
</details>

<details><summary><code>f8e4Kernel</code></summary>

```lean
def f8e4Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e4nv] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e4nv] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e4nv] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e4nv] 0 1
```
</details>

<details><summary><code>f8e5Kernel</code></summary>

```lean
def f8e5Kernel (oneTile candidate : Bool) : Source :=
  match oneTile, candidate with
  | Bool.true, Bool.false => powScalarFloating%[tl.float8e5] 1 0
  | Bool.true, Bool.true => powScalarFloating%[tl.float8e5] 1 1
  | Bool.false, Bool.false => powScalarFloating%[tl.float8e5] 0 0
  | Bool.false, Bool.true => powScalarFloating%[tl.float8e5] 0 1
```
</details>
