# Spec sheet — `bench/tritonbench_g/pow_scalar_tensor/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/pow_scalar_tensor/pow_scalar_tensor.py`

## Public theorem: `floating_source_equivalence`

<details><summary>docstring</summary>

```
/-- Full typed state/failure equivalence for the complete fp32-exponent source,
both constexpr branches and every formal floating destination. Only the
natural tile-offset product changes, so no numerical FP assumptions apply.
Signed strides and the separate native int32-exponent path are not covered. -/
```
</details>

**Statement:**
```lean
specification floating_source_equivalence (output : TypedSource.FloatingOutput)
    (oneTile : Bool) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    program output oneTile Bool.false base X Y sx sy N tasks tiles B ≡[assumptions]
      program output oneTile Bool.true base X Y sx sy N tasks tiles B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `kernel`, `fp32Kernel`, `fp64Kernel`, `fp16Kernel`, `bf16Kernel`, `f8e4Kernel`, `f8e5Kernel`

<details><summary><code>program</code></summary>

```lean
def program (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) : Program where
  kernel := TypedSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

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

## Public theorem: `integer_source_equivalence`

<details><summary>docstring</summary>

```
/-- Complete state/failure equality for the native int32 exponent source and
all six floating destinations. The original integer values, range checks and
native powi callback are retained, including missing-backend failure. Only the
natural offset product changes; no numerical FP assumption is required. -/
```
</details>

**Statement:**
```lean
specification integer_source_equivalence (output : TypedSource.FloatingOutput)
    (oneTile : Bool) (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) :
    integerProgram output oneTile Bool.false base X Y sx sy N tasks tiles B ≡[assumptions]
      integerProgram output oneTile Bool.true base X Y sx sy N tasks tiles B
```

**Closed-form spec defs (transitive):** `integerProgram`, `assumptions`, `kernel`, `fp32Kernel`, `fp64Kernel`, `fp16Kernel`, `bf16Kernel`, `f8e4Kernel`, `f8e5Kernel`

<details><summary><code>integerProgram</code></summary>

```lean
def integerProgram (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (base : ℝ) (X Y : RegionName) (sx sy N tasks tiles B : Nat) : Program where
  kernel := TypedIntegerSource.kernel output oneTile candidate base X Y sx sy N tasks tiles B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

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
