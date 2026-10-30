# Spec sheet — `bench/tritonbench_g/swiglu_fwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/swiglu_fwd/swiglu_fwd.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Both complete floating executions have equal state, memory and failure behavior;
only three natural row-address products commute, with all floating operations retained. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (input partner output : Storage)
    (X Y OUT : RegionName) (sx sy so N B : Nat) :
    program input partner output Bool.false X Y OUT sx sy so N B ≡[assumptions]
      program input partner output Bool.true X Y OUT sx sy so N B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`, `typed_fp32_fp32`, `typed_fp32_fp16`, `typed_fp32_bf16`, `typed_fp32_fp64`, `typed_fp32_f8e4`, `typed_fp32_f8e5`, `typed_fp16_fp32`, `typed_fp16_fp16`, `typed_fp16_bf16`, `typed_fp16_fp64`, `typed_fp16_f8e4`, `typed_fp16_f8e5`, `typed_bf16_fp32`, `typed_bf16_fp16`, `typed_bf16_bf16`, `typed_bf16_fp64`, `typed_bf16_f8e4`, `typed_bf16_f8e5`, `typed_fp64_fp32`, `typed_fp64_fp16`, `typed_fp64_bf16`, `typed_fp64_fp64`, `typed_fp64_f8e4`, `typed_fp64_f8e5`, `typed_f8e4_fp32`, `typed_f8e4_fp16`, `typed_f8e4_bf16`, `typed_f8e4_fp64`, `typed_f8e4_f8e4`, `typed_f8e4_f8e5`, `typed_f8e5_fp32`, `typed_f8e5_fp16`, `typed_f8e5_bf16`, `typed_f8e5_fp64`, `typed_f8e5_f8e4`, `typed_f8e5_f8e5`

<details><summary><code>program</code></summary>

```lean
def program (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : Program where
  kernel := typedKernel input partner output v X Y OUT sx sy so N B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typedKernel</code></summary>

```
/-- Both complete source variants for all independent floating storage triples. -/
```
```lean
def typedKernel (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match input, partner with
  | .fp32, .fp32 => typed_fp32_fp32 output v X Y OUT sx sy so N B
  | .fp32, .fp16 => typed_fp32_fp16 output v X Y OUT sx sy so N B
  | .fp32, .bf16 => typed_fp32_bf16 output v X Y OUT sx sy so N B
  | .fp32, .fp64 => typed_fp32_fp64 output v X Y OUT sx sy so N B
  | .fp32, .f8e4 => typed_fp32_f8e4 output v X Y OUT sx sy so N B
  | .fp32, .f8e5 => typed_fp32_f8e5 output v X Y OUT sx sy so N B
  | .fp16, .fp32 => typed_fp16_fp32 output v X Y OUT sx sy so N B
  | .fp16, .fp16 => typed_fp16_fp16 output v X Y OUT sx sy so N B
  | .fp16, .bf16 => typed_fp16_bf16 output v X Y OUT sx sy so N B
  | .fp16, .fp64 => typed_fp16_fp64 output v X Y OUT sx sy so N B
  | .fp16, .f8e4 => typed_fp16_f8e4 output v X Y OUT sx sy so N B
  | .fp16, .f8e5 => typed_fp16_f8e5 output v X Y OUT sx sy so N B
  | .bf16, .fp32 => typed_bf16_fp32 output v X Y OUT sx sy so N B
  | .bf16, .fp16 => typed_bf16_fp16 output v X Y OUT sx sy so N B
  | .bf16, .bf16 => typed_bf16_bf16 output v X Y OUT sx sy so N B
  | .bf16, .fp64 => typed_bf16_fp64 output v X Y OUT sx sy so N B
  | .bf16, .f8e4 => typed_bf16_f8e4 output v X Y OUT sx sy so N B
  | .bf16, .f8e5 => typed_bf16_f8e5 output v X Y OUT sx sy so N B
  | .fp64, .fp32 => typed_fp64_fp32 output v X Y OUT sx sy so N B
  | .fp64, .fp16 => typed_fp64_fp16 output v X Y OUT sx sy so N B
  | .fp64, .bf16 => typed_fp64_bf16 output v X Y OUT sx sy so N B
  | .fp64, .fp64 => typed_fp64_fp64 output v X Y OUT sx sy so N B
  | .fp64, .f8e4 => typed_fp64_f8e4 output v X Y OUT sx sy so N B
  | .fp64, .f8e5 => typed_fp64_f8e5 output v X Y OUT sx sy so N B
  | .f8e4, .fp32 => typed_f8e4_fp32 output v X Y OUT sx sy so N B
  | .f8e4, .fp16 => typed_f8e4_fp16 output v X Y OUT sx sy so N B
  | .f8e4, .bf16 => typed_f8e4_bf16 output v X Y OUT sx sy so N B
  | .f8e4, .fp64 => typed_f8e4_fp64 output v X Y OUT sx sy so N B
  | .f8e4, .f8e4 => typed_f8e4_f8e4 output v X Y OUT sx sy so N B
  | .f8e4, .f8e5 => typed_f8e4_f8e5 output v X Y OUT sx sy so N B
  | .f8e5, .fp32 => typed_f8e5_fp32 output v X Y OUT sx sy so N B
  | .f8e5, .fp16 => typed_f8e5_fp16 output v X Y OUT sx sy so N B
  | .f8e5, .bf16 => typed_f8e5_bf16 output v X Y OUT sx sy so N B
  | .f8e5, .fp64 => typed_f8e5_fp64 output v X Y OUT sx sy so N B
  | .f8e5, .f8e4 => typed_f8e5_f8e4 output v X Y OUT sx sy so N B
  | .f8e5, .f8e5 => typed_f8e5_f8e5 output v X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp32_fp32</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp32_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp32_fp16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp32_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp32_bf16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp32_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp32_fp64</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp32_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp32_f8e4</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp32_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp32_f8e5</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp32_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp16_fp32</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp16_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp16_fp16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp16_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp16_bf16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp16_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp16_fp64</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp16_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp16_f8e4</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp16_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp16_f8e5</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp16_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_bf16_fp32</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_bf16_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_bf16_fp16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_bf16_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_bf16_bf16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_bf16_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_bf16_fp64</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_bf16_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_bf16_f8e4</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_bf16_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_bf16_f8e5</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_bf16_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp64_fp32</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp64_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp64_fp16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp64_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp64_bf16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp64_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp64_fp64</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp64_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp64_f8e4</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp64_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_fp64_f8e5</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_fp64_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e4_fp32</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e4_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e4_fp16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e4_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e4_bf16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e4_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e4_fp64</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e4_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e4_f8e4</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e4_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e4_f8e5</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e4_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e5_fp32</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e5_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e5_fp16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e5_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e5_bf16</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e5_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e5_fp64</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e5_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e5_f8e4</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e5_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>

<details><summary><code>typed_f8e5_f8e5</code></summary>

```
/-- Full source with fixed input storage and independent output storage. -/
```
```lean
def typed_f8e5_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B
```
</details>
