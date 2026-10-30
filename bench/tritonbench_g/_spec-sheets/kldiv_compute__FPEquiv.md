# Spec sheet — `bench/tritonbench_g/kldiv_compute/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/kldiv_compute/kldiv_compute.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Complete typed state, memory and failure equivalence for every admitted
source floating triple and every initial state, without a numerical rewrite assumption. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (x y output : Storage)
    (_hSource : sourceSupported x y = Bool.true)
    (X Y O : RegionName) (N B : Nat) :
    program x y output Bool.false X Y O N B ≡[assumptions]
      program x y output Bool.true X Y O N B
```

**Assumptions / layout contracts:**
- `_hSource : sourceSupported x y = Bool.true`

**Closed-form spec defs (transitive):** `sourceSupported`, `program`, `assumptions`, `kernel`, `kernel_fp32_fp32`, `kernel_fp32_fp16`, `kernel_fp32_bf16`, `kernel_fp32_fp64`, `kernel_fp32_f8e4`, `kernel_fp32_f8e5`, `kernel_fp16_fp32`, `kernel_fp16_fp16`, `kernel_fp16_bf16`, `kernel_fp16_fp64`, `kernel_fp16_f8e4`, `kernel_fp16_f8e5`, `kernel_bf16_fp32`, `kernel_bf16_fp16`, `kernel_bf16_bf16`, `kernel_bf16_fp64`, `kernel_bf16_f8e4`, `kernel_bf16_f8e5`, `kernel_fp64_fp32`, `kernel_fp64_fp16`, `kernel_fp64_bf16`, `kernel_fp64_fp64`, `kernel_fp64_f8e4`, `kernel_fp64_f8e5`, `kernel_f8e4_fp32`, `kernel_f8e4_fp16`, `kernel_f8e4_bf16`, `kernel_f8e4_fp64`, `kernel_f8e4_f8e4`, `kernel_f8e4_f8e5`, `kernel_f8e5_fp32`, `kernel_f8e5_fp16`, `kernel_f8e5_bf16`, `kernel_f8e5_fp64`, `kernel_f8e5_f8e4`, `kernel_f8e5_f8e5`

<details><summary><code>sourceSupported</code></summary>

```
/-- Pinned Triton rejects both-fp8 input pairs before the logarithm: division
leaves fp8 or fp16, while `tl.log` accepts only fp32/fp64. This classifier does
not establish target lowering or general native support. -/
```
```lean
def sourceSupported (x y : Storage) : Bool :=
  match x, y with
  | .f8e4, .f8e4 | .f8e4, .f8e5 | .f8e5, .f8e4 | .f8e5, .f8e5 => Bool.false
  | _, _ => Bool.true

syntax "kldivSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(kldivSource%[$xType:tritonDType, $yType:tritonDType, $outType:tritonDType,
      $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| block_start = pid * $(B))
      else `(tritonStmt| block_start = $(B) * pid)
    let xWide : Bool ← match xType with
      | `(tritonDType| tl.float64) => pure Bool.true
      | _ => pure Bool.false
    let yWide : Bool ← match yType with
      | `(tritonDType| tl.float64) => pure Bool.true
      | _ => pure Bool.false
    let wide : Bool := xWide || yWide
    let same : Bool ← match outType with
      | `(tritonDType| tl.float64) => pure wide
      | `(tritonDType| tl.float32) => pure (!wide)
      | _ => pure Bool.false
    let store : TSyntax `tritonStmt ←
      if same then `(tritonStmt| tl.store(O + offsets, output, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, output.to($outType), mask=mask))
    `(fun (X Y O : RegionName) (N B : Nat) =>
      ComputeKernel.mk [X, Y] [O] (triton {
        tl.region X = $xType, Y = $yType, O = $outType
        pid = tl.program_id(0)
        $blockStart
        offsets = block_start + tl.arange(0, $(B))
        mask = offsets < $(N)
        x = tl.load(X + offsets, mask=mask)
        y = tl.load(Y + offsets, mask=mask)
        output = x * tl.log(x / y)
        $store
      }).surfaceBody)
```
</details>

<details><summary><code>program</code></summary>

```lean
def program (x y output : Storage) (v : Bool)
    (X Y O : RegionName) (N B : Nat) : Program where
  kernel := kernel x y output v X Y O N B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (x y output : Storage) (v : Bool) : Source :=
  match x, y with
  | .fp32, .fp32 => kernel_fp32_fp32 output v
  | .fp32, .fp16 => kernel_fp32_fp16 output v
  | .fp32, .bf16 => kernel_fp32_bf16 output v
  | .fp32, .fp64 => kernel_fp32_fp64 output v
  | .fp32, .f8e4 => kernel_fp32_f8e4 output v
  | .fp32, .f8e5 => kernel_fp32_f8e5 output v
  | .fp16, .fp32 => kernel_fp16_fp32 output v
  | .fp16, .fp16 => kernel_fp16_fp16 output v
  | .fp16, .bf16 => kernel_fp16_bf16 output v
  | .fp16, .fp64 => kernel_fp16_fp64 output v
  | .fp16, .f8e4 => kernel_fp16_f8e4 output v
  | .fp16, .f8e5 => kernel_fp16_f8e5 output v
  | .bf16, .fp32 => kernel_bf16_fp32 output v
  | .bf16, .fp16 => kernel_bf16_fp16 output v
  | .bf16, .bf16 => kernel_bf16_bf16 output v
  | .bf16, .fp64 => kernel_bf16_fp64 output v
  | .bf16, .f8e4 => kernel_bf16_f8e4 output v
  | .bf16, .f8e5 => kernel_bf16_f8e5 output v
  | .fp64, .fp32 => kernel_fp64_fp32 output v
  | .fp64, .fp16 => kernel_fp64_fp16 output v
  | .fp64, .bf16 => kernel_fp64_bf16 output v
  | .fp64, .fp64 => kernel_fp64_fp64 output v
  | .fp64, .f8e4 => kernel_fp64_f8e4 output v
  | .fp64, .f8e5 => kernel_fp64_f8e5 output v
  | .f8e4, .fp32 => kernel_f8e4_fp32 output v
  | .f8e4, .fp16 => kernel_f8e4_fp16 output v
  | .f8e4, .bf16 => kernel_f8e4_bf16 output v
  | .f8e4, .fp64 => kernel_f8e4_fp64 output v
  | .f8e4, .f8e4 => kernel_f8e4_f8e4 output v
  | .f8e4, .f8e5 => kernel_f8e4_f8e5 output v
  | .f8e5, .fp32 => kernel_f8e5_fp32 output v
  | .f8e5, .fp16 => kernel_f8e5_fp16 output v
  | .f8e5, .bf16 => kernel_f8e5_bf16 output v
  | .f8e5, .fp64 => kernel_f8e5_fp64 output v
  | .f8e5, .f8e4 => kernel_f8e5_f8e4 output v
  | .f8e5, .f8e5 => kernel_f8e5_f8e5 output v
```
</details>

<details><summary><code>kernel_fp32_fp32</code></summary>

```lean
def kernel_fp32_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp32_fp16</code></summary>

```lean
def kernel_fp32_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp32_bf16</code></summary>

```lean
def kernel_fp32_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp32_fp64</code></summary>

```lean
def kernel_fp32_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp32_f8e4</code></summary>

```lean
def kernel_fp32_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp32_f8e5</code></summary>

```lean
def kernel_fp32_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp16_fp32</code></summary>

```lean
def kernel_fp16_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp16_fp16</code></summary>

```lean
def kernel_fp16_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp16_bf16</code></summary>

```lean
def kernel_fp16_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp16_fp64</code></summary>

```lean
def kernel_fp16_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp16_f8e4</code></summary>

```lean
def kernel_fp16_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp16_f8e5</code></summary>

```lean
def kernel_fp16_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_bf16_fp32</code></summary>

```lean
def kernel_bf16_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_bf16_fp16</code></summary>

```lean
def kernel_bf16_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_bf16_bf16</code></summary>

```lean
def kernel_bf16_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_bf16_fp64</code></summary>

```lean
def kernel_bf16_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_bf16_f8e4</code></summary>

```lean
def kernel_bf16_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_bf16_f8e5</code></summary>

```lean
def kernel_bf16_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp64_fp32</code></summary>

```lean
def kernel_fp64_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp64_fp16</code></summary>

```lean
def kernel_fp64_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp64_bf16</code></summary>

```lean
def kernel_fp64_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp64_fp64</code></summary>

```lean
def kernel_fp64_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp64_f8e4</code></summary>

```lean
def kernel_fp64_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_fp64_f8e5</code></summary>

```lean
def kernel_fp64_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e4_fp32</code></summary>

```lean
def kernel_f8e4_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e4_fp16</code></summary>

```lean
def kernel_f8e4_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e4_bf16</code></summary>

```lean
def kernel_f8e4_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e4_fp64</code></summary>

```lean
def kernel_f8e4_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e4_f8e4</code></summary>

```lean
def kernel_f8e4_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e4_f8e5</code></summary>

```lean
def kernel_f8e4_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e5_fp32</code></summary>

```lean
def kernel_f8e5_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e5_fp16</code></summary>

```lean
def kernel_f8e5_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e5_bf16</code></summary>

```lean
def kernel_f8e5_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e5_fp64</code></summary>

```lean
def kernel_f8e5_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e5_f8e4</code></summary>

```lean
def kernel_f8e5_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted]
```
</details>

<details><summary><code>kernel_f8e5_f8e5</code></summary>

```lean
def kernel_f8e5_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e5, commuted]
```
</details>
