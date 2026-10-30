# Spec sheet — `bench/tritonbench_g/rmsnorm_implementation/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_implementation/rmsnorm_implementation.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- The whole two-pass program preserves all backend results and failures, all
registers and all memory, including aliases and arbitrary masked-load undefined
values. Only exact natural-address multiplication changes; no numerical atom,
finite-input assumption or successful-output assumption is required. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (input weight output : Storage)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    program input weight output Bool.false X W O sxb sxm sxk sw sob som sok N B eps ≡[assumptions]
      program input weight output Bool.true X W O sxb sxm sxk sw sob som sok N B eps
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`, `originalTyped`

<details><summary><code>program</code></summary>

```lean
def program (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : Program where
  kernel := typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps
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
/-- Candidate changes only the first row-address product's operand order. -/
```
```lean
def typedKernel (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : ComputeKernel :=
  let source := originalTyped input weight output X W O sxb sxm sxk sw sob som sok N B eps
  if v then ComputeKernel.mk source.inputs source.outputs
    (source.surfaceBody.take 2 ++
      [.assign .nat [] "offset_m" (.alg
        (.add .nat .nil
          (.mul .nat .nil (.constNat sxb) (.ref .nat [] "pid_batch"))
          (.mul .nat .nil (.ref .nat [] "pid_m") (.constNat sxm))))] ++
      source.surfaceBody.drop 3)
  else source
```
</details>

<details><summary><code>originalTyped</code></summary>

```
/-- All source statements in both loops, including the uncast second input load
and weight load without `other`. The explicit store conversion models Triton's
implicit conversion to the output pointer element type. -/
```
```lean
def originalTyped (input weight output : Storage) : Source :=
  match input, weight, output with
  | .fp32, .fp32, .fp32 => rmsnormTyped%[tl.float32, tl.float32, tl.float32]
  | .fp32, .fp32, .fp64 => rmsnormTyped%[tl.float32, tl.float32, tl.float64]
  | .fp32, .fp32, .fp16 => rmsnormTyped%[tl.float32, tl.float32, tl.float16]
  | .fp32, .fp32, .bf16 => rmsnormTyped%[tl.float32, tl.float32, tl.bfloat16]
  | .fp32, .fp32, .f8e4 => rmsnormTyped%[tl.float32, tl.float32, tl.float8e4nv]
  | .fp32, .fp32, .f8e5 => rmsnormTyped%[tl.float32, tl.float32, tl.float8e5]
  | .fp32, .fp64, .fp32 => rmsnormTyped%[tl.float32, tl.float64, tl.float32]
  | .fp32, .fp64, .fp64 => rmsnormTyped%[tl.float32, tl.float64, tl.float64]
  | .fp32, .fp64, .fp16 => rmsnormTyped%[tl.float32, tl.float64, tl.float16]
  | .fp32, .fp64, .bf16 => rmsnormTyped%[tl.float32, tl.float64, tl.bfloat16]
  | .fp32, .fp64, .f8e4 => rmsnormTyped%[tl.float32, tl.float64, tl.float8e4nv]
  | .fp32, .fp64, .f8e5 => rmsnormTyped%[tl.float32, tl.float64, tl.float8e5]
  | .fp32, .fp16, .fp32 => rmsnormTyped%[tl.float32, tl.float16, tl.float32]
  | .fp32, .fp16, .fp64 => rmsnormTyped%[tl.float32, tl.float16, tl.float64]
  | .fp32, .fp16, .fp16 => rmsnormTyped%[tl.float32, tl.float16, tl.float16]
  | .fp32, .fp16, .bf16 => rmsnormTyped%[tl.float32, tl.float16, tl.bfloat16]
  | .fp32, .fp16, .f8e4 => rmsnormTyped%[tl.float32, tl.float16, tl.float8e4nv]
  | .fp32, .fp16, .f8e5 => rmsnormTyped%[tl.float32, tl.float16, tl.float8e5]
  | .fp32, .bf16, .fp32 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float32]
  | .fp32, .bf16, .fp64 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float64]
  | .fp32, .bf16, .fp16 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float16]
  | .fp32, .bf16, .bf16 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.bfloat16]
  | .fp32, .bf16, .f8e4 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float8e4nv]
  | .fp32, .bf16, .f8e5 => rmsnormTyped%[tl.float32, tl.bfloat16, tl.float8e5]
  | .fp32, .f8e4, .fp32 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float32]
  | .fp32, .f8e4, .fp64 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float64]
  | .fp32, .f8e4, .fp16 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float16]
  | .fp32, .f8e4, .bf16 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.bfloat16]
  | .fp32, .f8e4, .f8e4 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float8e4nv]
  | .fp32, .f8e4, .f8e5 => rmsnormTyped%[tl.float32, tl.float8e4nv, tl.float8e5]
  | .fp32, .f8e5, .fp32 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float32]
  | .fp32, .f8e5, .fp64 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float64]
  | .fp32, .f8e5, .fp16 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float16]
  | .fp32, .f8e5, .bf16 => rmsnormTyped%[tl.float32, tl.float8e5, tl.bfloat16]
  | .fp32, .f8e5, .f8e4 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float8e4nv]
  | .fp32, .f8e5, .f8e5 => rmsnormTyped%[tl.float32, tl.float8e5, tl.float8e5]
  | .fp64, .fp32, .fp32 => rmsnormTyped%[tl.float64, tl.float32, tl.float32]
  | .fp64, .fp32, .fp64 => rmsnormTyped%[tl.float64, tl.float32, tl.float64]
  | .fp64, .fp32, .fp16 => rmsnormTyped%[tl.float64, tl.float32, tl.float16]
  | .fp64, .fp32, .bf16 => rmsnormTyped%[tl.float64, tl.float32, tl.bfloat16]
  | .fp64, .fp32, .f8e4 => rmsnormTyped%[tl.float64, tl.float32, tl.float8e4nv]
  | .fp64, .fp32, .f8e5 => rmsnormTyped%[tl.float64, tl.float32, tl.float8e5]
  | .fp64, .fp64, .fp32 => rmsnormTyped%[tl.float64, tl.float64, tl.float32]
  | .fp64, .fp64, .fp64 => rmsnormTyped%[tl.float64, tl.float64, tl.float64]
  | .fp64, .fp64, .fp16 => rmsnormTyped%[tl.float64, tl.float64, tl.float16]
  | .fp64, .fp64, .bf16 => rmsnormTyped%[tl.float64, tl.float64, tl.bfloat16]
  | .fp64, .fp64, .f8e4 => rmsnormTyped%[tl.float64, tl.float64, tl.float8e4nv]
  | .fp64, .fp64, .f8e5 => rmsnormTyped%[tl.float64, tl.float64, tl.float8e5]
  | .fp64, .fp16, .fp32 => rmsnormTyped%[tl.float64, tl.float16, tl.float32]
  | .fp64, .fp16, .fp64 => rmsnormTyped%[tl.float64, tl.float16, tl.float64]
  | .fp64, .fp16, .fp16 => rmsnormTyped%[tl.float64, tl.float16, tl.float16]
  | .fp64, .fp16, .bf16 => rmsnormTyped%[tl.float64, tl.float16, tl.bfloat16]
  | .fp64, .fp16, .f8e4 => rmsnormTyped%[tl.float64, tl.float16, tl.float8e4nv]
  | .fp64, .fp16, .f8e5 => rmsnormTyped%[tl.float64, tl.float16, tl.float8e5]
  | .fp64, .bf16, .fp32 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float32]
  | .fp64, .bf16, .fp64 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float64]
  | .fp64, .bf16, .fp16 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float16]
  | .fp64, .bf16, .bf16 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.bfloat16]
  | .fp64, .bf16, .f8e4 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float8e4nv]
  | .fp64, .bf16, .f8e5 => rmsnormTyped%[tl.float64, tl.bfloat16, tl.float8e5]
  | .fp64, .f8e4, .fp32 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float32]
  | .fp64, .f8e4, .fp64 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float64]
  | .fp64, .f8e4, .fp16 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float16]
  | .fp64, .f8e4, .bf16 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.bfloat16]
  | .fp64, .f8e4, .f8e4 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float8e4nv]
  | .fp64, .f8e4, .f8e5 => rmsnormTyped%[tl.float64, tl.float8e4nv, tl.float8e5]
  | .fp64, .f8e5, .fp32 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float32]
  | .fp64, .f8e5, .fp64 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float64]
  | .fp64, .f8e5, .fp16 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float16]
  | .fp64, .f8e5, .bf16 => rmsnormTyped%[tl.float64, tl.float8e5, tl.bfloat16]
  | .fp64, .f8e5, .f8e4 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float8e4nv]
  | .fp64, .f8e5, .f8e5 => rmsnormTyped%[tl.float64, tl.float8e5, tl.float8e5]
  | .fp16, .fp32, .fp32 => rmsnormTyped%[tl.float16, tl.float32, tl.float32]
  | .fp16, .fp32, .fp64 => rmsnormTyped%[tl.float16, tl.float32, tl.float64]
  | .fp16, .fp32, .fp16 => rmsnormTyped%[tl.float16, tl.float32, tl.float16]
  | .fp16, .fp32, .bf16 => rmsnormTyped%[tl.float16, tl.float32, tl.bfloat16]
  | .fp16, .fp32, .f8e4 => rmsnormTyped%[tl.float16, tl.float32, tl.float8e4nv]
  | .fp16, .fp32, .f8e5 => rmsnormTyped%[tl.float16, tl.float32, tl.float8e5]
  | .fp16, .fp64, .fp32 => rmsnormTyped%[tl.float16, tl.float64, tl.float32]
  | .fp16, .fp64, .fp64 => rmsnormTyped%[tl.float16, tl.float64, tl.float64]
  | .fp16, .fp64, .fp16 => rmsnormTyped%[tl.float16, tl.float64, tl.float16]
  | .fp16, .fp64, .bf16 => rmsnormTyped%[tl.float16, tl.float64, tl.bfloat16]
  | .fp16, .fp64, .f8e4 => rmsnormTyped%[tl.float16, tl.float64, tl.float8e4nv]
  | .fp16, .fp64, .f8e5 => rmsnormTyped%[tl.float16, tl.float64, tl.float8e5]
  | .fp16, .fp16, .fp32 => rmsnormTyped%[tl.float16, tl.float16, tl.float32]
  | .fp16, .fp16, .fp64 => rmsnormTyped%[tl.float16, tl.float16, tl.float64]
  | .fp16, .fp16, .fp16 => rmsnormTyped%[tl.float16, tl.float16, tl.float16]
  | .fp16, .fp16, .bf16 => rmsnormTyped%[tl.float16, tl.float16, tl.bfloat16]
  | .fp16, .fp16, .f8e4 => rmsnormTyped%[tl.float16, tl.float16, tl.float8e4nv]
  | .fp16, .fp16, .f8e5 => rmsnormTyped%[tl.float16, tl.float16, tl.float8e5]
  | .fp16, .bf16, .fp32 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float32]
  | .fp16, .bf16, .fp64 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float64]
  | .fp16, .bf16, .fp16 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float16]
  | .fp16, .bf16, .bf16 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.bfloat16]
  | .fp16, .bf16, .f8e4 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float8e4nv]
  | .fp16, .bf16, .f8e5 => rmsnormTyped%[tl.float16, tl.bfloat16, tl.float8e5]
  | .fp16, .f8e4, .fp32 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float32]
  | .fp16, .f8e4, .fp64 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float64]
  | .fp16, .f8e4, .fp16 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float16]
  | .fp16, .f8e4, .bf16 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.bfloat16]
  | .fp16, .f8e4, .f8e4 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float8e4nv]
  | .fp16, .f8e4, .f8e5 => rmsnormTyped%[tl.float16, tl.float8e4nv, tl.float8e5]
  | .fp16, .f8e5, .fp32 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float32]
  | .fp16, .f8e5, .fp64 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float64]
  | .fp16, .f8e5, .fp16 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float16]
  | .fp16, .f8e5, .bf16 => rmsnormTyped%[tl.float16, tl.float8e5, tl.bfloat16]
  | .fp16, .f8e5, .f8e4 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float8e4nv]
  | .fp16, .f8e5, .f8e5 => rmsnormTyped%[tl.float16, tl.float8e5, tl.float8e5]
  | .bf16, .fp32, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float32]
  | .bf16, .fp32, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float64]
  | .bf16, .fp32, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float16]
  | .bf16, .fp32, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.bfloat16]
  | .bf16, .fp32, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float8e4nv]
  | .bf16, .fp32, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float32, tl.float8e5]
  | .bf16, .fp64, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float32]
  | .bf16, .fp64, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float64]
  | .bf16, .fp64, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float16]
  | .bf16, .fp64, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.bfloat16]
  | .bf16, .fp64, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float8e4nv]
  | .bf16, .fp64, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float64, tl.float8e5]
  | .bf16, .fp16, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float32]
  | .bf16, .fp16, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float64]
  | .bf16, .fp16, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float16]
  | .bf16, .fp16, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.bfloat16]
  | .bf16, .fp16, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float8e4nv]
  | .bf16, .fp16, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float16, tl.float8e5]
  | .bf16, .bf16, .fp32 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float32]
  | .bf16, .bf16, .fp64 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float64]
  | .bf16, .bf16, .fp16 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float16]
  | .bf16, .bf16, .bf16 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.bfloat16]
  | .bf16, .bf16, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float8e4nv]
  | .bf16, .bf16, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.bfloat16, tl.float8e5]
  | .bf16, .f8e4, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float32]
  | .bf16, .f8e4, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float64]
  | .bf16, .f8e4, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float16]
  | .bf16, .f8e4, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.bfloat16]
  | .bf16, .f8e4, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv]
  | .bf16, .f8e4, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e5]
  | .bf16, .f8e5, .fp32 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float32]
  | .bf16, .f8e5, .fp64 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float64]
  | .bf16, .f8e5, .fp16 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float16]
  | .bf16, .f8e5, .bf16 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.bfloat16]
  | .bf16, .f8e5, .f8e4 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float8e4nv]
  | .bf16, .f8e5, .f8e5 => rmsnormTyped%[tl.bfloat16, tl.float8e5, tl.float8e5]
  | .f8e4, .fp32, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float32]
  | .f8e4, .fp32, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float64]
  | .f8e4, .fp32, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float16]
  | .f8e4, .fp32, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.bfloat16]
  | .f8e4, .fp32, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float8e4nv]
  | .f8e4, .fp32, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float32, tl.float8e5]
  | .f8e4, .fp64, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float32]
  | .f8e4, .fp64, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float64]
  | .f8e4, .fp64, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float16]
  | .f8e4, .fp64, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.bfloat16]
  | .f8e4, .fp64, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float8e4nv]
  | .f8e4, .fp64, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float64, tl.float8e5]
  | .f8e4, .fp16, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float32]
  | .f8e4, .fp16, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float64]
  | .f8e4, .fp16, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float16]
  | .f8e4, .fp16, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.bfloat16]
  | .f8e4, .fp16, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float8e4nv]
  | .f8e4, .fp16, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float16, tl.float8e5]
  | .f8e4, .bf16, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float32]
  | .f8e4, .bf16, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float64]
  | .f8e4, .bf16, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float16]
  | .f8e4, .bf16, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.bfloat16]
  | .f8e4, .bf16, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv]
  | .f8e4, .bf16, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e5]
  | .f8e4, .f8e4, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float32]
  | .f8e4, .f8e4, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float64]
  | .f8e4, .f8e4, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float16]
  | .f8e4, .f8e4, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16]
  | .f8e4, .f8e4, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv]
  | .f8e4, .f8e4, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e5]
  | .f8e4, .f8e5, .fp32 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float32]
  | .f8e4, .f8e5, .fp64 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float64]
  | .f8e4, .f8e5, .fp16 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float16]
  | .f8e4, .f8e5, .bf16 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.bfloat16]
  | .f8e4, .f8e5, .f8e4 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float8e4nv]
  | .f8e4, .f8e5, .f8e5 => rmsnormTyped%[tl.float8e4nv, tl.float8e5, tl.float8e5]
  | .f8e5, .fp32, .fp32 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float32]
  | .f8e5, .fp32, .fp64 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float64]
  | .f8e5, .fp32, .fp16 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float16]
  | .f8e5, .fp32, .bf16 => rmsnormTyped%[tl.float8e5, tl.float32, tl.bfloat16]
  | .f8e5, .fp32, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float8e4nv]
  | .f8e5, .fp32, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float32, tl.float8e5]
  | .f8e5, .fp64, .fp32 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float32]
  | .f8e5, .fp64, .fp64 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float64]
  | .f8e5, .fp64, .fp16 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float16]
  | .f8e5, .fp64, .bf16 => rmsnormTyped%[tl.float8e5, tl.float64, tl.bfloat16]
  | .f8e5, .fp64, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float8e4nv]
  | .f8e5, .fp64, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float64, tl.float8e5]
  | .f8e5, .fp16, .fp32 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float32]
  | .f8e5, .fp16, .fp64 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float64]
  | .f8e5, .fp16, .fp16 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float16]
  | .f8e5, .fp16, .bf16 => rmsnormTyped%[tl.float8e5, tl.float16, tl.bfloat16]
  | .f8e5, .fp16, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float8e4nv]
  | .f8e5, .fp16, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float16, tl.float8e5]
  | .f8e5, .bf16, .fp32 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float32]
  | .f8e5, .bf16, .fp64 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float64]
  | .f8e5, .bf16, .fp16 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float16]
  | .f8e5, .bf16, .bf16 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.bfloat16]
  | .f8e5, .bf16, .f8e4 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float8e4nv]
  | .f8e5, .bf16, .f8e5 => rmsnormTyped%[tl.float8e5, tl.bfloat16, tl.float8e5]
  | .f8e5, .f8e4, .fp32 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float32]
  | .f8e5, .f8e4, .fp64 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float64]
  | .f8e5, .f8e4, .fp16 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float16]
  | .f8e5, .f8e4, .bf16 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.bfloat16]
  | .f8e5, .f8e4, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float8e4nv]
  | .f8e5, .f8e4, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float8e4nv, tl.float8e5]
  | .f8e5, .f8e5, .fp32 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float32]
  | .f8e5, .f8e5, .fp64 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float64]
  | .f8e5, .f8e5, .fp16 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float16]
  | .f8e5, .f8e5, .bf16 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.bfloat16]
  | .f8e5, .f8e5, .f8e4 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float8e4nv]
  | .f8e5, .f8e5, .f8e5 => rmsnormTyped%[tl.float8e5, tl.float8e5, tl.float8e5]
```
</details>
