# Spec sheet — `bench/tritonbench_g/rmsnorm_fused/Correct.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_fused/rmsnorm_fused.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Every floating storage profile and both complete source variants implement the
independent full logical-row RMS formula for arbitrary row lengths, with successful
flat execution and a full memory frame. Output/input separation protects the initial
row snapshot across multiple source iterations. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXY : X ≠ Y) (hWY : W ≠ Y) :
    General.FullExactIO
      (typedKernel input weight output v X Y W stride N B eps).eraseDType
      X Y W stride N eps
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hXY : X ≠ Y`
- `hWY : W ≠ Y`

**Closed-form spec defs (transitive):** `typedKernel`, `originalTyped`

<details><summary><code>typedKernel</code></summary>

```
/-- Only the natural multiplication in the output row-pointer update commutes. -/
```
```lean
def typedKernel (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : ComputeKernel :=
  let source := originalTyped input weight output X Y W stride N B eps
  if v then ComputeKernel.mk source.inputs source.outputs
    (source.surfaceBody.take 1 ++
      [.assign .ptr [] "Y" (.alg (.ptrAdd .nil (.ptrBase Y)
        (.mul .nat .nil (.constNat stride) (.ref .nat [] "row"))))] ++ source.surfaceBody.drop 2)
  else source
```
</details>

<details><summary><code>originalTyped</code></summary>

```
/-- Complete source body, retaining both loops, the extra variance mask and both
fp32 input conversions. The final cast represents implicit destination conversion. -/
```
```lean
def originalTyped (input weight output : Storage) : Source :=
  match input, weight, output with
  | .fp32, .fp32, .fp32 => rmsFusedTyped%[tl.float32, tl.float32, tl.float32]
  | .fp32, .fp32, .fp64 => rmsFusedTyped%[tl.float32, tl.float32, tl.float64]
  | .fp32, .fp32, .fp16 => rmsFusedTyped%[tl.float32, tl.float32, tl.float16]
  | .fp32, .fp32, .bf16 => rmsFusedTyped%[tl.float32, tl.float32, tl.bfloat16]
  | .fp32, .fp32, .f8e4 => rmsFusedTyped%[tl.float32, tl.float32, tl.float8e4nv]
  | .fp32, .fp32, .f8e5 => rmsFusedTyped%[tl.float32, tl.float32, tl.float8e5]
  | .fp32, .fp64, .fp32 => rmsFusedTyped%[tl.float32, tl.float64, tl.float32]
  | .fp32, .fp64, .fp64 => rmsFusedTyped%[tl.float32, tl.float64, tl.float64]
  | .fp32, .fp64, .fp16 => rmsFusedTyped%[tl.float32, tl.float64, tl.float16]
  | .fp32, .fp64, .bf16 => rmsFusedTyped%[tl.float32, tl.float64, tl.bfloat16]
  | .fp32, .fp64, .f8e4 => rmsFusedTyped%[tl.float32, tl.float64, tl.float8e4nv]
  | .fp32, .fp64, .f8e5 => rmsFusedTyped%[tl.float32, tl.float64, tl.float8e5]
  | .fp32, .fp16, .fp32 => rmsFusedTyped%[tl.float32, tl.float16, tl.float32]
  | .fp32, .fp16, .fp64 => rmsFusedTyped%[tl.float32, tl.float16, tl.float64]
  | .fp32, .fp16, .fp16 => rmsFusedTyped%[tl.float32, tl.float16, tl.float16]
  | .fp32, .fp16, .bf16 => rmsFusedTyped%[tl.float32, tl.float16, tl.bfloat16]
  | .fp32, .fp16, .f8e4 => rmsFusedTyped%[tl.float32, tl.float16, tl.float8e4nv]
  | .fp32, .fp16, .f8e5 => rmsFusedTyped%[tl.float32, tl.float16, tl.float8e5]
  | .fp32, .bf16, .fp32 => rmsFusedTyped%[tl.float32, tl.bfloat16, tl.float32]
  | .fp32, .bf16, .fp64 => rmsFusedTyped%[tl.float32, tl.bfloat16, tl.float64]
  | .fp32, .bf16, .fp16 => rmsFusedTyped%[tl.float32, tl.bfloat16, tl.float16]
  | .fp32, .bf16, .bf16 => rmsFusedTyped%[tl.float32, tl.bfloat16, tl.bfloat16]
  | .fp32, .bf16, .f8e4 => rmsFusedTyped%[tl.float32, tl.bfloat16, tl.float8e4nv]
  | .fp32, .bf16, .f8e5 => rmsFusedTyped%[tl.float32, tl.bfloat16, tl.float8e5]
  | .fp32, .f8e4, .fp32 => rmsFusedTyped%[tl.float32, tl.float8e4nv, tl.float32]
  | .fp32, .f8e4, .fp64 => rmsFusedTyped%[tl.float32, tl.float8e4nv, tl.float64]
  | .fp32, .f8e4, .fp16 => rmsFusedTyped%[tl.float32, tl.float8e4nv, tl.float16]
  | .fp32, .f8e4, .bf16 => rmsFusedTyped%[tl.float32, tl.float8e4nv, tl.bfloat16]
  | .fp32, .f8e4, .f8e4 => rmsFusedTyped%[tl.float32, tl.float8e4nv, tl.float8e4nv]
  | .fp32, .f8e4, .f8e5 => rmsFusedTyped%[tl.float32, tl.float8e4nv, tl.float8e5]
  | .fp32, .f8e5, .fp32 => rmsFusedTyped%[tl.float32, tl.float8e5, tl.float32]
  | .fp32, .f8e5, .fp64 => rmsFusedTyped%[tl.float32, tl.float8e5, tl.float64]
  | .fp32, .f8e5, .fp16 => rmsFusedTyped%[tl.float32, tl.float8e5, tl.float16]
  | .fp32, .f8e5, .bf16 => rmsFusedTyped%[tl.float32, tl.float8e5, tl.bfloat16]
  | .fp32, .f8e5, .f8e4 => rmsFusedTyped%[tl.float32, tl.float8e5, tl.float8e4nv]
  | .fp32, .f8e5, .f8e5 => rmsFusedTyped%[tl.float32, tl.float8e5, tl.float8e5]
  | .fp64, .fp32, .fp32 => rmsFusedTyped%[tl.float64, tl.float32, tl.float32]
  | .fp64, .fp32, .fp64 => rmsFusedTyped%[tl.float64, tl.float32, tl.float64]
  | .fp64, .fp32, .fp16 => rmsFusedTyped%[tl.float64, tl.float32, tl.float16]
  | .fp64, .fp32, .bf16 => rmsFusedTyped%[tl.float64, tl.float32, tl.bfloat16]
  | .fp64, .fp32, .f8e4 => rmsFusedTyped%[tl.float64, tl.float32, tl.float8e4nv]
  | .fp64, .fp32, .f8e5 => rmsFusedTyped%[tl.float64, tl.float32, tl.float8e5]
  | .fp64, .fp64, .fp32 => rmsFusedTyped%[tl.float64, tl.float64, tl.float32]
  | .fp64, .fp64, .fp64 => rmsFusedTyped%[tl.float64, tl.float64, tl.float64]
  | .fp64, .fp64, .fp16 => rmsFusedTyped%[tl.float64, tl.float64, tl.float16]
  | .fp64, .fp64, .bf16 => rmsFusedTyped%[tl.float64, tl.float64, tl.bfloat16]
  | .fp64, .fp64, .f8e4 => rmsFusedTyped%[tl.float64, tl.float64, tl.float8e4nv]
  | .fp64, .fp64, .f8e5 => rmsFusedTyped%[tl.float64, tl.float64, tl.float8e5]
  | .fp64, .fp16, .fp32 => rmsFusedTyped%[tl.float64, tl.float16, tl.float32]
  | .fp64, .fp16, .fp64 => rmsFusedTyped%[tl.float64, tl.float16, tl.float64]
  | .fp64, .fp16, .fp16 => rmsFusedTyped%[tl.float64, tl.float16, tl.float16]
  | .fp64, .fp16, .bf16 => rmsFusedTyped%[tl.float64, tl.float16, tl.bfloat16]
  | .fp64, .fp16, .f8e4 => rmsFusedTyped%[tl.float64, tl.float16, tl.float8e4nv]
  | .fp64, .fp16, .f8e5 => rmsFusedTyped%[tl.float64, tl.float16, tl.float8e5]
  | .fp64, .bf16, .fp32 => rmsFusedTyped%[tl.float64, tl.bfloat16, tl.float32]
  | .fp64, .bf16, .fp64 => rmsFusedTyped%[tl.float64, tl.bfloat16, tl.float64]
  | .fp64, .bf16, .fp16 => rmsFusedTyped%[tl.float64, tl.bfloat16, tl.float16]
  | .fp64, .bf16, .bf16 => rmsFusedTyped%[tl.float64, tl.bfloat16, tl.bfloat16]
  | .fp64, .bf16, .f8e4 => rmsFusedTyped%[tl.float64, tl.bfloat16, tl.float8e4nv]
  | .fp64, .bf16, .f8e5 => rmsFusedTyped%[tl.float64, tl.bfloat16, tl.float8e5]
  | .fp64, .f8e4, .fp32 => rmsFusedTyped%[tl.float64, tl.float8e4nv, tl.float32]
  | .fp64, .f8e4, .fp64 => rmsFusedTyped%[tl.float64, tl.float8e4nv, tl.float64]
  | .fp64, .f8e4, .fp16 => rmsFusedTyped%[tl.float64, tl.float8e4nv, tl.float16]
  | .fp64, .f8e4, .bf16 => rmsFusedTyped%[tl.float64, tl.float8e4nv, tl.bfloat16]
  | .fp64, .f8e4, .f8e4 => rmsFusedTyped%[tl.float64, tl.float8e4nv, tl.float8e4nv]
  | .fp64, .f8e4, .f8e5 => rmsFusedTyped%[tl.float64, tl.float8e4nv, tl.float8e5]
  | .fp64, .f8e5, .fp32 => rmsFusedTyped%[tl.float64, tl.float8e5, tl.float32]
  | .fp64, .f8e5, .fp64 => rmsFusedTyped%[tl.float64, tl.float8e5, tl.float64]
  | .fp64, .f8e5, .fp16 => rmsFusedTyped%[tl.float64, tl.float8e5, tl.float16]
  | .fp64, .f8e5, .bf16 => rmsFusedTyped%[tl.float64, tl.float8e5, tl.bfloat16]
  | .fp64, .f8e5, .f8e4 => rmsFusedTyped%[tl.float64, tl.float8e5, tl.float8e4nv]
  | .fp64, .f8e5, .f8e5 => rmsFusedTyped%[tl.float64, tl.float8e5, tl.float8e5]
  | .fp16, .fp32, .fp32 => rmsFusedTyped%[tl.float16, tl.float32, tl.float32]
  | .fp16, .fp32, .fp64 => rmsFusedTyped%[tl.float16, tl.float32, tl.float64]
  | .fp16, .fp32, .fp16 => rmsFusedTyped%[tl.float16, tl.float32, tl.float16]
  | .fp16, .fp32, .bf16 => rmsFusedTyped%[tl.float16, tl.float32, tl.bfloat16]
  | .fp16, .fp32, .f8e4 => rmsFusedTyped%[tl.float16, tl.float32, tl.float8e4nv]
  | .fp16, .fp32, .f8e5 => rmsFusedTyped%[tl.float16, tl.float32, tl.float8e5]
  | .fp16, .fp64, .fp32 => rmsFusedTyped%[tl.float16, tl.float64, tl.float32]
  | .fp16, .fp64, .fp64 => rmsFusedTyped%[tl.float16, tl.float64, tl.float64]
  | .fp16, .fp64, .fp16 => rmsFusedTyped%[tl.float16, tl.float64, tl.float16]
  | .fp16, .fp64, .bf16 => rmsFusedTyped%[tl.float16, tl.float64, tl.bfloat16]
  | .fp16, .fp64, .f8e4 => rmsFusedTyped%[tl.float16, tl.float64, tl.float8e4nv]
  | .fp16, .fp64, .f8e5 => rmsFusedTyped%[tl.float16, tl.float64, tl.float8e5]
  | .fp16, .fp16, .fp32 => rmsFusedTyped%[tl.float16, tl.float16, tl.float32]
  | .fp16, .fp16, .fp64 => rmsFusedTyped%[tl.float16, tl.float16, tl.float64]
  | .fp16, .fp16, .fp16 => rmsFusedTyped%[tl.float16, tl.float16, tl.float16]
  | .fp16, .fp16, .bf16 => rmsFusedTyped%[tl.float16, tl.float16, tl.bfloat16]
  | .fp16, .fp16, .f8e4 => rmsFusedTyped%[tl.float16, tl.float16, tl.float8e4nv]
  | .fp16, .fp16, .f8e5 => rmsFusedTyped%[tl.float16, tl.float16, tl.float8e5]
  | .fp16, .bf16, .fp32 => rmsFusedTyped%[tl.float16, tl.bfloat16, tl.float32]
  | .fp16, .bf16, .fp64 => rmsFusedTyped%[tl.float16, tl.bfloat16, tl.float64]
  | .fp16, .bf16, .fp16 => rmsFusedTyped%[tl.float16, tl.bfloat16, tl.float16]
  | .fp16, .bf16, .bf16 => rmsFusedTyped%[tl.float16, tl.bfloat16, tl.bfloat16]
  | .fp16, .bf16, .f8e4 => rmsFusedTyped%[tl.float16, tl.bfloat16, tl.float8e4nv]
  | .fp16, .bf16, .f8e5 => rmsFusedTyped%[tl.float16, tl.bfloat16, tl.float8e5]
  | .fp16, .f8e4, .fp32 => rmsFusedTyped%[tl.float16, tl.float8e4nv, tl.float32]
  | .fp16, .f8e4, .fp64 => rmsFusedTyped%[tl.float16, tl.float8e4nv, tl.float64]
  | .fp16, .f8e4, .fp16 => rmsFusedTyped%[tl.float16, tl.float8e4nv, tl.float16]
  | .fp16, .f8e4, .bf16 => rmsFusedTyped%[tl.float16, tl.float8e4nv, tl.bfloat16]
  | .fp16, .f8e4, .f8e4 => rmsFusedTyped%[tl.float16, tl.float8e4nv, tl.float8e4nv]
  | .fp16, .f8e4, .f8e5 => rmsFusedTyped%[tl.float16, tl.float8e4nv, tl.float8e5]
  | .fp16, .f8e5, .fp32 => rmsFusedTyped%[tl.float16, tl.float8e5, tl.float32]
  | .fp16, .f8e5, .fp64 => rmsFusedTyped%[tl.float16, tl.float8e5, tl.float64]
  | .fp16, .f8e5, .fp16 => rmsFusedTyped%[tl.float16, tl.float8e5, tl.float16]
  | .fp16, .f8e5, .bf16 => rmsFusedTyped%[tl.float16, tl.float8e5, tl.bfloat16]
  | .fp16, .f8e5, .f8e4 => rmsFusedTyped%[tl.float16, tl.float8e5, tl.float8e4nv]
  | .fp16, .f8e5, .f8e5 => rmsFusedTyped%[tl.float16, tl.float8e5, tl.float8e5]
  | .bf16, .fp32, .fp32 => rmsFusedTyped%[tl.bfloat16, tl.float32, tl.float32]
  | .bf16, .fp32, .fp64 => rmsFusedTyped%[tl.bfloat16, tl.float32, tl.float64]
  | .bf16, .fp32, .fp16 => rmsFusedTyped%[tl.bfloat16, tl.float32, tl.float16]
  | .bf16, .fp32, .bf16 => rmsFusedTyped%[tl.bfloat16, tl.float32, tl.bfloat16]
  | .bf16, .fp32, .f8e4 => rmsFusedTyped%[tl.bfloat16, tl.float32, tl.float8e4nv]
  | .bf16, .fp32, .f8e5 => rmsFusedTyped%[tl.bfloat16, tl.float32, tl.float8e5]
  | .bf16, .fp64, .fp32 => rmsFusedTyped%[tl.bfloat16, tl.float64, tl.float32]
  | .bf16, .fp64, .fp64 => rmsFusedTyped%[tl.bfloat16, tl.float64, tl.float64]
  | .bf16, .fp64, .fp16 => rmsFusedTyped%[tl.bfloat16, tl.float64, tl.float16]
  | .bf16, .fp64, .bf16 => rmsFusedTyped%[tl.bfloat16, tl.float64, tl.bfloat16]
  | .bf16, .fp64, .f8e4 => rmsFusedTyped%[tl.bfloat16, tl.float64, tl.float8e4nv]
  | .bf16, .fp64, .f8e5 => rmsFusedTyped%[tl.bfloat16, tl.float64, tl.float8e5]
  | .bf16, .fp16, .fp32 => rmsFusedTyped%[tl.bfloat16, tl.float16, tl.float32]
  | .bf16, .fp16, .fp64 => rmsFusedTyped%[tl.bfloat16, tl.float16, tl.float64]
  | .bf16, .fp16, .fp16 => rmsFusedTyped%[tl.bfloat16, tl.float16, tl.float16]
  | .bf16, .fp16, .bf16 => rmsFusedTyped%[tl.bfloat16, tl.float16, tl.bfloat16]
  | .bf16, .fp16, .f8e4 => rmsFusedTyped%[tl.bfloat16, tl.float16, tl.float8e4nv]
  | .bf16, .fp16, .f8e5 => rmsFusedTyped%[tl.bfloat16, tl.float16, tl.float8e5]
  | .bf16, .bf16, .fp32 => rmsFusedTyped%[tl.bfloat16, tl.bfloat16, tl.float32]
  | .bf16, .bf16, .fp64 => rmsFusedTyped%[tl.bfloat16, tl.bfloat16, tl.float64]
  | .bf16, .bf16, .fp16 => rmsFusedTyped%[tl.bfloat16, tl.bfloat16, tl.float16]
  | .bf16, .bf16, .bf16 => rmsFusedTyped%[tl.bfloat16, tl.bfloat16, tl.bfloat16]
  | .bf16, .bf16, .f8e4 => rmsFusedTyped%[tl.bfloat16, tl.bfloat16, tl.float8e4nv]
  | .bf16, .bf16, .f8e5 => rmsFusedTyped%[tl.bfloat16, tl.bfloat16, tl.float8e5]
  | .bf16, .f8e4, .fp32 => rmsFusedTyped%[tl.bfloat16, tl.float8e4nv, tl.float32]
  | .bf16, .f8e4, .fp64 => rmsFusedTyped%[tl.bfloat16, tl.float8e4nv, tl.float64]
  | .bf16, .f8e4, .fp16 => rmsFusedTyped%[tl.bfloat16, tl.float8e4nv, tl.float16]
  | .bf16, .f8e4, .bf16 => rmsFusedTyped%[tl.bfloat16, tl.float8e4nv, tl.bfloat16]
  | .bf16, .f8e4, .f8e4 => rmsFusedTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv]
  | .bf16, .f8e4, .f8e5 => rmsFusedTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e5]
  | .bf16, .f8e5, .fp32 => rmsFusedTyped%[tl.bfloat16, tl.float8e5, tl.float32]
  | .bf16, .f8e5, .fp64 => rmsFusedTyped%[tl.bfloat16, tl.float8e5, tl.float64]
  | .bf16, .f8e5, .fp16 => rmsFusedTyped%[tl.bfloat16, tl.float8e5, tl.float16]
  | .bf16, .f8e5, .bf16 => rmsFusedTyped%[tl.bfloat16, tl.float8e5, tl.bfloat16]
  | .bf16, .f8e5, .f8e4 => rmsFusedTyped%[tl.bfloat16, tl.float8e5, tl.float8e4nv]
  | .bf16, .f8e5, .f8e5 => rmsFusedTyped%[tl.bfloat16, tl.float8e5, tl.float8e5]
  | .f8e4, .fp32, .fp32 => rmsFusedTyped%[tl.float8e4nv, tl.float32, tl.float32]
  | .f8e4, .fp32, .fp64 => rmsFusedTyped%[tl.float8e4nv, tl.float32, tl.float64]
  | .f8e4, .fp32, .fp16 => rmsFusedTyped%[tl.float8e4nv, tl.float32, tl.float16]
  | .f8e4, .fp32, .bf16 => rmsFusedTyped%[tl.float8e4nv, tl.float32, tl.bfloat16]
  | .f8e4, .fp32, .f8e4 => rmsFusedTyped%[tl.float8e4nv, tl.float32, tl.float8e4nv]
  | .f8e4, .fp32, .f8e5 => rmsFusedTyped%[tl.float8e4nv, tl.float32, tl.float8e5]
  | .f8e4, .fp64, .fp32 => rmsFusedTyped%[tl.float8e4nv, tl.float64, tl.float32]
  | .f8e4, .fp64, .fp64 => rmsFusedTyped%[tl.float8e4nv, tl.float64, tl.float64]
  | .f8e4, .fp64, .fp16 => rmsFusedTyped%[tl.float8e4nv, tl.float64, tl.float16]
  | .f8e4, .fp64, .bf16 => rmsFusedTyped%[tl.float8e4nv, tl.float64, tl.bfloat16]
  | .f8e4, .fp64, .f8e4 => rmsFusedTyped%[tl.float8e4nv, tl.float64, tl.float8e4nv]
  | .f8e4, .fp64, .f8e5 => rmsFusedTyped%[tl.float8e4nv, tl.float64, tl.float8e5]
  | .f8e4, .fp16, .fp32 => rmsFusedTyped%[tl.float8e4nv, tl.float16, tl.float32]
  | .f8e4, .fp16, .fp64 => rmsFusedTyped%[tl.float8e4nv, tl.float16, tl.float64]
  | .f8e4, .fp16, .fp16 => rmsFusedTyped%[tl.float8e4nv, tl.float16, tl.float16]
  | .f8e4, .fp16, .bf16 => rmsFusedTyped%[tl.float8e4nv, tl.float16, tl.bfloat16]
  | .f8e4, .fp16, .f8e4 => rmsFusedTyped%[tl.float8e4nv, tl.float16, tl.float8e4nv]
  | .f8e4, .fp16, .f8e5 => rmsFusedTyped%[tl.float8e4nv, tl.float16, tl.float8e5]
  | .f8e4, .bf16, .fp32 => rmsFusedTyped%[tl.float8e4nv, tl.bfloat16, tl.float32]
  | .f8e4, .bf16, .fp64 => rmsFusedTyped%[tl.float8e4nv, tl.bfloat16, tl.float64]
  | .f8e4, .bf16, .fp16 => rmsFusedTyped%[tl.float8e4nv, tl.bfloat16, tl.float16]
  | .f8e4, .bf16, .bf16 => rmsFusedTyped%[tl.float8e4nv, tl.bfloat16, tl.bfloat16]
  | .f8e4, .bf16, .f8e4 => rmsFusedTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv]
  | .f8e4, .bf16, .f8e5 => rmsFusedTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e5]
  | .f8e4, .f8e4, .fp32 => rmsFusedTyped%[tl.float8e4nv, tl.float8e4nv, tl.float32]
  | .f8e4, .f8e4, .fp64 => rmsFusedTyped%[tl.float8e4nv, tl.float8e4nv, tl.float64]
  | .f8e4, .f8e4, .fp16 => rmsFusedTyped%[tl.float8e4nv, tl.float8e4nv, tl.float16]
  | .f8e4, .f8e4, .bf16 => rmsFusedTyped%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16]
  | .f8e4, .f8e4, .f8e4 => rmsFusedTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv]
  | .f8e4, .f8e4, .f8e5 => rmsFusedTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e5]
  | .f8e4, .f8e5, .fp32 => rmsFusedTyped%[tl.float8e4nv, tl.float8e5, tl.float32]
  | .f8e4, .f8e5, .fp64 => rmsFusedTyped%[tl.float8e4nv, tl.float8e5, tl.float64]
  | .f8e4, .f8e5, .fp16 => rmsFusedTyped%[tl.float8e4nv, tl.float8e5, tl.float16]
  | .f8e4, .f8e5, .bf16 => rmsFusedTyped%[tl.float8e4nv, tl.float8e5, tl.bfloat16]
  | .f8e4, .f8e5, .f8e4 => rmsFusedTyped%[tl.float8e4nv, tl.float8e5, tl.float8e4nv]
  | .f8e4, .f8e5, .f8e5 => rmsFusedTyped%[tl.float8e4nv, tl.float8e5, tl.float8e5]
  | .f8e5, .fp32, .fp32 => rmsFusedTyped%[tl.float8e5, tl.float32, tl.float32]
  | .f8e5, .fp32, .fp64 => rmsFusedTyped%[tl.float8e5, tl.float32, tl.float64]
  | .f8e5, .fp32, .fp16 => rmsFusedTyped%[tl.float8e5, tl.float32, tl.float16]
  | .f8e5, .fp32, .bf16 => rmsFusedTyped%[tl.float8e5, tl.float32, tl.bfloat16]
  | .f8e5, .fp32, .f8e4 => rmsFusedTyped%[tl.float8e5, tl.float32, tl.float8e4nv]
  | .f8e5, .fp32, .f8e5 => rmsFusedTyped%[tl.float8e5, tl.float32, tl.float8e5]
  | .f8e5, .fp64, .fp32 => rmsFusedTyped%[tl.float8e5, tl.float64, tl.float32]
  | .f8e5, .fp64, .fp64 => rmsFusedTyped%[tl.float8e5, tl.float64, tl.float64]
  | .f8e5, .fp64, .fp16 => rmsFusedTyped%[tl.float8e5, tl.float64, tl.float16]
  | .f8e5, .fp64, .bf16 => rmsFusedTyped%[tl.float8e5, tl.float64, tl.bfloat16]
  | .f8e5, .fp64, .f8e4 => rmsFusedTyped%[tl.float8e5, tl.float64, tl.float8e4nv]
  | .f8e5, .fp64, .f8e5 => rmsFusedTyped%[tl.float8e5, tl.float64, tl.float8e5]
  | .f8e5, .fp16, .fp32 => rmsFusedTyped%[tl.float8e5, tl.float16, tl.float32]
  | .f8e5, .fp16, .fp64 => rmsFusedTyped%[tl.float8e5, tl.float16, tl.float64]
  | .f8e5, .fp16, .fp16 => rmsFusedTyped%[tl.float8e5, tl.float16, tl.float16]
  | .f8e5, .fp16, .bf16 => rmsFusedTyped%[tl.float8e5, tl.float16, tl.bfloat16]
  | .f8e5, .fp16, .f8e4 => rmsFusedTyped%[tl.float8e5, tl.float16, tl.float8e4nv]
  | .f8e5, .fp16, .f8e5 => rmsFusedTyped%[tl.float8e5, tl.float16, tl.float8e5]
  | .f8e5, .bf16, .fp32 => rmsFusedTyped%[tl.float8e5, tl.bfloat16, tl.float32]
  | .f8e5, .bf16, .fp64 => rmsFusedTyped%[tl.float8e5, tl.bfloat16, tl.float64]
  | .f8e5, .bf16, .fp16 => rmsFusedTyped%[tl.float8e5, tl.bfloat16, tl.float16]
  | .f8e5, .bf16, .bf16 => rmsFusedTyped%[tl.float8e5, tl.bfloat16, tl.bfloat16]
  | .f8e5, .bf16, .f8e4 => rmsFusedTyped%[tl.float8e5, tl.bfloat16, tl.float8e4nv]
  | .f8e5, .bf16, .f8e5 => rmsFusedTyped%[tl.float8e5, tl.bfloat16, tl.float8e5]
  | .f8e5, .f8e4, .fp32 => rmsFusedTyped%[tl.float8e5, tl.float8e4nv, tl.float32]
  | .f8e5, .f8e4, .fp64 => rmsFusedTyped%[tl.float8e5, tl.float8e4nv, tl.float64]
  | .f8e5, .f8e4, .fp16 => rmsFusedTyped%[tl.float8e5, tl.float8e4nv, tl.float16]
  | .f8e5, .f8e4, .bf16 => rmsFusedTyped%[tl.float8e5, tl.float8e4nv, tl.bfloat16]
  | .f8e5, .f8e4, .f8e4 => rmsFusedTyped%[tl.float8e5, tl.float8e4nv, tl.float8e4nv]
  | .f8e5, .f8e4, .f8e5 => rmsFusedTyped%[tl.float8e5, tl.float8e4nv, tl.float8e5]
  | .f8e5, .f8e5, .fp32 => rmsFusedTyped%[tl.float8e5, tl.float8e5, tl.float32]
  | .f8e5, .f8e5, .fp64 => rmsFusedTyped%[tl.float8e5, tl.float8e5, tl.float64]
  | .f8e5, .f8e5, .fp16 => rmsFusedTyped%[tl.float8e5, tl.float8e5, tl.float16]
  | .f8e5, .f8e5, .bf16 => rmsFusedTyped%[tl.float8e5, tl.float8e5, tl.bfloat16]
  | .f8e5, .f8e5, .f8e4 => rmsFusedTyped%[tl.float8e5, tl.float8e5, tl.float8e4nv]
  | .f8e5, .f8e5, .f8e5 => rmsFusedTyped%[tl.float8e5, tl.float8e5, tl.float8e5]
```
</details>
