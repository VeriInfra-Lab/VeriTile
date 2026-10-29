# Spec sheet — `bench/tritonbench_g/triton_softmax/Correct.lean`

**Python source:** `bench/tritonbench_g/triton_softmax/triton_softmax.py`

## Public theorem: `softmax_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete source variants implement the independently defined masked
stable-softmax formula. Symbolic positive block size, both independent strides,
all partial rows and both input/output floating storage choices are retained.
This is real-erasure correctness; the FP theorem retains every precision. -/
```
</details>

**Statement:**
```lean
specification softmax_typed_real_correct (input output : Storage) (v : Bool)
    (out inp : RegionName) (si so n B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedIO input output v out inp si so n B ⊨
      fun xs i => softmaxSpec n B xs i)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `typedIO`, `typedKernel`

<details><summary><code>typedIO</code></summary>

```
/-- Complete typed source projected to real arithmetic. Every active input
and output address is specified; the mathematical contract includes success
and the full frame outside active writes. -/
```
```lean
def typedIO (input output : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) : MaskedKernelIO₁ :=
  { softmaxIO out inp si so n B with
    kernel := (typedKernel input output v out inp si so n B).eraseDType
    projection := typed_supported input output v out inp si so n B }
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (input output : Storage) (commuted : Bool) : Source :=
  match input, output, commuted with
  | .fp32, .fp32, Bool.false => softmaxSource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => softmaxSource%[tl.float32, tl.float32, indexCommuted]
  | .fp32, .fp64, Bool.false => softmaxSource%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => softmaxSource%[tl.float32, tl.float64, indexCommuted]
  | .fp32, .fp16, Bool.false => softmaxSource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => softmaxSource%[tl.float32, tl.float16, indexCommuted]
  | .fp32, .bf16, Bool.false => softmaxSource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => softmaxSource%[tl.float32, tl.bfloat16, indexCommuted]
  | .fp32, .f8e4, Bool.false => softmaxSource%[tl.float32, tl.float8e4nv, original]
  | .fp32, .f8e4, Bool.true => softmaxSource%[tl.float32, tl.float8e4nv, indexCommuted]
  | .fp32, .f8e5, Bool.false => softmaxSource%[tl.float32, tl.float8e5, original]
  | .fp32, .f8e5, Bool.true => softmaxSource%[tl.float32, tl.float8e5, indexCommuted]
  | .fp64, .fp32, Bool.false => softmaxSource%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => softmaxSource%[tl.float64, tl.float32, indexCommuted]
  | .fp64, .fp64, Bool.false => softmaxSource%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => softmaxSource%[tl.float64, tl.float64, indexCommuted]
  | .fp64, .fp16, Bool.false => softmaxSource%[tl.float64, tl.float16, original]
  | .fp64, .fp16, Bool.true => softmaxSource%[tl.float64, tl.float16, indexCommuted]
  | .fp64, .bf16, Bool.false => softmaxSource%[tl.float64, tl.bfloat16, original]
  | .fp64, .bf16, Bool.true => softmaxSource%[tl.float64, tl.bfloat16, indexCommuted]
  | .fp64, .f8e4, Bool.false => softmaxSource%[tl.float64, tl.float8e4nv, original]
  | .fp64, .f8e4, Bool.true => softmaxSource%[tl.float64, tl.float8e4nv, indexCommuted]
  | .fp64, .f8e5, Bool.false => softmaxSource%[tl.float64, tl.float8e5, original]
  | .fp64, .f8e5, Bool.true => softmaxSource%[tl.float64, tl.float8e5, indexCommuted]
  | .fp16, .fp32, Bool.false => softmaxSource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => softmaxSource%[tl.float16, tl.float32, indexCommuted]
  | .fp16, .fp64, Bool.false => softmaxSource%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => softmaxSource%[tl.float16, tl.float64, indexCommuted]
  | .fp16, .fp16, Bool.false => softmaxSource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => softmaxSource%[tl.float16, tl.float16, indexCommuted]
  | .fp16, .bf16, Bool.false => softmaxSource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => softmaxSource%[tl.float16, tl.bfloat16, indexCommuted]
  | .fp16, .f8e4, Bool.false => softmaxSource%[tl.float16, tl.float8e4nv, original]
  | .fp16, .f8e4, Bool.true => softmaxSource%[tl.float16, tl.float8e4nv, indexCommuted]
  | .fp16, .f8e5, Bool.false => softmaxSource%[tl.float16, tl.float8e5, original]
  | .fp16, .f8e5, Bool.true => softmaxSource%[tl.float16, tl.float8e5, indexCommuted]
  | .bf16, .fp32, Bool.false => softmaxSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => softmaxSource%[tl.bfloat16, tl.float32, indexCommuted]
  | .bf16, .fp64, Bool.false => softmaxSource%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => softmaxSource%[tl.bfloat16, tl.float64, indexCommuted]
  | .bf16, .fp16, Bool.false => softmaxSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => softmaxSource%[tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, .bf16, Bool.false => softmaxSource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => softmaxSource%[tl.bfloat16, tl.bfloat16, indexCommuted]
  | .bf16, .f8e4, Bool.false => softmaxSource%[tl.bfloat16, tl.float8e4nv, original]
  | .bf16, .f8e4, Bool.true => softmaxSource%[tl.bfloat16, tl.float8e4nv, indexCommuted]
  | .bf16, .f8e5, Bool.false => softmaxSource%[tl.bfloat16, tl.float8e5, original]
  | .bf16, .f8e5, Bool.true => softmaxSource%[tl.bfloat16, tl.float8e5, indexCommuted]
  | .f8e4, .fp32, Bool.false => softmaxSource%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => softmaxSource%[tl.float8e4nv, tl.float32, indexCommuted]
  | .f8e4, .fp64, Bool.false => softmaxSource%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => softmaxSource%[tl.float8e4nv, tl.float64, indexCommuted]
  | .f8e4, .fp16, Bool.false => softmaxSource%[tl.float8e4nv, tl.float16, original]
  | .f8e4, .fp16, Bool.true => softmaxSource%[tl.float8e4nv, tl.float16, indexCommuted]
  | .f8e4, .bf16, Bool.false => softmaxSource%[tl.float8e4nv, tl.bfloat16, original]
  | .f8e4, .bf16, Bool.true => softmaxSource%[tl.float8e4nv, tl.bfloat16, indexCommuted]
  | .f8e4, .f8e4, Bool.false => softmaxSource%[tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, .f8e4, Bool.true => softmaxSource%[tl.float8e4nv, tl.float8e4nv, indexCommuted]
  | .f8e4, .f8e5, Bool.false => softmaxSource%[tl.float8e4nv, tl.float8e5, original]
  | .f8e4, .f8e5, Bool.true => softmaxSource%[tl.float8e4nv, tl.float8e5, indexCommuted]
  | .f8e5, .fp32, Bool.false => softmaxSource%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => softmaxSource%[tl.float8e5, tl.float32, indexCommuted]
  | .f8e5, .fp64, Bool.false => softmaxSource%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => softmaxSource%[tl.float8e5, tl.float64, indexCommuted]
  | .f8e5, .fp16, Bool.false => softmaxSource%[tl.float8e5, tl.float16, original]
  | .f8e5, .fp16, Bool.true => softmaxSource%[tl.float8e5, tl.float16, indexCommuted]
  | .f8e5, .bf16, Bool.false => softmaxSource%[tl.float8e5, tl.bfloat16, original]
  | .f8e5, .bf16, Bool.true => softmaxSource%[tl.float8e5, tl.bfloat16, indexCommuted]
  | .f8e5, .f8e4, Bool.false => softmaxSource%[tl.float8e5, tl.float8e4nv, original]
  | .f8e5, .f8e4, Bool.true => softmaxSource%[tl.float8e5, tl.float8e4nv, indexCommuted]
  | .f8e5, .f8e5, Bool.false => softmaxSource%[tl.float8e5, tl.float8e5, original]
  | .f8e5, .f8e5, Bool.true => softmaxSource%[tl.float8e5, tl.float8e5, indexCommuted]
```
</details>
