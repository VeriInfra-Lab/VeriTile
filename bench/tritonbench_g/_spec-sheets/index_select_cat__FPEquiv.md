# Spec sheet — `bench/tritonbench_g/index_select_cat/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/index_select_cat/index_select_cat.py`

## Public theorem: `index_equiv`

<details><summary>docstring</summary>

```
/-- Complete typed gather executions agree after commuting only the natural
row-block address product. Signed index loads, masks, store conversions,
ordered duplicate writes and buffer aliasing are all preserved. -/
```
</details>

**Statement:**
```lean
specification index_equiv (input output : Storage) (index : IndexStorage)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) :
    program input output index Bool.false out idx inp n c s0 s1 BI BC ≡[assumptions]
      program input output index Bool.true out idx inp n c s0 s1 BI BC
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `source`

<details><summary><code>program</code></summary>

```lean
def program (input output : Storage) (index : IndexStorage) (v : Bool)
    (out idx inp : RegionName) (n c s0 s1 BI BC : Nat) : Program where
  kernel := source input output index v out idx inp n c s0 s1 BI BC
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (input output : Storage) (index : IndexStorage) (candidate : Bool) : Source :=
  match input, output, index, candidate with
  | .fp32, .fp32, .int32, Bool.false => indexCat%[tl.float32, tl.float32, tl.int32, original]
  | .fp32, .fp32, .int32, Bool.true => indexCat%[tl.float32, tl.float32, tl.int32, indexCommuted]
  | .fp32, .fp32, .int64, Bool.false => indexCat%[tl.float32, tl.float32, tl.int64, original]
  | .fp32, .fp32, .int64, Bool.true => indexCat%[tl.float32, tl.float32, tl.int64, indexCommuted]
  | .fp32, .fp16, .int32, Bool.false => indexCat%[tl.float32, tl.float16, tl.int32, original]
  | .fp32, .fp16, .int32, Bool.true => indexCat%[tl.float32, tl.float16, tl.int32, indexCommuted]
  | .fp32, .fp16, .int64, Bool.false => indexCat%[tl.float32, tl.float16, tl.int64, original]
  | .fp32, .fp16, .int64, Bool.true => indexCat%[tl.float32, tl.float16, tl.int64, indexCommuted]
  | .fp32, .bf16, .int32, Bool.false => indexCat%[tl.float32, tl.bfloat16, tl.int32, original]
  | .fp32, .bf16, .int32, Bool.true => indexCat%[tl.float32, tl.bfloat16, tl.int32, indexCommuted]
  | .fp32, .bf16, .int64, Bool.false => indexCat%[tl.float32, tl.bfloat16, tl.int64, original]
  | .fp32, .bf16, .int64, Bool.true => indexCat%[tl.float32, tl.bfloat16, tl.int64, indexCommuted]
  | .fp32, .f8e4, .int32, Bool.false => indexCat%[tl.float32, tl.float8e4nv, tl.int32, original]
  | .fp32, .f8e4, .int32, Bool.true => indexCat%[tl.float32, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp32, .f8e4, .int64, Bool.false => indexCat%[tl.float32, tl.float8e4nv, tl.int64, original]
  | .fp32, .f8e4, .int64, Bool.true => indexCat%[tl.float32, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp32, .f8e5, .int32, Bool.false => indexCat%[tl.float32, tl.float8e5, tl.int32, original]
  | .fp32, .f8e5, .int32, Bool.true => indexCat%[tl.float32, tl.float8e5, tl.int32, indexCommuted]
  | .fp32, .f8e5, .int64, Bool.false => indexCat%[tl.float32, tl.float8e5, tl.int64, original]
  | .fp32, .f8e5, .int64, Bool.true => indexCat%[tl.float32, tl.float8e5, tl.int64, indexCommuted]
  | .fp32, .fp64, .int32, Bool.false => indexCat%[tl.float32, tl.float64, tl.int32, original]
  | .fp32, .fp64, .int32, Bool.true => indexCat%[tl.float32, tl.float64, tl.int32, indexCommuted]
  | .fp32, .fp64, .int64, Bool.false => indexCat%[tl.float32, tl.float64, tl.int64, original]
  | .fp32, .fp64, .int64, Bool.true => indexCat%[tl.float32, tl.float64, tl.int64, indexCommuted]
  | .fp16, .fp32, .int32, Bool.false => indexCat%[tl.float16, tl.float32, tl.int32, original]
  | .fp16, .fp32, .int32, Bool.true => indexCat%[tl.float16, tl.float32, tl.int32, indexCommuted]
  | .fp16, .fp32, .int64, Bool.false => indexCat%[tl.float16, tl.float32, tl.int64, original]
  | .fp16, .fp32, .int64, Bool.true => indexCat%[tl.float16, tl.float32, tl.int64, indexCommuted]
  | .fp16, .fp16, .int32, Bool.false => indexCat%[tl.float16, tl.float16, tl.int32, original]
  | .fp16, .fp16, .int32, Bool.true => indexCat%[tl.float16, tl.float16, tl.int32, indexCommuted]
  | .fp16, .fp16, .int64, Bool.false => indexCat%[tl.float16, tl.float16, tl.int64, original]
  | .fp16, .fp16, .int64, Bool.true => indexCat%[tl.float16, tl.float16, tl.int64, indexCommuted]
  | .fp16, .bf16, .int32, Bool.false => indexCat%[tl.float16, tl.bfloat16, tl.int32, original]
  | .fp16, .bf16, .int32, Bool.true => indexCat%[tl.float16, tl.bfloat16, tl.int32, indexCommuted]
  | .fp16, .bf16, .int64, Bool.false => indexCat%[tl.float16, tl.bfloat16, tl.int64, original]
  | .fp16, .bf16, .int64, Bool.true => indexCat%[tl.float16, tl.bfloat16, tl.int64, indexCommuted]
  | .fp16, .f8e4, .int32, Bool.false => indexCat%[tl.float16, tl.float8e4nv, tl.int32, original]
  | .fp16, .f8e4, .int32, Bool.true => indexCat%[tl.float16, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp16, .f8e4, .int64, Bool.false => indexCat%[tl.float16, tl.float8e4nv, tl.int64, original]
  | .fp16, .f8e4, .int64, Bool.true => indexCat%[tl.float16, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp16, .f8e5, .int32, Bool.false => indexCat%[tl.float16, tl.float8e5, tl.int32, original]
  | .fp16, .f8e5, .int32, Bool.true => indexCat%[tl.float16, tl.float8e5, tl.int32, indexCommuted]
  | .fp16, .f8e5, .int64, Bool.false => indexCat%[tl.float16, tl.float8e5, tl.int64, original]
  | .fp16, .f8e5, .int64, Bool.true => indexCat%[tl.float16, tl.float8e5, tl.int64, indexCommuted]
  | .fp16, .fp64, .int32, Bool.false => indexCat%[tl.float16, tl.float64, tl.int32, original]
  | .fp16, .fp64, .int32, Bool.true => indexCat%[tl.float16, tl.float64, tl.int32, indexCommuted]
  | .fp16, .fp64, .int64, Bool.false => indexCat%[tl.float16, tl.float64, tl.int64, original]
  | .fp16, .fp64, .int64, Bool.true => indexCat%[tl.float16, tl.float64, tl.int64, indexCommuted]
  | .bf16, .fp32, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float32, tl.int32, original]
  | .bf16, .fp32, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float32, tl.int32, indexCommuted]
  | .bf16, .fp32, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float32, tl.int64, original]
  | .bf16, .fp32, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float32, tl.int64, indexCommuted]
  | .bf16, .fp16, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float16, tl.int32, original]
  | .bf16, .fp16, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float16, tl.int32, indexCommuted]
  | .bf16, .fp16, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float16, tl.int64, original]
  | .bf16, .fp16, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float16, tl.int64, indexCommuted]
  | .bf16, .bf16, .int32, Bool.false => indexCat%[tl.bfloat16, tl.bfloat16, tl.int32, original]
  | .bf16, .bf16, .int32, Bool.true => indexCat%[tl.bfloat16, tl.bfloat16, tl.int32, indexCommuted]
  | .bf16, .bf16, .int64, Bool.false => indexCat%[tl.bfloat16, tl.bfloat16, tl.int64, original]
  | .bf16, .bf16, .int64, Bool.true => indexCat%[tl.bfloat16, tl.bfloat16, tl.int64, indexCommuted]
  | .bf16, .f8e4, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int32, original]
  | .bf16, .f8e4, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int32, indexCommuted]
  | .bf16, .f8e4, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int64, original]
  | .bf16, .f8e4, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int64, indexCommuted]
  | .bf16, .f8e5, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float8e5, tl.int32, original]
  | .bf16, .f8e5, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float8e5, tl.int32, indexCommuted]
  | .bf16, .f8e5, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float8e5, tl.int64, original]
  | .bf16, .f8e5, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float8e5, tl.int64, indexCommuted]
  | .bf16, .fp64, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float64, tl.int32, original]
  | .bf16, .fp64, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float64, tl.int32, indexCommuted]
  | .bf16, .fp64, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float64, tl.int64, original]
  | .bf16, .fp64, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float64, tl.int64, indexCommuted]
  | .f8e4, .fp32, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float32, tl.int32, original]
  | .f8e4, .fp32, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float32, tl.int32, indexCommuted]
  | .f8e4, .fp32, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float32, tl.int64, original]
  | .f8e4, .fp32, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float32, tl.int64, indexCommuted]
  | .f8e4, .fp16, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float16, tl.int32, original]
  | .f8e4, .fp16, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float16, tl.int32, indexCommuted]
  | .f8e4, .fp16, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float16, tl.int64, original]
  | .f8e4, .fp16, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float16, tl.int64, indexCommuted]
  | .f8e4, .bf16, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int32, original]
  | .f8e4, .bf16, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int32, indexCommuted]
  | .f8e4, .bf16, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int64, original]
  | .f8e4, .bf16, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int64, indexCommuted]
  | .f8e4, .f8e4, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int32, original]
  | .f8e4, .f8e4, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e4, .f8e4, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int64, original]
  | .f8e4, .f8e4, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e4, .f8e5, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float8e5, tl.int32, original]
  | .f8e4, .f8e5, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float8e5, tl.int32, indexCommuted]
  | .f8e4, .f8e5, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float8e5, tl.int64, original]
  | .f8e4, .f8e5, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float8e5, tl.int64, indexCommuted]
  | .f8e4, .fp64, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float64, tl.int32, original]
  | .f8e4, .fp64, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float64, tl.int32, indexCommuted]
  | .f8e4, .fp64, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float64, tl.int64, original]
  | .f8e4, .fp64, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float64, tl.int64, indexCommuted]
  | .f8e5, .fp32, .int32, Bool.false => indexCat%[tl.float8e5, tl.float32, tl.int32, original]
  | .f8e5, .fp32, .int32, Bool.true => indexCat%[tl.float8e5, tl.float32, tl.int32, indexCommuted]
  | .f8e5, .fp32, .int64, Bool.false => indexCat%[tl.float8e5, tl.float32, tl.int64, original]
  | .f8e5, .fp32, .int64, Bool.true => indexCat%[tl.float8e5, tl.float32, tl.int64, indexCommuted]
  | .f8e5, .fp16, .int32, Bool.false => indexCat%[tl.float8e5, tl.float16, tl.int32, original]
  | .f8e5, .fp16, .int32, Bool.true => indexCat%[tl.float8e5, tl.float16, tl.int32, indexCommuted]
  | .f8e5, .fp16, .int64, Bool.false => indexCat%[tl.float8e5, tl.float16, tl.int64, original]
  | .f8e5, .fp16, .int64, Bool.true => indexCat%[tl.float8e5, tl.float16, tl.int64, indexCommuted]
  | .f8e5, .bf16, .int32, Bool.false => indexCat%[tl.float8e5, tl.bfloat16, tl.int32, original]
  | .f8e5, .bf16, .int32, Bool.true => indexCat%[tl.float8e5, tl.bfloat16, tl.int32, indexCommuted]
  | .f8e5, .bf16, .int64, Bool.false => indexCat%[tl.float8e5, tl.bfloat16, tl.int64, original]
  | .f8e5, .bf16, .int64, Bool.true => indexCat%[tl.float8e5, tl.bfloat16, tl.int64, indexCommuted]
  | .f8e5, .f8e4, .int32, Bool.false => indexCat%[tl.float8e5, tl.float8e4nv, tl.int32, original]
  | .f8e5, .f8e4, .int32, Bool.true => indexCat%[tl.float8e5, tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e5, .f8e4, .int64, Bool.false => indexCat%[tl.float8e5, tl.float8e4nv, tl.int64, original]
  | .f8e5, .f8e4, .int64, Bool.true => indexCat%[tl.float8e5, tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e5, .f8e5, .int32, Bool.false => indexCat%[tl.float8e5, tl.float8e5, tl.int32, original]
  | .f8e5, .f8e5, .int32, Bool.true => indexCat%[tl.float8e5, tl.float8e5, tl.int32, indexCommuted]
  | .f8e5, .f8e5, .int64, Bool.false => indexCat%[tl.float8e5, tl.float8e5, tl.int64, original]
  | .f8e5, .f8e5, .int64, Bool.true => indexCat%[tl.float8e5, tl.float8e5, tl.int64, indexCommuted]
  | .f8e5, .fp64, .int32, Bool.false => indexCat%[tl.float8e5, tl.float64, tl.int32, original]
  | .f8e5, .fp64, .int32, Bool.true => indexCat%[tl.float8e5, tl.float64, tl.int32, indexCommuted]
  | .f8e5, .fp64, .int64, Bool.false => indexCat%[tl.float8e5, tl.float64, tl.int64, original]
  | .f8e5, .fp64, .int64, Bool.true => indexCat%[tl.float8e5, tl.float64, tl.int64, indexCommuted]
  | .fp64, .fp32, .int32, Bool.false => indexCat%[tl.float64, tl.float32, tl.int32, original]
  | .fp64, .fp32, .int32, Bool.true => indexCat%[tl.float64, tl.float32, tl.int32, indexCommuted]
  | .fp64, .fp32, .int64, Bool.false => indexCat%[tl.float64, tl.float32, tl.int64, original]
  | .fp64, .fp32, .int64, Bool.true => indexCat%[tl.float64, tl.float32, tl.int64, indexCommuted]
  | .fp64, .fp16, .int32, Bool.false => indexCat%[tl.float64, tl.float16, tl.int32, original]
  | .fp64, .fp16, .int32, Bool.true => indexCat%[tl.float64, tl.float16, tl.int32, indexCommuted]
  | .fp64, .fp16, .int64, Bool.false => indexCat%[tl.float64, tl.float16, tl.int64, original]
  | .fp64, .fp16, .int64, Bool.true => indexCat%[tl.float64, tl.float16, tl.int64, indexCommuted]
  | .fp64, .bf16, .int32, Bool.false => indexCat%[tl.float64, tl.bfloat16, tl.int32, original]
  | .fp64, .bf16, .int32, Bool.true => indexCat%[tl.float64, tl.bfloat16, tl.int32, indexCommuted]
  | .fp64, .bf16, .int64, Bool.false => indexCat%[tl.float64, tl.bfloat16, tl.int64, original]
  | .fp64, .bf16, .int64, Bool.true => indexCat%[tl.float64, tl.bfloat16, tl.int64, indexCommuted]
  | .fp64, .f8e4, .int32, Bool.false => indexCat%[tl.float64, tl.float8e4nv, tl.int32, original]
  | .fp64, .f8e4, .int32, Bool.true => indexCat%[tl.float64, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp64, .f8e4, .int64, Bool.false => indexCat%[tl.float64, tl.float8e4nv, tl.int64, original]
  | .fp64, .f8e4, .int64, Bool.true => indexCat%[tl.float64, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp64, .f8e5, .int32, Bool.false => indexCat%[tl.float64, tl.float8e5, tl.int32, original]
  | .fp64, .f8e5, .int32, Bool.true => indexCat%[tl.float64, tl.float8e5, tl.int32, indexCommuted]
  | .fp64, .f8e5, .int64, Bool.false => indexCat%[tl.float64, tl.float8e5, tl.int64, original]
  | .fp64, .f8e5, .int64, Bool.true => indexCat%[tl.float64, tl.float8e5, tl.int64, indexCommuted]
  | .fp64, .fp64, .int32, Bool.false => indexCat%[tl.float64, tl.float64, tl.int32, original]
  | .fp64, .fp64, .int32, Bool.true => indexCat%[tl.float64, tl.float64, tl.int32, indexCommuted]
  | .fp64, .fp64, .int64, Bool.false => indexCat%[tl.float64, tl.float64, tl.int64, original]
  | .fp64, .fp64, .int64, Bool.true => indexCat%[tl.float64, tl.float64, tl.int64, indexCommuted]
```
</details>
