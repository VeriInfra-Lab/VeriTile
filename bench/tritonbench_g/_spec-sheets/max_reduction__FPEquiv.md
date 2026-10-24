# Spec sheet — `bench/tritonbench_g/max_reduction/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/max_reduction/max_reduction.py`

## Public theorem: `stage1_index_equiv`

<details><summary>docstring</summary>

```
/-- Complete first-stage typed executions agree after commuting the natural
index product. Every floating load, reduction, narrowing cast and store stays
in the same order and uses the same backend. No floating atom is admitted. -/
```
</details>

**Statement:**
```lean
specification stage1_index_equiv (storage : Storage) (inp out : RegionName) (M B : Nat) :
    stage1Program storage Bool.false inp out M B ≡[assumptions]
      stage1Program storage Bool.true inp out M B
```

**Closed-form spec defs (transitive):** `stage1Program`, `assumptions`, `stage1`

<details><summary><code>stage1Program</code></summary>

```lean
def stage1Program (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) : Program where
  kernel := stage1 storage v inp out M B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>stage1</code></summary>

```lean
def stage1 (storage : Storage) (candidate : Bool) : ValueSource :=
  match storage, candidate with
  | .fp32, Bool.false => maxStage1%[tl.float32, original]
  | .fp32, Bool.true => maxStage1%[tl.float32, indexCommuted]
  | .fp16, Bool.false => maxStage1%[tl.float16, original]
  | .fp16, Bool.true => maxStage1%[tl.float16, indexCommuted]
  | .bf16, Bool.false => maxStage1%[tl.bfloat16, original]
  | .bf16, Bool.true => maxStage1%[tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => maxStage1%[tl.float8e4nv, original]
  | .f8e4, Bool.true => maxStage1%[tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => maxStage1%[tl.float8e5, original]
  | .f8e5, Bool.true => maxStage1%[tl.float8e5, indexCommuted]
  | .fp64, Bool.false => maxStage1%[tl.float64, original]
  | .fp64, Bool.true => maxStage1%[tl.float64, indexCommuted]
```
</details>

## Public theorem: `stage2_mask_equiv`

<details><summary>docstring</summary>

```
/-- Complete second-stage typed executions agree after complementing the
reversed natural bound. The full input mask, not a precomputed maximum, is
rewritten. No floating atom is admitted. -/
```
</details>

**Statement:**
```lean
specification stage2_mask_equiv (storage : Storage) (inp out : RegionName) (M B : Nat) :
    stage2Program storage Bool.false inp out M B ≡[assumptions]
      stage2Program storage Bool.true inp out M B
```

**Closed-form spec defs (transitive):** `stage2Program`, `assumptions`, `stage2`

<details><summary><code>stage2Program</code></summary>

```lean
def stage2Program (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) : Program where
  kernel := stage2 storage v inp out M B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>stage2</code></summary>

```lean
def stage2 (storage : Storage) (candidate : Bool) : ValueSource :=
  match storage, candidate with
  | .fp32, Bool.false => maxStage2%[tl.float32, original]
  | .fp32, Bool.true => maxStage2%[tl.float32, maskComplement]
  | .fp16, Bool.false => maxStage2%[tl.float16, original]
  | .fp16, Bool.true => maxStage2%[tl.float16, maskComplement]
  | .bf16, Bool.false => maxStage2%[tl.bfloat16, original]
  | .bf16, Bool.true => maxStage2%[tl.bfloat16, maskComplement]
  | .f8e4, Bool.false => maxStage2%[tl.float8e4nv, original]
  | .f8e4, Bool.true => maxStage2%[tl.float8e4nv, maskComplement]
  | .f8e5, Bool.false => maxStage2%[tl.float8e5, original]
  | .f8e5, Bool.true => maxStage2%[tl.float8e5, maskComplement]
  | .fp64, Bool.false => maxStage2%[tl.float64, original]
  | .fp64, Bool.true => maxStage2%[tl.float64, maskComplement]

/-- Full dimension-reduction source, including both masked output streams.
`out_index` is signed int64 storage as allocated by the Python wrapper.
The explicit store casts expose conversions implicit in Triton's store. -/
syntax "maxDimension%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(maxDimension%[$dtype:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))
      else `(tritonStmt| m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M)))
    let writeValue : TSyntax `tritonStmt ← match dtype with
      | `(tritonDType| tl.bfloat16) =>
          `(tritonStmt| tl.store(out_value_ptrs, result_value.to(tl.bfloat16), mask=mask1))
      | _ => `(tritonStmt| tl.store(out_value_ptrs, result_value, mask=mask1))
    `(fun (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [inp] [out_value, out_index] (triton {
  tl.region inp = $dtype, out_value = $dtype, out_index = tl.int64
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  $rowIndex
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  $writeValue
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}).surfaceBody)
```
</details>

## Public theorem: `dimension_index_equiv`

<details><summary>docstring</summary>

```
/-- The complete dimension kernel preserves both floating values and signed
integer output indices while commuting only the natural row-index product.
Both selected value and index retain the paired indexed-max backend. -/
```
</details>

**Statement:**
```lean
specification dimension_index_equiv (storage : Storage) (inp ov oi : RegionName)
    (M N K BM BN : Nat) :
    dimensionProgram storage Bool.false inp ov oi M N K BM BN ≡[assumptions]
      dimensionProgram storage Bool.true inp ov oi M N K BM BN
```

**Closed-form spec defs (transitive):** `dimensionProgram`, `assumptions`, `dimensionSource`

<details><summary><code>dimensionProgram</code></summary>

```lean
def dimensionProgram (storage : Storage) (v : Bool) (inp ov oi : RegionName)
    (M N K BM BN : Nat) : Program where
  kernel := dimensionSource storage v inp ov oi M N K BM BN
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>dimensionSource</code></summary>

```lean
def dimensionSource (storage : Storage) (candidate : Bool)
    (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  (match storage, candidate with
  | .fp32, Bool.false => maxDimension%[tl.float32, original]
  | .fp32, Bool.true => maxDimension%[tl.float32, indexCommuted]
  | .fp16, Bool.false => maxDimension%[tl.float16, original]
  | .fp16, Bool.true => maxDimension%[tl.float16, indexCommuted]
  | .bf16, Bool.false => maxDimension%[tl.bfloat16, original]
  | .bf16, Bool.true => maxDimension%[tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => maxDimension%[tl.float8e4nv, original]
  | .f8e4, Bool.true => maxDimension%[tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => maxDimension%[tl.float8e5, original]
  | .f8e5, Bool.true => maxDimension%[tl.float8e5, indexCommuted]
  | .fp64, Bool.false => maxDimension%[tl.float64, original]
  | .fp64, Bool.true => maxDimension%[tl.float64, indexCommuted])
    inp out_value out_index M N K BLOCK_M BLOCK_N
```
</details>
