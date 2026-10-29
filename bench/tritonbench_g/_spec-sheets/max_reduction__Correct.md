# Spec sheet — `bench/tritonbench_g/max_reduction/Correct.lean`

**Python source:** `bench/tritonbench_g/max_reduction/max_reduction.py`

## Public theorem: `stage2_math_correct`

<details><summary>docstring</summary>

```
/-- Full second-stage IO correctness against a finite mathematical maximum. -/
```
</details>

**Statement:**
```lean
specification stage2_math_correct (mid out : RegionName) (M B : Nat) (hB : 0 < B) :
    stage2IO mid out M B ⊨ fun _ xs _ => blockMaximum M B 0 xs
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `stage1_real_correct`

<details><summary>docstring</summary>

```
/-- Each source version independently implements the finite maximum of the
masked original input window, terminates, and preserves every other flat cell. -/
```
</details>

**Statement:**
```lean
specification stage1_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage1IO storage v inp out M B ⊨
      fun pid xs _ => blockMaximum M B pid xs)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `typedStage1IO`, `stage1`

<details><summary><code>typedStage1IO</code></summary>

```lean
def typedStage1IO (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    MaskedTileKernelIO₁ :=
  { maxKernel1IO inp out M B with
    kernel := (stage1 storage v inp out M B).eraseDType
    projection := stage1_supported storage v inp out M B }
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

## Public theorem: `stage2_real_correct`

<details><summary>docstring</summary>

```
/-- Each second-stage source independently computes the finite maximum of the
masked intermediate window, with successful execution and a full flat frame. -/
```
</details>

**Statement:**
```lean
specification stage2_real_correct (storage : Storage) (v : Bool)
    (inp out : RegionName) (M B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real (typedStage2IO storage v inp out M B ⊨
      fun _ xs _ => blockMaximum M B 0 xs)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `typedStage2IO`, `stage2`

<details><summary><code>typedStage2IO</code></summary>

```lean
def typedStage2IO (storage : Storage) (v : Bool) (inp out : RegionName) (M B : Nat) :
    MaskedTileKernelIO₁ :=
  { stage2IO inp out M B with
    kernel := (stage2 storage v inp out M B).eraseDType
    projection := stage2_supported storage v inp out M B }
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

## Public theorem: `dimension_real_correct`

<details><summary>docstring</summary>

```
/-- Both full indexed-reduction sources independently compute row maxima and
leftmost indices from the complete masked input window. The signed index
output, value output, successful execution and full flat frame are all covered.
Distinct output regions and injective row addresses prevent conflicting stores. -/
```
</details>

**Statement:**
```lean
specification dimension_real_correct (storage : Storage) (v : Bool)
    (inp ov oi : RegionName) (M N K BM BN : Nat)
    (hN : 0 < BN) (hRegions : ov ≠ oi)
    (hOutInj : Function.Injective (fun i : Fin BM => i.val * K)) :
    VeriTile.Spec.Real (DimensionIO storage v inp ov oi M N K BM BN)
```

**Assumptions / layout contracts:**
- `hN : 0 < BN`
- `hRegions : ov ≠ oi`
- `hOutInj : Function.Injective (fun i : Fin BM => i.val * K)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
