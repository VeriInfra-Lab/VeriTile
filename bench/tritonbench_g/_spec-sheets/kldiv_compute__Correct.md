# Spec sheet — `bench/tritonbench_g/kldiv_compute/Correct.lean`

**Python source:** `bench/tritonbench_g/kldiv_compute/kldiv_compute.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Independent pointwise real formula for either complete typed source, including
successful flat execution, active-address bounds and all unwritten memory. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (x y output : Storage) (v : Bool)
    (_hSource : sourceSupported x y = Bool.true)
    (X Y O : RegionName) (N B : Nat) :
    VeriTile.Spec.Real (typedIO x y output v X Y O N B ⊨
      fun xs ys i => klDivSpec (xs i) (ys i))
```

**Assumptions / layout contracts:**
- `_hSource : sourceSupported x y = Bool.true`

**Closed-form spec defs (transitive):** `sourceSupported`

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
