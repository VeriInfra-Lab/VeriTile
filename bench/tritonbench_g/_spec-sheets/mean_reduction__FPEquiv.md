# Spec sheet — `bench/tritonbench_g/mean_reduction/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/mean_reduction/mean_reduction.py`

## Public theorem: `mean_index_equiv`

<details><summary>docstring</summary>

```
/-- The complete accumulation loop and its precision conversions are unchanged.
Only exact natural-number address arithmetic is commuted; every execution result,
including failure, all registers and all memory, is preserved. -/
```
</details>

**Statement:**
```lean
specification mean_index_equiv (input output : Storage) (X Mean : RegionName)
    (M N BM BN : Nat) (_hM : 0 < BM) (_hN : 0 < BN) :
    program input output Bool.false X Mean M N BM BN ≡[assumptions]
      program input output Bool.true X Mean M N BM BN
```

**Assumptions / layout contracts:**
- `_hM : 0 < BM`
- `_hN : 0 < BN`

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`

<details><summary><code>program</code></summary>

```lean
def program (input output : Storage) (commuted : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) : Program where
  kernel := typedKernel input output commuted X Mean M N BM BN
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (input output : Storage) (commuted : Bool) : Source :=
  match input, commuted with
  | .fp32, Bool.false => meanOutputs%[tl.float32, original] output
  | .fp32, Bool.true => meanOutputs%[tl.float32, indexCommuted] output
  | .fp16, Bool.false => meanOutputs%[tl.float16, original] output
  | .fp16, Bool.true => meanOutputs%[tl.float16, indexCommuted] output
  | .bf16, Bool.false => meanOutputs%[tl.bfloat16, original] output
  | .bf16, Bool.true => meanOutputs%[tl.bfloat16, indexCommuted] output
  | .fp64, Bool.false => meanOutputs%[tl.float64, original] output
  | .fp64, Bool.true => meanOutputs%[tl.float64, indexCommuted] output
```
</details>
