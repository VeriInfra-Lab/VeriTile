# Spec sheet — `bench/tritonbench_g/softmax_triton3/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/softmax_triton3/softmax_triton3.py`

## Public theorem: `softmax_reciprocal_equiv`

**Statement:**
```lean
specification softmax_reciprocal_equiv (inputType maskType : Storage)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool)
    (R : FP.ScalarArithmetic.Rules) :
    original inputType maskType out inp mask stride n B hasMask ≡[R]
      reciprocal inputType maskType out inp mask stride n B hasMask
```

**Assumptions / layout contracts:**
- `hasMask : Bool`

**Closed-form spec defs (transitive):** `original`, `reciprocal`, `typedKernel`, `divisionSite`, `Source`

<details><summary><code>original</code></summary>

```lean
def original (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) : Program where
  kernel := typedKernel inputType maskType Bool.false out inp mask stride n B hasMask
  domain := [divisionSite inputType maskType out inp mask stride n B hasMask]
```
</details>

<details><summary><code>reciprocal</code></summary>

```lean
def reciprocal (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) : Program :=
  { original inputType maskType out inp mask stride n B hasMask with
    kernel := typedKernel inputType maskType Bool.true out inp mask stride n B hasMask }
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (inputType maskType : Storage) (reciprocal : Bool) : Source :=
  match inputType, maskType, reciprocal with
  | .fp32, .fp32, Bool.false => softmax3Source%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => softmax3Source%[tl.float32, tl.float32, reciprocal]
  | .fp32, .fp16, Bool.false => softmax3Source%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => softmax3Source%[tl.float32, tl.float16, reciprocal]
  | .fp32, .bf16, Bool.false => softmax3Source%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => softmax3Source%[tl.float32, tl.bfloat16, reciprocal]
  | .fp16, .fp32, Bool.false => softmax3Source%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => softmax3Source%[tl.float16, tl.float32, reciprocal]
  | .fp16, .fp16, Bool.false => softmax3Source%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => softmax3Source%[tl.float16, tl.float16, reciprocal]
  | .fp16, .bf16, Bool.false => softmax3Source%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => softmax3Source%[tl.float16, tl.bfloat16, reciprocal]
  | .bf16, .fp32, Bool.false => softmax3Source%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => softmax3Source%[tl.bfloat16, tl.float32, reciprocal]
  | .bf16, .fp16, Bool.false => softmax3Source%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => softmax3Source%[tl.bfloat16, tl.float16, reciprocal]
  | .bf16, .bf16, Bool.false => softmax3Source%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => softmax3Source%[tl.bfloat16, tl.bfloat16, reciprocal]
```
</details>

<details><summary><code>divisionSite</code></summary>

```
/-- Finiteness of actual computed numerators and finiteness/nonzero of the actual
row sum, including all evaluated inactive lanes. The prefix includes the optional
additive mask and its cast. No intrinsic exponential identity is assumed. -/
```
```lean
def divisionSite (inputType maskType : Storage) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) : Site :=
  .divScalar (Source.prefixBefore
    (typedKernel inputType maskType Bool.false out inp mask stride n B hasMask).surfaceBody
    "softmax_output") B [] (.ref .real [B] "numerator") (.ref .real [] "denominator")
```
</details>

<details><summary><code>Source</code></summary>

```lean
abbrev Source := RegionName → RegionName → RegionName → Nat → Nat → Nat → Bool → ComputeKernel
```
</details>
