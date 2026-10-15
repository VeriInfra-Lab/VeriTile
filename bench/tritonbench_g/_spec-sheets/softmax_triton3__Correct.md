# Spec sheet — `bench/tritonbench_g/softmax_triton3/Correct.lean`

**Python source:** `bench/tritonbench_g/softmax_triton3/softmax_triton3.py`

## Public theorem: `softmax_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete typed sources implement mathematical softmax, including
termination, active-address safety and preservation of every other memory cell. -/
```
</details>

**Statement:**
```lean
specification softmax_typed_real_correct (inputType maskType : Storage) (reciprocal : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hB : 0 < B) :
    Spec.Real (typedIO inputType maskType reciprocal out inp mask stride n B ⊨
      fun xs i => Real.exp (xs i) /
        ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `typedIO`, `typedKernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (inputType maskType : Storage) (reciprocal : Bool) (out inp mask : RegionName)
    (stride n B : Nat) : MaskedKernelIO₁ where
  kernel := (typedKernel inputType maskType reciprocal out inp mask stride n B Bool.false).eraseDType
  projection := typed_supported ..
  inp := inp
  out := out
  B := B
  read := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n
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

## Public theorem: `softmax_typed_masked_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete typed sources implement softmax of input plus additive mask.
The stable shift uses the input maximum, and cancels from this real formula. -/
```
</details>

**Statement:**
```lean
specification softmax_typed_masked_real_correct (inputType maskType : Storage) (reciprocal : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hB : 0 < B) :
    Spec.Real (typedMaskedIO inputType maskType reciprocal out inp mask stride n B ⊨
      fun xs ms i => Real.exp (xs i + ms i) /
        ∑ j : Fin B, if j.val < n then Real.exp (xs j + ms j) else 0)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`

**Closed-form spec defs (transitive):** `typedMaskedIO`, `typedKernel`

<details><summary><code>typedMaskedIO</code></summary>

```
/-- The optional additive mask is a genuine second input channel with independent
storage; all three windows use the source's shared row stride and tail mask. -/
```
```lean
def typedMaskedIO (inputType maskType : Storage) (reciprocal : Bool) (out inp mask : RegionName)
    (stride n B : Nat) : MaskedKernelIO₂ where
  kernel := (typedKernel inputType maskType reciprocal out inp mask stride n B Bool.true).eraseDType
  projection := typed_supported ..
  in1 := inp
  in2 := mask
  out := out
  B := B
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n
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
