# Spec sheet — `bench/tritonbench_g/masked_add_cuda/Correct.lean`

**Python source:** `bench/tritonbench_g/masked_add_cuda/masked_add_cuda.py`

## Public theorem: `masked_add_integer_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete tested source profiles implement the in-place mathematical
formula. Integer mask values are arbitrary signed integers; exactly zero writes.
The contract includes termination, active-address safety and the full frame. -/
```
</details>

**Statement:**
```lean
specification masked_add_integer_real_correct (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) :
    Spec.Real ((integerIO storage candidate grad p mask n alpha B).Implements
      (fun _ _ _ xs ys j => ys j + xs j * alpha))
```

**Closed-form spec defs (transitive):** `integerIO`, `integer_kernel`

<details><summary><code>integerIO</code></summary>

```
/-- Integer mask storage is a genuine input channel. The input and output grad
windows share the same buffer, and nonzero mask values veto stores. -/
```
```lean
def integerIO (storage : MaskStorage) (candidate : Bool) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : IntMasked2DKernelIO₂ where
  kernel := (integer_kernel storage candidate grad p mask n alpha B).eraseDType
  projection := integer_supported ..
  in1 := p
  in2 := grad
  mbuf := mask
  out := grad
  B := B
  read1 := fun pid _ j => pid * B + j.val
  read2 := fun pid _ j => pid * B + j.val
  readm := fun pid _ j => pid * B + j.val
  write := fun pid _ j => pid * B + j.val
  mask := fun pid _ j => pid * B + j.val < n
  writeMask := fun pid _ ms j => pid * B + j.val < n ∧ ms j = 0
```
</details>

<details><summary><code>integer_kernel</code></summary>

```
/-- Complete fp32 data source, with signed int32 or int64 mask storage. -/
```
```lean
def integer_kernel (storage : MaskStorage) (candidate : Bool)
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, candidate with
  | .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
```
</details>

## Public theorem: `masked_add_fp64_integer_real_correct`

<details><summary>docstring</summary>

```
/-- Both float64-region transcriptions satisfy the mathematical formula.
This is a real-only result; the pending FP translation gap is documented in README. Integer mask values are arbitrary signed integers; exactly zero writes.
The contract includes termination, active-address safety and the full frame. -/
```
</details>

**Statement:**
```lean
specification masked_add_fp64_integer_real_correct (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) :
    Spec.Real ((fp64IntegerIO storage candidate grad p mask n alpha B).Implements
      (fun _ _ _ xs ys j => ys j + xs j * alpha))
```

**Closed-form spec defs (transitive):** `fp64IntegerIO`, `integer_fp64_kernel`

<details><summary><code>fp64IntegerIO</code></summary>

```
/-- Integer mask storage is a genuine input channel. The input and output grad
windows share the same buffer, and nonzero mask values veto stores. -/
```
```lean
def fp64IntegerIO (storage : MaskStorage) (candidate : Bool) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : IntMasked2DKernelIO₂ where
  kernel := (integer_fp64_kernel storage candidate grad p mask n alpha B).eraseDType
  projection := fp64_integer_supported ..
  in1 := p
  in2 := grad
  mbuf := mask
  out := grad
  B := B
  read1 := fun pid _ j => pid * B + j.val
  read2 := fun pid _ j => pid * B + j.val
  readm := fun pid _ j => pid * B + j.val
  write := fun pid _ j => pid * B + j.val
  mask := fun pid _ j => pid * B + j.val < n
  writeMask := fun pid _ ms j => pid * B + j.val < n ∧ ms j = 0
```
</details>

<details><summary><code>integer_fp64_kernel</code></summary>

```
/-- Float64-region transcription for real-only verification. Current DSL load
profile erasure and mixed runtime-scalar precision prevent a faithful FP claim. -/
```
```lean
def integer_fp64_kernel (storage : MaskStorage) (candidate : Bool)
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, candidate with
  | .int32, Bool.false =>
      maskedAddSource%[tl.float64, tl.float64, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.false =>
      maskedAddSource%[tl.float64, tl.float64, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int32, Bool.true =>
      maskedAddSource%[tl.float64, tl.float64, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .int64, Bool.true =>
      maskedAddSource%[tl.float64, tl.float64, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE

/-- Floating storage profiles whose runtime fp32 alpha promotes arithmetic to fp32. -/
inductive FloatStorage where
  | fp32 | fp16 | bf16
  deriving DecidableEq, Repr
```
</details>

## Public theorem: `masked_add_typed_integer_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete typed source profiles implement the in-place mathematical
formula. Integer mask values are arbitrary signed integers; exactly zero writes.
The contract includes termination, active-address safety and the full frame. -/
```
</details>

**Statement:**
```lean
specification masked_add_typed_integer_real_correct (gradType pType : FloatStorage)
    (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) :
    Spec.Real ((fullIntegerIO gradType pType storage candidate grad p mask n alpha B).Implements
      (fun _ _ _ xs ys j => ys j + xs j * alpha))
```

**Closed-form spec defs (transitive):** `fullIntegerIO`, `typed_integer_kernel`

<details><summary><code>fullIntegerIO</code></summary>

```
/-- Integer mask storage is a genuine input channel. The input and output grad
windows share the same buffer, and nonzero mask values veto stores. -/
```
```lean
def fullIntegerIO (gradType pType : FloatStorage) (storage : MaskStorage) (candidate : Bool)
    (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : IntMasked2DKernelIO₂ where
  kernel := (typed_integer_kernel gradType pType storage candidate grad p mask n alpha B).eraseDType
  projection := full_integer_supported ..
  in1 := p
  in2 := grad
  mbuf := mask
  out := grad
  B := B
  read1 := fun pid _ j => pid * B + j.val
  read2 := fun pid _ j => pid * B + j.val
  readm := fun pid _ j => pid * B + j.val
  write := fun pid _ j => pid * B + j.val
  mask := fun pid _ j => pid * B + j.val < n
  writeMask := fun pid _ ms j => pid * B + j.val < n ∧ ms j = 0
```
</details>

<details><summary><code>typed_integer_kernel</code></summary>

```
/-- Full independent gradient/data storage combinations. The final cast spells
out Triton's implicit conversion back to the gradient pointer element type. -/
```
```lean
def typed_integer_kernel (gradType pType : FloatStorage) (maskType : MaskStorage)
    (candidate : Bool) (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradType, pType, maskType, candidate with
  | .fp32, .fp32, .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp32, .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp32, .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp32, .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.float16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.float16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.float16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .fp16, .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.float16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int32, Bool.false =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int32, Bool.true =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int64, Bool.false =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp32, .bf16, .int64, Bool.true =>
      maskedAddSource%[tl.float32, tl.bfloat16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int32, Bool.false =>
      maskedAddSource%[tl.float16, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int32, Bool.true =>
      maskedAddSource%[tl.float16, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int64, Bool.false =>
      maskedAddSource%[tl.float16, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp32, .int64, Bool.true =>
      maskedAddSource%[tl.float16, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int32, Bool.false =>
      maskedAddSource%[tl.float16, tl.float16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int32, Bool.true =>
      maskedAddSource%[tl.float16, tl.float16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int64, Bool.false =>
      maskedAddSource%[tl.float16, tl.float16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .fp16, .int64, Bool.true =>
      maskedAddSource%[tl.float16, tl.float16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int32, Bool.false =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int32, Bool.true =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int64, Bool.false =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .fp16, .bf16, .int64, Bool.true =>
      maskedAddSource%[tl.float16, tl.bfloat16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int32, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int32, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int64, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp32, .int64, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float32, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int32, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int32, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int64, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .fp16, .int64, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.float16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int32, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int32, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int32, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int32, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int64, Bool.false =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int64, original]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
  | .bf16, .bf16, .int64, Bool.true =>
      maskedAddSource%[tl.bfloat16, tl.bfloat16, tl.int64, commuted]
        grad_ptr p_ptr p_mask_ptr n_elements alpha BLOCK_SIZE
```
</details>
