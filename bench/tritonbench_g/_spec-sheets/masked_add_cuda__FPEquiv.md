# Spec sheet — `bench/tritonbench_g/masked_add_cuda/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/masked_add_cuda/masked_add_cuda.py`

## Public theorem: `masked_add_commuted_equiv`

**Statement:**
```lean
specification masked_add_commuted_equiv (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) (R : FP.ScalarArithmetic.Rules) :
    original storage grad p mask n alpha B ≡[R] commuted storage grad p mask n alpha B
```

**Closed-form spec defs (transitive):** `original`, `commuted`, `integer_kernel`, `additionSite`

<details><summary><code>original</code></summary>

```lean
def original (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program where
  kernel := integer_kernel storage Bool.false grad p mask n alpha B
  domain := [additionSite storage grad p mask n alpha B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program :=
  { original storage grad p mask n alpha B with
    kernel := integer_kernel storage Bool.true grad p mask n alpha B }
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

<details><summary><code>additionSite</code></summary>

```lean
def additionSite (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Site :=
  .addFP32 (Source.prefixBefore
    (integer_kernel storage Bool.false grad p mask n alpha B).surfaceBody "grad" 1) [B]
    (.ref .real [B] "grad")
    (.mul .real .scalarR (.ref .real [B] "p") (.castFloat .real .real (.const alpha)))
```
</details>

## Public theorem: `masked_add_typed_commuted_equiv`

**Statement:**
```lean
specification masked_add_typed_commuted_equiv (gt pt : FloatStorage) (mt : MaskStorage)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) (R : FP.ScalarArithmetic.Rules) :
    typedOriginal gt pt mt g p m n a B ≡[R] typedCommuted gt pt mt g p m n a B
```

**Closed-form spec defs (transitive):** `typedOriginal`, `typedCommuted`, `typed_integer_kernel`, `typedAdditionSite`, `original`, `commuted`, `storedOperand`, `integer_kernel`, `additionSite`

<details><summary><code>typedOriginal</code></summary>

```lean
def typedOriginal (gt pt : FloatStorage) (mt : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) : Program where
  kernel := typed_integer_kernel gt pt mt Bool.false g p m n a B
  domain := [typedAdditionSite gt pt mt g p m n a B]
```
</details>

<details><summary><code>typedCommuted</code></summary>

```lean
def typedCommuted (gt pt : FloatStorage) (mt : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) : Program :=
  { typedOriginal gt pt mt g p m n a B with
    kernel := typed_integer_kernel gt pt mt Bool.true g p m n a B }
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

<details><summary><code>typedAdditionSite</code></summary>

```lean
def typedAdditionSite (gt pt : FloatStorage) (mt : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) : Site :=
  .addFP32 (Source.prefixBefore
    (typed_integer_kernel gt pt mt Bool.false g p m n a B).surfaceBody "grad" 1) [B]
    (storedOperand gt "grad" B)
    (.mul .real .scalarR (storedOperand pt "p" B) (.castFloat .real .real (.const a)))
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program where
  kernel := integer_kernel storage Bool.false grad p mask n alpha B
  domain := [additionSite storage grad p mask n alpha B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program :=
  { original storage grad p mask n alpha B with
    kernel := integer_kernel storage Bool.true grad p mask n alpha B }
```
</details>

<details><summary><code>storedOperand</code></summary>

```lean
def storedOperand (storage : FloatStorage) (name : RegName) (B : Nat) : Op .real [B] :=
  match storage with
  | .fp32 => .ref .real [B] name
  | .fp16 => .castFloat .fp16 .real (.ref .fp16 [B] name)
  | .bf16 => .castFloat .bf16 .real (.ref .bf16 [B] name)
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

<details><summary><code>additionSite</code></summary>

```lean
def additionSite (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Site :=
  .addFP32 (Source.prefixBefore
    (integer_kernel storage Bool.false grad p mask n alpha B).surfaceBody "grad" 1) [B]
    (.ref .real [B] "grad")
    (.mul .real .scalarR (.ref .real [B] "p") (.castFloat .real .real (.const alpha)))
```
</details>

## Public theorem: `masked_add_bool_commuted_equiv`

**Statement:**
```lean
specification masked_add_bool_commuted_equiv (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) (R : FP.ScalarArithmetic.Rules) :
    boolOriginal grad p mask n alpha B ≡[R] boolCommuted grad p mask n alpha B
```

**Closed-form spec defs (transitive):** `boolOriginal`, `boolCommuted`, `masked_add_kernel`, `masked_add_commuted_kernel`, `original`, `commuted`, `integer_kernel`, `additionSite`

<details><summary><code>boolOriginal</code></summary>

```
/-- Legacy bool-buffer specialization, with the same fp32 arithmetic guard. -/
```
```lean
def boolOriginal (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) : Program where
  kernel := masked_add_kernel grad p mask n alpha B
  domain := [.add (Source.prefixBefore
    (masked_add_kernel grad p mask n alpha B).surfaceBody "grad" 1) [B]
    (.ref .real [B] "grad")
    (.mul .real .scalarR (.ref .real [B] "p") (.const alpha))]
```
</details>

<details><summary><code>boolCommuted</code></summary>

```lean
def boolCommuted (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) : Program :=
  { boolOriginal grad p mask n alpha B with
    kernel := masked_add_commuted_kernel grad p mask n alpha B }
```
</details>

<details><summary><code>masked_add_kernel</code></summary>

```
/-- Faithful transcription of `masked_add_cuda.py`'s `masked_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def masked_add_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad += p * $(alpha)
  tl.store(grad_ptr + offsets, grad, mask=mask)
}
```
</details>

<details><summary><code>masked_add_commuted_kernel</code></summary>

```
/-- Commuted addition for the preserved boolean-buffer specialization. -/
```
```lean
def masked_add_commuted_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad = p * $(alpha) + grad
  tl.store(grad_ptr + offsets, grad, mask=mask)
}


/-- Tested Python integer mask specializations. Both use signed integer memory. -/
inductive MaskStorage where
  | int32 | int64
  deriving DecidableEq, Repr

open Lean
set_option hygiene false in
macro "maskedAddSource%[" gradType:tritonDType "," pType:tritonDType "," maskType:tritonDType "," candidate:ident "]" : term => do
  let update ← if candidate.getId == `original then
    `(tritonStmt| grad += p * $((alpha : ℝ)).to(tl.float32))
  else if candidate.getId == `commuted then
    `(tritonStmt| grad = p * $((alpha : ℝ)).to(tl.float32) + grad)
  else Macro.throwError "expected original or commuted"
  let store ← match gradType with
    | `(tritonDType| tl.float32) | `(tritonDType| tl.float64) =>
        `(tritonStmt| tl.store(grad_ptr + offsets, grad, mask=mask))
    | _ => `(tritonStmt| tl.store(grad_ptr + offsets, grad.to($gradType:tritonDType), mask=mask))
  `(fun (grad_ptr p_ptr p_mask_ptr : RegionName) (n_elements : Nat) (alpha : ℝ)
      (BLOCK_SIZE : Nat) => triton {
    tl.region grad_ptr = $gradType, p_ptr = $pType, p_mask_ptr = $maskType
    pid = tl.program_id(axis=0)
    block_start = pid * $(BLOCK_SIZE)
    offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
    mask = offsets < $(n_elements)
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    $update:tritonStmt
    $store:tritonStmt
  })
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program where
  kernel := integer_kernel storage Bool.false grad p mask n alpha B
  domain := [additionSite storage grad p mask n alpha B]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program :=
  { original storage grad p mask n alpha B with
    kernel := integer_kernel storage Bool.true grad p mask n alpha B }
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

<details><summary><code>additionSite</code></summary>

```lean
def additionSite (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Site :=
  .addFP32 (Source.prefixBefore
    (integer_kernel storage Bool.false grad p mask n alpha B).surfaceBody "grad" 1) [B]
    (.ref .real [B] "grad")
    (.mul .real .scalarR (.ref .real [B] "p") (.castFloat .real .real (.const alpha)))
```
</details>
