# Spec sheet — `bench/tritonbench_g/softmax_flaggems/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/softmax_flaggems/softmax_flaggems.py`

## Public theorem: `forward_non_inner_equiv`

**Statement:**
```lean
specification forward_non_inner_equiv (inputType outputType : Storage)
    (output_ptr input_ptr : RegionName) (M N K TN TK : Nat) (oneTile : Bool) :
    program (softmax_kernel_non_inner inputType outputType
      output_ptr input_ptr M N K TN TK oneTile) ≡[assumptions]
    program (softmax_kernel_non_inner_candidate inputType outputType
      output_ptr input_ptr M N K TN TK oneTile)
```

**Closed-form spec defs (transitive):** `program`, `softmax_kernel_non_inner`, `assumptions`, `softmax_kernel_non_inner_candidate`

<details><summary><code>program</code></summary>

```lean
def program (kernel : ComputeKernel) : Program := ⟨kernel, []⟩
```
</details>

<details><summary><code>softmax_kernel_non_inner</code></summary>

```lean
def softmax_kernel_non_inner (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems0%[tl.float16, tl.float16, original]
  | .fp16, .bf16 => flaggems0%[tl.float16, tl.bfloat16, original]
  | .fp16, .fp32 => flaggems0%[tl.float16, tl.float32, original]
  | .fp16, .fp64 => flaggems0%[tl.float16, tl.float64, original]
  | .bf16, .fp16 => flaggems0%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 => flaggems0%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32 => flaggems0%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp64 => flaggems0%[tl.bfloat16, tl.float64, original]
  | .fp32, .fp16 => flaggems0%[tl.float32, tl.float16, original]
  | .fp32, .bf16 => flaggems0%[tl.float32, tl.bfloat16, original]
  | .fp32, .fp32 => flaggems0%[tl.float32, tl.float32, original]
  | .fp32, .fp64 => flaggems0%[tl.float32, tl.float64, original]
  | .fp64, .fp16 => flaggems0%[tl.float64, tl.float16, original]
  | .fp64, .bf16 => flaggems0%[tl.float64, tl.bfloat16, original]
  | .fp64, .fp32 => flaggems0%[tl.float64, tl.float32, original]
  | .fp64, .fp64 => flaggems0%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>softmax_kernel_non_inner_candidate</code></summary>

```lean
def softmax_kernel_non_inner_candidate (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems0%[tl.float16, tl.float16, address_commuted]
  | .fp16, .bf16 => flaggems0%[tl.float16, tl.bfloat16, address_commuted]
  | .fp16, .fp32 => flaggems0%[tl.float16, tl.float32, address_commuted]
  | .fp16, .fp64 => flaggems0%[tl.float16, tl.float64, address_commuted]
  | .bf16, .fp16 => flaggems0%[tl.bfloat16, tl.float16, address_commuted]
  | .bf16, .bf16 => flaggems0%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .bf16, .fp32 => flaggems0%[tl.bfloat16, tl.float32, address_commuted]
  | .bf16, .fp64 => flaggems0%[tl.bfloat16, tl.float64, address_commuted]
  | .fp32, .fp16 => flaggems0%[tl.float32, tl.float16, address_commuted]
  | .fp32, .bf16 => flaggems0%[tl.float32, tl.bfloat16, address_commuted]
  | .fp32, .fp32 => flaggems0%[tl.float32, tl.float32, address_commuted]
  | .fp32, .fp64 => flaggems0%[tl.float32, tl.float64, address_commuted]
  | .fp64, .fp16 => flaggems0%[tl.float64, tl.float16, address_commuted]
  | .fp64, .bf16 => flaggems0%[tl.float64, tl.bfloat16, address_commuted]
  | .fp64, .fp32 => flaggems0%[tl.float64, tl.float32, address_commuted]
  | .fp64, .fp64 => flaggems0%[tl.float64, tl.float64, address_commuted]
```
</details>

## Public theorem: `forward_inner_equiv`

**Statement:**
```lean
specification forward_inner_equiv (inputType outputType : Storage)
    (output_ptr input_ptr : RegionName) (M N TN : Nat) (oneTile : Bool) :
    program (softmax_kernel_inner inputType outputType
      output_ptr input_ptr M N TN oneTile) ≡[assumptions]
    program (softmax_kernel_inner_candidate inputType outputType
      output_ptr input_ptr M N TN oneTile)
```

**Closed-form spec defs (transitive):** `program`, `softmax_kernel_inner`, `assumptions`, `softmax_kernel_inner_candidate`

<details><summary><code>program</code></summary>

```lean
def program (kernel : ComputeKernel) : Program := ⟨kernel, []⟩
```
</details>

<details><summary><code>softmax_kernel_inner</code></summary>

```lean
def softmax_kernel_inner (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems1%[tl.float16, tl.float16, original]
  | .fp16, .bf16 => flaggems1%[tl.float16, tl.bfloat16, original]
  | .fp16, .fp32 => flaggems1%[tl.float16, tl.float32, original]
  | .fp16, .fp64 => flaggems1%[tl.float16, tl.float64, original]
  | .bf16, .fp16 => flaggems1%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 => flaggems1%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32 => flaggems1%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp64 => flaggems1%[tl.bfloat16, tl.float64, original]
  | .fp32, .fp16 => flaggems1%[tl.float32, tl.float16, original]
  | .fp32, .bf16 => flaggems1%[tl.float32, tl.bfloat16, original]
  | .fp32, .fp32 => flaggems1%[tl.float32, tl.float32, original]
  | .fp32, .fp64 => flaggems1%[tl.float32, tl.float64, original]
  | .fp64, .fp16 => flaggems1%[tl.float64, tl.float16, original]
  | .fp64, .bf16 => flaggems1%[tl.float64, tl.bfloat16, original]
  | .fp64, .fp32 => flaggems1%[tl.float64, tl.float32, original]
  | .fp64, .fp64 => flaggems1%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>softmax_kernel_inner_candidate</code></summary>

```lean
def softmax_kernel_inner_candidate (inputType outputType : Storage) :=
  match inputType, outputType with
  | .fp16, .fp16 => flaggems1%[tl.float16, tl.float16, address_commuted]
  | .fp16, .bf16 => flaggems1%[tl.float16, tl.bfloat16, address_commuted]
  | .fp16, .fp32 => flaggems1%[tl.float16, tl.float32, address_commuted]
  | .fp16, .fp64 => flaggems1%[tl.float16, tl.float64, address_commuted]
  | .bf16, .fp16 => flaggems1%[tl.bfloat16, tl.float16, address_commuted]
  | .bf16, .bf16 => flaggems1%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .bf16, .fp32 => flaggems1%[tl.bfloat16, tl.float32, address_commuted]
  | .bf16, .fp64 => flaggems1%[tl.bfloat16, tl.float64, address_commuted]
  | .fp32, .fp16 => flaggems1%[tl.float32, tl.float16, address_commuted]
  | .fp32, .bf16 => flaggems1%[tl.float32, tl.bfloat16, address_commuted]
  | .fp32, .fp32 => flaggems1%[tl.float32, tl.float32, address_commuted]
  | .fp32, .fp64 => flaggems1%[tl.float32, tl.float64, address_commuted]
  | .fp64, .fp16 => flaggems1%[tl.float64, tl.float16, address_commuted]
  | .fp64, .bf16 => flaggems1%[tl.float64, tl.bfloat16, address_commuted]
  | .fp64, .fp32 => flaggems1%[tl.float64, tl.float32, address_commuted]
  | .fp64, .fp64 => flaggems1%[tl.float64, tl.float64, address_commuted]
```
</details>

## Public theorem: `backward_non_inner_equiv`

**Statement:**
```lean
specification backward_non_inner_equiv (storage : Storage)
    (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N K TN TK : Nat) (oneTile : Bool) :
    program (softmax_backward_kernel_non_inner storage
      out_ptr out_grad_ptr in_grad_ptr M N K TN TK oneTile) ≡[assumptions]
    program (softmax_backward_kernel_non_inner_candidate storage
      out_ptr out_grad_ptr in_grad_ptr M N K TN TK oneTile)
```

**Closed-form spec defs (transitive):** `program`, `softmax_backward_kernel_non_inner`, `assumptions`, `softmax_backward_kernel_non_inner_candidate`

<details><summary><code>program</code></summary>

```lean
def program (kernel : ComputeKernel) : Program := ⟨kernel, []⟩
```
</details>

<details><summary><code>softmax_backward_kernel_non_inner</code></summary>

```lean
def softmax_backward_kernel_non_inner (storage : Storage) :=
  match storage with
  | .fp16 => flaggems2%[tl.float16, tl.float16, original]
  | .bf16 => flaggems2%[tl.bfloat16, tl.bfloat16, original]
  | .fp32 => flaggems2%[tl.float32, tl.float32, original]
  | .fp64 => flaggems2%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>softmax_backward_kernel_non_inner_candidate</code></summary>

```lean
def softmax_backward_kernel_non_inner_candidate (storage : Storage) :=
  match storage with
  | .fp16 => flaggems2%[tl.float16, tl.float16, address_commuted]
  | .bf16 => flaggems2%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .fp32 => flaggems2%[tl.float32, tl.float32, address_commuted]
  | .fp64 => flaggems2%[tl.float64, tl.float64, address_commuted]
```
</details>

## Public theorem: `backward_inner_equiv`

**Statement:**
```lean
specification backward_inner_equiv (storage : Storage)
    (out_ptr out_grad_ptr in_grad_ptr : RegionName) (M N TM TN : Nat) (oneTile : Bool) :
    program (softmax_backward_kernel_inner storage
      out_ptr out_grad_ptr in_grad_ptr M N TM TN oneTile) ≡[assumptions]
    program (softmax_backward_kernel_inner_candidate storage
      out_ptr out_grad_ptr in_grad_ptr M N TM TN oneTile)
```

**Closed-form spec defs (transitive):** `program`, `softmax_backward_kernel_inner`, `assumptions`, `softmax_backward_kernel_inner_candidate`

<details><summary><code>program</code></summary>

```lean
def program (kernel : ComputeKernel) : Program := ⟨kernel, []⟩
```
</details>

<details><summary><code>softmax_backward_kernel_inner</code></summary>

```lean
def softmax_backward_kernel_inner (storage : Storage) :=
  match storage with
  | .fp16 => flaggems3%[tl.float16, tl.float16, original]
  | .bf16 => flaggems3%[tl.bfloat16, tl.bfloat16, original]
  | .fp32 => flaggems3%[tl.float32, tl.float32, original]
  | .fp64 => flaggems3%[tl.float64, tl.float64, original]
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>softmax_backward_kernel_inner_candidate</code></summary>

```lean
def softmax_backward_kernel_inner_candidate (storage : Storage) :=
  match storage with
  | .fp16 => flaggems3%[tl.float16, tl.float16, address_commuted]
  | .bf16 => flaggems3%[tl.bfloat16, tl.bfloat16, address_commuted]
  | .fp32 => flaggems3%[tl.float32, tl.float32, address_commuted]
  | .fp64 => flaggems3%[tl.float64, tl.float64, address_commuted]
```
</details>
