# Spec sheet — `bench/tritonbench_g/geglu_tanh_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/geglu_tanh_triton/geglu_tanh_triton.py`

## Public theorem: `forward_index_equiv`

<details><summary>docstring</summary>

```
/-- Full forward state/failure equivalence. Only signed row-base products commute;
all floating operations and conversions remain in the source order. -/
```
</details>

**Statement:**
```lean
specification forward_index_equiv (storage partner : Storage)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    forward storage partner Bool.false a b c stride n_cols BLOCK_SIZE ≡[assumptions]
      forward storage partner Bool.true a b c stride n_cols BLOCK_SIZE
```

**Closed-form spec defs (transitive):** `forward`, `assumptions`, `typedFwd`

<details><summary><code>forward</code></summary>

```lean
def forward (storage partner : Storage) (commuted : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Program where
  kernel := typedFwd storage partner commuted a b c stride n_cols BLOCK_SIZE
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typedFwd</code></summary>

```
/-- Complete forward source family with independent input storage. -/
```
```lean
def typedFwd (storage partner : Storage) (commuted : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, partner, commuted with
  | .fp32, .fp32, Bool.false =>
    gegluForward%[tl.float32, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    gegluForward%[tl.float32, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    gegluForward%[tl.float32, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    gegluForward%[tl.float32, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    gegluForward%[tl.float32, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    gegluForward%[tl.float32, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    gegluForward%[tl.float32, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    gegluForward%[tl.float32, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e4, Bool.false =>
    gegluForward%[tl.float32, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e4, Bool.true =>
    gegluForward%[tl.float32, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e5, Bool.false =>
    gegluForward%[tl.float32, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e5, Bool.true =>
    gegluForward%[tl.float32, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    gegluForward%[tl.float16, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    gegluForward%[tl.float16, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    gegluForward%[tl.float16, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    gegluForward%[tl.float16, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    gegluForward%[tl.float16, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    gegluForward%[tl.float16, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    gegluForward%[tl.float16, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    gegluForward%[tl.float16, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e4, Bool.false =>
    gegluForward%[tl.float16, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e4, Bool.true =>
    gegluForward%[tl.float16, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e5, Bool.false =>
    gegluForward%[tl.float16, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e5, Bool.true =>
    gegluForward%[tl.float16, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    gegluForward%[tl.bfloat16, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    gegluForward%[tl.bfloat16, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e4, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e4, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e5, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e5, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    gegluForward%[tl.float64, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    gegluForward%[tl.float64, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    gegluForward%[tl.float64, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    gegluForward%[tl.float64, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    gegluForward%[tl.float64, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    gegluForward%[tl.float64, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    gegluForward%[tl.float64, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    gegluForward%[tl.float64, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e4, Bool.false =>
    gegluForward%[tl.float64, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e4, Bool.true =>
    gegluForward%[tl.float64, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e5, Bool.false =>
    gegluForward%[tl.float64, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e5, Bool.true =>
    gegluForward%[tl.float64, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp32, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp32, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp16, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp16, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .bf16, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .bf16, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp64, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp64, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp32, Bool.false =>
    gegluForward%[tl.float8e5, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp32, Bool.true =>
    gegluForward%[tl.float8e5, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp16, Bool.false =>
    gegluForward%[tl.float8e5, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp16, Bool.true =>
    gegluForward%[tl.float8e5, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .bf16, Bool.false =>
    gegluForward%[tl.float8e5, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .bf16, Bool.true =>
    gegluForward%[tl.float8e5, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp64, Bool.false =>
    gegluForward%[tl.float8e5, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp64, Bool.true =>
    gegluForward%[tl.float8e5, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4, Bool.false =>
    gegluForward%[tl.float8e5, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4, Bool.true =>
    gegluForward%[tl.float8e5, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5, Bool.false =>
    gegluForward%[tl.float8e5, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5, Bool.true =>
    gegluForward%[tl.float8e5, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
```
</details>

## Public theorem: `backward_index_equiv`

<details><summary>docstring</summary>

```
/-- Full backward state/failure equivalence, including both ordered in-place
stores and every input dtype combination, without a numerical-domain premise. -/
```
</details>

**Statement:**
```lean
specification backward_index_equiv (storage partner gradient : Storage)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    backward storage partner gradient Bool.false dc a b stride n_cols BLOCK_SIZE ≡[assumptions]
      backward storage partner gradient Bool.true dc a b stride n_cols BLOCK_SIZE
```

**Closed-form spec defs (transitive):** `backward`, `assumptions`, `typedBwd`, `typedBwd_fp32_fp32`, `typedBwd_fp32_fp16`, `typedBwd_fp32_bf16`, `typedBwd_fp32_fp64`, `typedBwd_fp32_f8e4`, `typedBwd_fp32_f8e5`, `typedBwd_fp16_fp32`, `typedBwd_fp16_fp16`, `typedBwd_fp16_bf16`, `typedBwd_fp16_fp64`, `typedBwd_fp16_f8e4`, `typedBwd_fp16_f8e5`, `typedBwd_bf16_fp32`, `typedBwd_bf16_fp16`, `typedBwd_bf16_bf16`, `typedBwd_bf16_fp64`, `typedBwd_bf16_f8e4`, `typedBwd_bf16_f8e5`, `typedBwd_fp64_fp32`, `typedBwd_fp64_fp16`, `typedBwd_fp64_bf16`, `typedBwd_fp64_fp64`, `typedBwd_fp64_f8e4`, `typedBwd_fp64_f8e5`, `typedBwd_f8e4_fp32`, `typedBwd_f8e4_fp16`, `typedBwd_f8e4_bf16`, `typedBwd_f8e4_fp64`, `typedBwd_f8e4_f8e4`, `typedBwd_f8e4_f8e5`, `typedBwd_f8e5_fp32`, `typedBwd_f8e5_fp16`, `typedBwd_f8e5_bf16`, `typedBwd_f8e5_fp64`, `typedBwd_f8e5_f8e4`, `typedBwd_f8e5_f8e5`

<details><summary><code>backward</code></summary>

```lean
def backward (storage partner gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : Program where
  kernel := typedBwd storage partner gradient commuted dc a b stride n_cols BLOCK_SIZE
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typedBwd</code></summary>

```
/-- Complete in-place backward source family with independent input storage. -/
```
```lean
def typedBwd (storage partner gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, partner with
  | .fp32, .fp32 => typedBwd_fp32_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .fp16 => typedBwd_fp32_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .bf16 => typedBwd_fp32_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .fp64 => typedBwd_fp32_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .f8e4 => typedBwd_fp32_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .f8e5 => typedBwd_fp32_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .fp32 => typedBwd_fp16_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .fp16 => typedBwd_fp16_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .bf16 => typedBwd_fp16_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .fp64 => typedBwd_fp16_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .f8e4 => typedBwd_fp16_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .f8e5 => typedBwd_fp16_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .fp32 => typedBwd_bf16_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .fp16 => typedBwd_bf16_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .bf16 => typedBwd_bf16_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .fp64 => typedBwd_bf16_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .f8e4 => typedBwd_bf16_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .f8e5 => typedBwd_bf16_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .fp32 => typedBwd_fp64_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .fp16 => typedBwd_fp64_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .bf16 => typedBwd_fp64_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .fp64 => typedBwd_fp64_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .f8e4 => typedBwd_fp64_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .f8e5 => typedBwd_fp64_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .fp32 => typedBwd_f8e4_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .fp16 => typedBwd_f8e4_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .bf16 => typedBwd_f8e4_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .fp64 => typedBwd_f8e4_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4 => typedBwd_f8e4_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5 => typedBwd_f8e4_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .fp32 => typedBwd_f8e5_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .fp16 => typedBwd_f8e5_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .bf16 => typedBwd_f8e5_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .fp64 => typedBwd_f8e5_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4 => typedBwd_f8e5_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5 => typedBwd_f8e5_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>typedBwd_fp32_fp32</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp32_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp32_fp16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp32_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp32_bf16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp32_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp32_fp64</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp32_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp32_f8e4</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp32_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp32_f8e5</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp32_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp16_fp32</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp16_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp16_fp16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp16_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp16_bf16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp16_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp16_fp64</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp16_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp16_f8e4</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp16_f8e5</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_bf16_fp32</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_bf16_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_bf16_fp16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_bf16_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_bf16_bf16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_bf16_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_bf16_fp64</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_bf16_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_bf16_f8e4</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_bf16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_bf16_f8e5</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_bf16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp64_fp32</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp64_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp64_fp16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp64_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp64_bf16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp64_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp64_fp64</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp64_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp64_f8e4</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp64_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_fp64_f8e5</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_fp64_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e4_fp32</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e4_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e4_fp16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e4_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e4_bf16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e4_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e4_fp64</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e4_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e4_f8e4</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e4_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e4_f8e5</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e4_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e5_fp32</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e5_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e5_fp16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e5_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e5_bf16</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e5_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e5_fp64</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e5_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e5_f8e4</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e5_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>

<details><summary><code>typedBwd_f8e5_f8e5</code></summary>

```
/-- Source specialization with fixed A/B storage and independent gradient storage. -/
```
```lean
def typedBwd_f8e5_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE
```
</details>
