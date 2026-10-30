# Spec sheet — `bench/tritonbench_g/l2_norm_triton2/Correct.lean`

**Python source:** `bench/tritonbench_g/l2_norm_triton2/l2_norm_triton2.py`

## Public theorem: `forward_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete source versions produce x / sqrt(sum(x²) + eps) in real
semantics, with successful flat execution, input/output bounds and a full frame. -/
```
</details>

**Statement:**
```lean
specification forward_typed_real_correct (storage : Storage) (v : Bool)
    (X Y : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat) (_hB : 0 < B) (_hN : N ≤ B) :
    VeriTile.Spec.Real (typedFwdIO storage v X Y stride N eps B ⊨
      fun xs i => l2FwdSpec N B eps xs i)
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`
- `_hN : N ≤ B`

**Closed-form spec defs (transitive):** `typedFwdIO`, `typedFwd`

<details><summary><code>typedFwdIO</code></summary>

```
/-- Complete typed forward source with the independently declared masked row IO. -/
```
```lean
def typedFwdIO (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₁ :=
  { l2FwdIO X Y stride N eps B with
    kernel := (typedFwd storage v X Y stride N eps B).eraseDType
    projection := fwd_supported storage v X Y stride N eps B }
```
</details>

<details><summary><code>typedFwd</code></summary>

```lean
def typedFwd (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, commuted with
  | .fp32, Bool.false => l2TwoFwd%[tl.float32, original] X Y stride_x_row N eps BLOCK_N
  | .fp32, Bool.true => l2TwoFwd%[tl.float32, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.false => l2TwoFwd%[tl.float16, original] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.true => l2TwoFwd%[tl.float16, commuted] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.false => l2TwoFwd%[tl.bfloat16, original] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.true => l2TwoFwd%[tl.bfloat16, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.false => l2TwoFwd%[tl.float64, original] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.true => l2TwoFwd%[tl.float64, commuted] X Y stride_x_row N eps BLOCK_N
```
</details>

## Public theorem: `backward_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete backward sources implement the independent L2 gradient formula
with both reductions, successful flat execution and every unwritten cell preserved. -/
```
</details>

**Statement:**
```lean
specification backward_typed_real_correct (storage gradient : Storage) (v : Bool)
    (X DY DX : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat) (_hB : 0 < B) (_hN : N ≤ B) :
    VeriTile.Spec.Real (typedBwdIO storage gradient v X DY DX stride N eps B ⊨
      fun xs dys i => l2BwdSpec N B eps xs dys i)
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`
- `_hN : N ≤ B`

**Closed-form spec defs (transitive):** `typedBwdIO`, `typedBwd`

<details><summary><code>typedBwdIO</code></summary>

```
/-- Complete typed backward source with separate X/DY input windows and DX output. -/
```
```lean
def typedBwdIO (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₂ :=
  { l2BwdIO X DY DX stride N eps B with
    kernel := (typedBwd storage gradient v X DY DX stride N eps B).eraseDType
    projection := bwd_supported storage gradient v X DY DX stride N eps B }
```
</details>

<details><summary><code>typedBwd</code></summary>

```lean
def typedBwd (storage gradient : Storage) (commuted : Bool)
    (X DY DX : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, gradient, commuted with
  | .fp32, .fp32, Bool.false =>
    l2TwoBwd%[tl.float32, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp32, Bool.true =>
    l2TwoBwd%[tl.float32, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp16, Bool.false =>
    l2TwoBwd%[tl.float32, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp16, Bool.true =>
    l2TwoBwd%[tl.float32, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .bf16, Bool.false =>
    l2TwoBwd%[tl.float32, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .bf16, Bool.true =>
    l2TwoBwd%[tl.float32, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp64, Bool.false =>
    l2TwoBwd%[tl.float32, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp64, Bool.true =>
    l2TwoBwd%[tl.float32, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp32, Bool.false =>
    l2TwoBwd%[tl.float16, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp32, Bool.true =>
    l2TwoBwd%[tl.float16, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp16, Bool.false =>
    l2TwoBwd%[tl.float16, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp16, Bool.true =>
    l2TwoBwd%[tl.float16, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .bf16, Bool.false =>
    l2TwoBwd%[tl.float16, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .bf16, Bool.true =>
    l2TwoBwd%[tl.float16, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp64, Bool.false =>
    l2TwoBwd%[tl.float16, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp64, Bool.true =>
    l2TwoBwd%[tl.float16, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp32, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp32, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp16, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp16, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .bf16, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .bf16, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp64, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp64, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp32, Bool.false =>
    l2TwoBwd%[tl.float64, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp32, Bool.true =>
    l2TwoBwd%[tl.float64, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp16, Bool.false =>
    l2TwoBwd%[tl.float64, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp16, Bool.true =>
    l2TwoBwd%[tl.float64, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .bf16, Bool.false =>
    l2TwoBwd%[tl.float64, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .bf16, Bool.true =>
    l2TwoBwd%[tl.float64, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp64, Bool.false =>
    l2TwoBwd%[tl.float64, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp64, Bool.true =>
    l2TwoBwd%[tl.float64, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>
