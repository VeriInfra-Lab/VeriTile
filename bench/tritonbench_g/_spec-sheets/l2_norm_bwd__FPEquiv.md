# Spec sheet — `bench/tritonbench_g/l2_norm_bwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/l2_norm_bwd/l2_norm_bwd.py`

## Public theorem: `source_equivalence`

**Statement:**
```lean
specification source_equivalence (storage gradient : Storage) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    program storage gradient Bool.false X DY DX stride N eps B ≡[assumptions]
      program storage gradient Bool.true X DY DX stride N eps B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedBwd`

<details><summary><code>program</code></summary>

```lean
def program (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typedBwd storage gradient v X DY DX stride N eps B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typedBwd</code></summary>

```lean
def typedBwd (storage gradient : Storage) (commuted : Bool)
    (X DY DX : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, gradient, commuted with
  | .fp32, .fp32, Bool.false =>
    l2Backward%[tl.float32, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp32, Bool.true =>
    l2Backward%[tl.float32, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp16, Bool.false =>
    l2Backward%[tl.float32, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp16, Bool.true =>
    l2Backward%[tl.float32, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .bf16, Bool.false =>
    l2Backward%[tl.float32, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .bf16, Bool.true =>
    l2Backward%[tl.float32, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp64, Bool.false =>
    l2Backward%[tl.float32, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp64, Bool.true =>
    l2Backward%[tl.float32, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp32, Bool.false =>
    l2Backward%[tl.float16, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp32, Bool.true =>
    l2Backward%[tl.float16, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp16, Bool.false =>
    l2Backward%[tl.float16, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp16, Bool.true =>
    l2Backward%[tl.float16, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .bf16, Bool.false =>
    l2Backward%[tl.float16, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .bf16, Bool.true =>
    l2Backward%[tl.float16, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp64, Bool.false =>
    l2Backward%[tl.float16, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp64, Bool.true =>
    l2Backward%[tl.float16, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp32, Bool.false =>
    l2Backward%[tl.bfloat16, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp32, Bool.true =>
    l2Backward%[tl.bfloat16, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp16, Bool.false =>
    l2Backward%[tl.bfloat16, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp16, Bool.true =>
    l2Backward%[tl.bfloat16, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .bf16, Bool.false =>
    l2Backward%[tl.bfloat16, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .bf16, Bool.true =>
    l2Backward%[tl.bfloat16, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp64, Bool.false =>
    l2Backward%[tl.bfloat16, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp64, Bool.true =>
    l2Backward%[tl.bfloat16, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp32, Bool.false =>
    l2Backward%[tl.float64, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp32, Bool.true =>
    l2Backward%[tl.float64, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp16, Bool.false =>
    l2Backward%[tl.float64, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp16, Bool.true =>
    l2Backward%[tl.float64, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .bf16, Bool.false =>
    l2Backward%[tl.float64, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .bf16, Bool.true =>
    l2Backward%[tl.float64, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp64, Bool.false =>
    l2Backward%[tl.float64, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp64, Bool.true =>
    l2Backward%[tl.float64, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N


#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>
