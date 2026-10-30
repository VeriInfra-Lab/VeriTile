# Spec sheet — `bench/tritonbench_g/l2_norm_triton1/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/l2_norm_triton1/l2_norm_triton1.py`

## Public theorem: `source_equivalence`

**Statement:**
```lean
specification source_equivalence (storage : Storage) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    program storage Bool.false X Y stride N eps B ≡[assumptions]
      program storage Bool.true X Y stride N eps B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`

<details><summary><code>program</code></summary>

```lean
def program (storage : Storage) (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typedKernel storage commuted X Y stride N eps B
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
def typedKernel (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, commuted with
  | .fp32, Bool.false => l2Source%[tl.float32, original] X Y stride_x_row N eps BLOCK_N
  | .fp32, Bool.true => l2Source%[tl.float32, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.false => l2Source%[tl.float16, original] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.true => l2Source%[tl.float16, commuted] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.false => l2Source%[tl.bfloat16, original] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.true => l2Source%[tl.bfloat16, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.false => l2Source%[tl.float64, original] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.true => l2Source%[tl.float64, commuted] X Y stride_x_row N eps BLOCK_N

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>
