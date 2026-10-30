# Spec sheet — `bench/tritonbench_g/rms_norm_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/rms_norm_triton/rms_norm_triton.py`

## Public theorem: `source_equivalence`

**Statement:**
```lean
specification source_equivalence (storage weight : Storage) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    program storage weight Bool.false Y X W ysr ysc xsr xsc N eps B ≡[assumptions]
      program storage weight Bool.true Y X W ysr ysc xsr xsc N eps B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typed`

<details><summary><code>program</code></summary>

```lean
def program (storage weight : Storage) (v : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) : Program where
  kernel := typed storage weight v Y X W ysr ysc xsr xsc N eps B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typed</code></summary>

```lean
def typed (storage weight : Storage) (commuted : Bool)
    (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, weight, commuted with
  | .fp32, .fp32, Bool.false =>
    rmsTyped%[tl.float32, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    rmsTyped%[tl.float32, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    rmsTyped%[tl.float32, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    rmsTyped%[tl.float32, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    rmsTyped%[tl.float32, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    rmsTyped%[tl.float32, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    rmsTyped%[tl.float32, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    rmsTyped%[tl.float32, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    rmsTyped%[tl.float16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    rmsTyped%[tl.float16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    rmsTyped%[tl.float16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    rmsTyped%[tl.float16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    rmsTyped%[tl.float16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    rmsTyped%[tl.float16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    rmsTyped%[tl.float16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    rmsTyped%[tl.float16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    rmsTyped%[tl.float64, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    rmsTyped%[tl.float64, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    rmsTyped%[tl.float64, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    rmsTyped%[tl.float64, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    rmsTyped%[tl.float64, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    rmsTyped%[tl.float64, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    rmsTyped%[tl.float64, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    rmsTyped%[tl.float64, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>
