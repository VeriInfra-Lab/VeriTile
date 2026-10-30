# Spec sheet — `bench/tritonbench_g/flash_attn/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/flash_attn/flash_attn.py`

## Public theorem: `source_equivalence`

**Statement:**
```lean
specification source_equivalence (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) :
    original input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal ≡[assumptions]
    indexCommuted input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`, `kernel`, `kernel_fp16_fp16`, `kernel_fp16_bf16`, `kernel_fp16_fp32`, `kernel_bf16_fp16`, `kernel_bf16_bf16`, `kernel_bf16_fp32`, `kernel_fp32_fp16`, `kernel_fp32_bf16`, `kernel_fp32_fp32`, `kernel_fp16_fp64`, `kernel_fp16_f8e4`, `kernel_fp16_f8e5`, `kernel_bf16_fp64`, `kernel_bf16_f8e4`, `kernel_bf16_f8e5`, `kernel_fp32_fp64`, `kernel_fp32_f8e4`, `kernel_fp32_f8e5`, `kernel_fp64_fp16`, `kernel_fp64_bf16`, `kernel_fp64_fp32`, `kernel_fp64_fp64`, `kernel_fp64_f8e4`, `kernel_fp64_f8e5`, `kernel_f8e4_fp16`, `kernel_f8e4_bf16`, `kernel_f8e4_fp32`, `kernel_f8e4_fp64`, `kernel_f8e4_f8e4`, `kernel_f8e4_f8e5`, `kernel_f8e5_fp16`, `kernel_f8e5_bf16`, `kernel_f8e5_fp32`, `kernel_f8e5_fp64`, `kernel_f8e5_f8e4`, `kernel_f8e5_f8e5`

<details><summary><code>original</code></summary>

```lean
def original (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) : Program where
  kernel := kernel input output Bool.false Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) : Program where
  kernel := kernel input output Bool.true Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal
  domain := []
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (input output : Storage) (swapped : Bool) := match input,output with
  | .fp16,.fp16 => kernel_fp16_fp16 swapped
  | .fp16,.bf16 => kernel_fp16_bf16 swapped
  | .fp16,.fp32 => kernel_fp16_fp32 swapped
  | .bf16,.fp16 => kernel_bf16_fp16 swapped
  | .bf16,.bf16 => kernel_bf16_bf16 swapped
  | .bf16,.fp32 => kernel_bf16_fp32 swapped
  | .fp32,.fp16 => kernel_fp32_fp16 swapped
  | .fp32,.bf16 => kernel_fp32_bf16 swapped
  | .fp32,.fp32 => kernel_fp32_fp32 swapped
  | .fp16,.fp64 => kernel_fp16_fp64 swapped
  | .fp16,.f8e4 => kernel_fp16_f8e4 swapped
  | .fp16,.f8e5 => kernel_fp16_f8e5 swapped
  | .bf16,.fp64 => kernel_bf16_fp64 swapped
  | .bf16,.f8e4 => kernel_bf16_f8e4 swapped
  | .bf16,.f8e5 => kernel_bf16_f8e5 swapped
  | .fp32,.fp64 => kernel_fp32_fp64 swapped
  | .fp32,.f8e4 => kernel_fp32_f8e4 swapped
  | .fp32,.f8e5 => kernel_fp32_f8e5 swapped
  | .fp64,.fp16 => kernel_fp64_fp16 swapped
  | .fp64,.bf16 => kernel_fp64_bf16 swapped
  | .fp64,.fp32 => kernel_fp64_fp32 swapped
  | .fp64,.fp64 => kernel_fp64_fp64 swapped
  | .fp64,.f8e4 => kernel_fp64_f8e4 swapped
  | .fp64,.f8e5 => kernel_fp64_f8e5 swapped
  | .f8e4,.fp16 => kernel_f8e4_fp16 swapped
  | .f8e4,.bf16 => kernel_f8e4_bf16 swapped
  | .f8e4,.fp32 => kernel_f8e4_fp32 swapped
  | .f8e4,.fp64 => kernel_f8e4_fp64 swapped
  | .f8e4,.f8e4 => kernel_f8e4_f8e4 swapped
  | .f8e4,.f8e5 => kernel_f8e4_f8e5 swapped
  | .f8e5,.fp16 => kernel_f8e5_fp16 swapped
  | .f8e5,.bf16 => kernel_f8e5_bf16 swapped
  | .f8e5,.fp32 => kernel_f8e5_fp32 swapped
  | .f8e5,.fp64 => kernel_f8e5_fp64 swapped
  | .f8e5,.f8e4 => kernel_f8e5_f8e4 swapped
  | .f8e5,.f8e5 => kernel_f8e5_f8e5 swapped
```
</details>

<details><summary><code>kernel_fp16_fp16</code></summary>

```lean
abbrev kernel_fp16_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float16,indexCommuted] else flashAttnSource%[tl.float16,tl.float16,original]
```
</details>

<details><summary><code>kernel_fp16_bf16</code></summary>

```lean
abbrev kernel_fp16_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.float16,tl.bfloat16,original]
```
</details>

<details><summary><code>kernel_fp16_fp32</code></summary>

```lean
abbrev kernel_fp16_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float32,indexCommuted] else flashAttnSource%[tl.float16,tl.float32,original]
```
</details>

<details><summary><code>kernel_bf16_fp16</code></summary>

```lean
abbrev kernel_bf16_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float16,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.float16,original]
```
</details>

<details><summary><code>kernel_bf16_bf16</code></summary>

```lean
abbrev kernel_bf16_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.bfloat16,original]
```
</details>

<details><summary><code>kernel_bf16_fp32</code></summary>

```lean
abbrev kernel_bf16_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float32,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.float32,original]
```
</details>

<details><summary><code>kernel_fp32_fp16</code></summary>

```lean
abbrev kernel_fp32_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float16,indexCommuted] else flashAttnSource%[tl.float32,tl.float16,original]
```
</details>

<details><summary><code>kernel_fp32_bf16</code></summary>

```lean
abbrev kernel_fp32_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.float32,tl.bfloat16,original]
```
</details>

<details><summary><code>kernel_fp32_fp32</code></summary>

```lean
abbrev kernel_fp32_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float32,indexCommuted] else flashAttnSource%[tl.float32,tl.float32,original]
```
</details>

<details><summary><code>kernel_fp16_fp64</code></summary>

```lean
abbrev kernel_fp16_fp64 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float64,indexCommuted] else flashAttnSource%[tl.float16,tl.float64,original]
```
</details>

<details><summary><code>kernel_fp16_f8e4</code></summary>

```lean
abbrev kernel_fp16_f8e4 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float8e4nv,indexCommuted] else flashAttnSource%[tl.float16,tl.float8e4nv,original]
```
</details>

<details><summary><code>kernel_fp16_f8e5</code></summary>

```lean
abbrev kernel_fp16_f8e5 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float8e5,indexCommuted] else flashAttnSource%[tl.float16,tl.float8e5,original]
```
</details>

<details><summary><code>kernel_bf16_fp64</code></summary>

```lean
abbrev kernel_bf16_fp64 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float64,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.float64,original]
```
</details>

<details><summary><code>kernel_bf16_f8e4</code></summary>

```lean
abbrev kernel_bf16_f8e4 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float8e4nv,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.float8e4nv,original]
```
</details>

<details><summary><code>kernel_bf16_f8e5</code></summary>

```lean
abbrev kernel_bf16_f8e5 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float8e5,indexCommuted] else flashAttnSource%[tl.bfloat16,tl.float8e5,original]
```
</details>

<details><summary><code>kernel_fp32_fp64</code></summary>

```lean
abbrev kernel_fp32_fp64 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float64,indexCommuted] else flashAttnSource%[tl.float32,tl.float64,original]
```
</details>

<details><summary><code>kernel_fp32_f8e4</code></summary>

```lean
abbrev kernel_fp32_f8e4 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float8e4nv,indexCommuted] else flashAttnSource%[tl.float32,tl.float8e4nv,original]
```
</details>

<details><summary><code>kernel_fp32_f8e5</code></summary>

```lean
abbrev kernel_fp32_f8e5 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float8e5,indexCommuted] else flashAttnSource%[tl.float32,tl.float8e5,original]
```
</details>

<details><summary><code>kernel_fp64_fp16</code></summary>

```lean
abbrev kernel_fp64_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float64,tl.float16,indexCommuted] else flashAttnSource%[tl.float64,tl.float16,original]
```
</details>

<details><summary><code>kernel_fp64_bf16</code></summary>

```lean
abbrev kernel_fp64_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float64,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.float64,tl.bfloat16,original]
```
</details>

<details><summary><code>kernel_fp64_fp32</code></summary>

```lean
abbrev kernel_fp64_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float64,tl.float32,indexCommuted] else flashAttnSource%[tl.float64,tl.float32,original]
```
</details>

<details><summary><code>kernel_fp64_fp64</code></summary>

```lean
abbrev kernel_fp64_fp64 (swapped : Bool) := if swapped then flashAttnSource%[tl.float64,tl.float64,indexCommuted] else flashAttnSource%[tl.float64,tl.float64,original]
```
</details>

<details><summary><code>kernel_fp64_f8e4</code></summary>

```lean
abbrev kernel_fp64_f8e4 (swapped : Bool) := if swapped then flashAttnSource%[tl.float64,tl.float8e4nv,indexCommuted] else flashAttnSource%[tl.float64,tl.float8e4nv,original]
```
</details>

<details><summary><code>kernel_fp64_f8e5</code></summary>

```lean
abbrev kernel_fp64_f8e5 (swapped : Bool) := if swapped then flashAttnSource%[tl.float64,tl.float8e5,indexCommuted] else flashAttnSource%[tl.float64,tl.float8e5,original]
```
</details>

<details><summary><code>kernel_f8e4_fp16</code></summary>

```lean
abbrev kernel_f8e4_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e4nv,tl.float16,indexCommuted] else flashAttnSource%[tl.float8e4nv,tl.float16,original]
```
</details>

<details><summary><code>kernel_f8e4_bf16</code></summary>

```lean
abbrev kernel_f8e4_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e4nv,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.float8e4nv,tl.bfloat16,original]
```
</details>

<details><summary><code>kernel_f8e4_fp32</code></summary>

```lean
abbrev kernel_f8e4_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e4nv,tl.float32,indexCommuted] else flashAttnSource%[tl.float8e4nv,tl.float32,original]
```
</details>

<details><summary><code>kernel_f8e4_fp64</code></summary>

```lean
abbrev kernel_f8e4_fp64 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e4nv,tl.float64,indexCommuted] else flashAttnSource%[tl.float8e4nv,tl.float64,original]
```
</details>

<details><summary><code>kernel_f8e4_f8e4</code></summary>

```lean
abbrev kernel_f8e4_f8e4 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e4nv,tl.float8e4nv,indexCommuted] else flashAttnSource%[tl.float8e4nv,tl.float8e4nv,original]
```
</details>

<details><summary><code>kernel_f8e4_f8e5</code></summary>

```lean
abbrev kernel_f8e4_f8e5 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e4nv,tl.float8e5,indexCommuted] else flashAttnSource%[tl.float8e4nv,tl.float8e5,original]
```
</details>

<details><summary><code>kernel_f8e5_fp16</code></summary>

```lean
abbrev kernel_f8e5_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e5,tl.float16,indexCommuted] else flashAttnSource%[tl.float8e5,tl.float16,original]
```
</details>

<details><summary><code>kernel_f8e5_bf16</code></summary>

```lean
abbrev kernel_f8e5_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e5,tl.bfloat16,indexCommuted] else flashAttnSource%[tl.float8e5,tl.bfloat16,original]
```
</details>

<details><summary><code>kernel_f8e5_fp32</code></summary>

```lean
abbrev kernel_f8e5_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e5,tl.float32,indexCommuted] else flashAttnSource%[tl.float8e5,tl.float32,original]
```
</details>

<details><summary><code>kernel_f8e5_fp64</code></summary>

```lean
abbrev kernel_f8e5_fp64 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e5,tl.float64,indexCommuted] else flashAttnSource%[tl.float8e5,tl.float64,original]
```
</details>

<details><summary><code>kernel_f8e5_f8e4</code></summary>

```lean
abbrev kernel_f8e5_f8e4 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e5,tl.float8e4nv,indexCommuted] else flashAttnSource%[tl.float8e5,tl.float8e4nv,original]
```
</details>

<details><summary><code>kernel_f8e5_f8e5</code></summary>

```lean
abbrev kernel_f8e5_f8e5 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e5,tl.float8e5,indexCommuted] else flashAttnSource%[tl.float8e5,tl.float8e5,original]
```
</details>

## Public theorem: `actual_source_equivalence`

<details><summary>docstring</summary>

```
/-- Complete original/candidate equivalence for the unmodified source with
fp16 block output, six floating Q labels and both causal modes. No cast is
added to either source store. Native wrapper Q=O requires fp16 Q as well. -/
```
</details>

**Statement:**
```lean
specification actual_source_equivalence (input : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) :
    ActualFPProof.program Bool.false input Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal ≡[assumptions]
    ActualFPProof.program Bool.true input Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal
```

**Closed-form spec defs (transitive):** `assumptions`

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>
