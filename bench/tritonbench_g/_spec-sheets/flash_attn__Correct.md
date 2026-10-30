# Spec sheet — `bench/tritonbench_g/flash_attn/Correct.lean`

**Python source:** `bench/tritonbench_g/flash_attn/flash_attn.py`

## Public theorem: `original_correctness`

**Statement:**
```lean
specification original_correctness (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (s : BlockState) (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.RegionIO (kernel input output Bool.false Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel p s Q K V L O)
```

**Assumptions / layout contracts:**
- `hBN : 0 < p.BLOCK_N`
- `hhi : 0 < hi p (s.pids 0)`

**Closed-form spec defs (transitive):** `kernel`, `kernel_fp16_fp16`, `kernel_fp16_bf16`, `kernel_fp16_fp32`, `kernel_bf16_fp16`, `kernel_bf16_bf16`, `kernel_bf16_fp32`, `kernel_fp32_fp16`, `kernel_fp32_bf16`, `kernel_fp32_fp32`, `kernel_fp16_fp64`, `kernel_fp16_f8e4`, `kernel_fp16_f8e5`, `kernel_bf16_fp64`, `kernel_bf16_f8e4`, `kernel_bf16_f8e5`, `kernel_fp32_fp64`, `kernel_fp32_f8e4`, `kernel_fp32_f8e5`, `kernel_fp64_fp16`, `kernel_fp64_bf16`, `kernel_fp64_fp32`, `kernel_fp64_fp64`, `kernel_fp64_f8e4`, `kernel_fp64_f8e5`, `kernel_f8e4_fp16`, `kernel_f8e4_bf16`, `kernel_f8e4_fp32`, `kernel_f8e4_fp64`, `kernel_f8e4_f8e4`, `kernel_f8e4_f8e5`, `kernel_f8e5_fp16`, `kernel_f8e5_bf16`, `kernel_f8e5_fp32`, `kernel_f8e5_fp64`, `kernel_f8e5_f8e4`, `kernel_f8e5_f8e5`

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

## Public theorem: `candidate_correctness`

**Statement:**
```lean
specification candidate_correctness (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (s : BlockState) (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.RegionIO (kernel input output Bool.true Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel p s Q K V L O)
```

**Assumptions / layout contracts:**
- `hBN : 0 < p.BLOCK_N`
- `hhi : 0 < hi p (s.pids 0)`

**Closed-form spec defs (transitive):** `kernel`, `kernel_fp16_fp16`, `kernel_fp16_bf16`, `kernel_fp16_fp32`, `kernel_bf16_fp16`, `kernel_bf16_bf16`, `kernel_bf16_fp32`, `kernel_fp32_fp16`, `kernel_fp32_bf16`, `kernel_fp32_fp32`, `kernel_fp16_fp64`, `kernel_fp16_f8e4`, `kernel_fp16_f8e5`, `kernel_bf16_fp64`, `kernel_bf16_f8e4`, `kernel_bf16_f8e5`, `kernel_fp32_fp64`, `kernel_fp32_f8e4`, `kernel_fp32_f8e5`, `kernel_fp64_fp16`, `kernel_fp64_bf16`, `kernel_fp64_fp32`, `kernel_fp64_fp64`, `kernel_fp64_f8e4`, `kernel_fp64_f8e5`, `kernel_f8e4_fp16`, `kernel_f8e4_bf16`, `kernel_f8e4_fp32`, `kernel_f8e4_fp64`, `kernel_f8e4_f8e4`, `kernel_f8e4_f8e5`, `kernel_f8e5_fp16`, `kernel_f8e5_bf16`, `kernel_f8e5_fp32`, `kernel_f8e5_fp64`, `kernel_f8e5_f8e4`, `kernel_f8e5_f8e5`

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

## Public theorem: `flat_correctness`

**Statement:**
```lean
specification flat_correctness (swapped : Bool) (input output : Storage) (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (hb : AddressBounds p s Q K V L O A.extent)
    (hundef : s.undef = fun _ _ => 0)
    (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.RegionFlatIO (kernel input output swapped Q K V L O p.sm_scale
      stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
      stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
      stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
      BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel A p s Q K V L O)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hb : AddressBounds p s Q K V L O A.extent`
- `hundef : s.undef = fun _ _ => 0`
- `hBN : 0 < p.BLOCK_N`
- `hhi : 0 < hi p (s.pids 0)`

**Closed-form spec defs (transitive):** `kernel`, `kernel_fp16_fp16`, `kernel_fp16_bf16`, `kernel_fp16_fp32`, `kernel_bf16_fp16`, `kernel_bf16_bf16`, `kernel_bf16_fp32`, `kernel_fp32_fp16`, `kernel_fp32_bf16`, `kernel_fp32_fp32`, `kernel_fp16_fp64`, `kernel_fp16_f8e4`, `kernel_fp16_f8e5`, `kernel_bf16_fp64`, `kernel_bf16_f8e4`, `kernel_bf16_f8e5`, `kernel_fp32_fp64`, `kernel_fp32_f8e4`, `kernel_fp32_f8e5`, `kernel_fp64_fp16`, `kernel_fp64_bf16`, `kernel_fp64_fp32`, `kernel_fp64_fp64`, `kernel_fp64_f8e4`, `kernel_fp64_f8e5`, `kernel_f8e4_fp16`, `kernel_f8e4_bf16`, `kernel_f8e4_fp32`, `kernel_f8e4_fp64`, `kernel_f8e4_f8e4`, `kernel_f8e4_f8e5`, `kernel_f8e5_fp16`, `kernel_f8e5_bf16`, `kernel_f8e5_fp32`, `kernel_f8e5_fp64`, `kernel_f8e5_f8e4`, `kernel_f8e5_f8e5`

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

## Public theorem: `full_flat_correctness`

<details><summary>docstring</summary>

```
/-- Both complete typed sources have actual access safety, successful finite-flat
execution, both independent attention outputs and the full raw-cell frame.
The formula includes every physically loaded padded KV lane. Output addresses
may alias: the reference retains the source's L-then-O store order. Floating
casts are erased here; this is not an IEEE or native frontend theorem. -/
```
</details>

**Statement:**
```lean
specification full_flat_correctness (swapped : Bool) (input : Storage)
    (Q K V L O : RegionName) (p : Layout)
    (stride_q_bs stride_k_bs stride_k_head stride_v_bs stride_v_head stride_v_seqlen
      stride_o_bs stride_o_head BS HEAD : Nat)
    (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (s : BlockState) (hb : AddressBounds p s Q K V L O A.extent)
    (hundef : s.undef = fun _ _ => 0)
    (hBN : 0 < p.BLOCK_N) (hhi : 0 < hi p (s.pids 0)) :
    VeriTile.Spec.Real (Contracts.FlatIO
      (actualKernel input swapped Q K V L O p.sm_scale
        stride_q_bs p.stride_q_head p.stride_q_seqlen p.stride_q_dim
        stride_k_bs stride_k_head p.stride_k_seqlen p.stride_k_dim
        stride_v_bs stride_v_head stride_v_seqlen p.stride_v_dim
        stride_o_bs stride_o_head p.stride_o_seqlen p.stride_o_dim
        BS HEAD p.SEQLEN p.BLOCK_M p.DIM p.BLOCK_N p.causal).eraseDType.toAlgKernel
      A p s Q K V L O)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hcov : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hb : AddressBounds p s Q K V L O A.extent`
- `hundef : s.undef = fun _ _ => 0`
- `hBN : 0 < p.BLOCK_N`
- `hhi : 0 < hi p (s.pids 0)`

**Closed-form spec defs (transitive):** `actualKernel`, `actual_fp16`, `actual_bf16`, `actual_fp32`, `actual_fp64`, `actual_f8e4`, `actual_f8e5`

<details><summary><code>actualKernel</code></summary>

```
/-- The complete unmodified source and its natural head-offset normalization.
The explicit output cast in Python requires an fp16 block-pointer destination.
Q's fp64 scaling is retained before its explicit fp16 conversion. -/
```
```lean
def actualKernel (input : Storage) (swapped : Bool) := match input with
  | .fp16 => actual_fp16 swapped
  | .bf16 => actual_bf16 swapped
  | .fp32 => actual_fp32 swapped
  | .fp64 => actual_fp64 swapped
  | .f8e4 => actual_f8e4 swapped
  | .f8e5 => actual_f8e5 swapped

end
#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>actual_fp16</code></summary>

```lean
abbrev actual_fp16 (swapped : Bool) := if swapped then flashAttnSource%[tl.float16,tl.float16,actualIndexCommuted] else flashAttnSource%[tl.float16,tl.float16,actualOriginal]
```
</details>

<details><summary><code>actual_bf16</code></summary>

```lean
abbrev actual_bf16 (swapped : Bool) := if swapped then flashAttnSource%[tl.bfloat16,tl.float16,actualIndexCommuted] else flashAttnSource%[tl.bfloat16,tl.float16,actualOriginal]
```
</details>

<details><summary><code>actual_fp32</code></summary>

```lean
abbrev actual_fp32 (swapped : Bool) := if swapped then flashAttnSource%[tl.float32,tl.float16,actualIndexCommuted] else flashAttnSource%[tl.float32,tl.float16,actualOriginal]
```
</details>

<details><summary><code>actual_fp64</code></summary>

```lean
abbrev actual_fp64 (swapped : Bool) := if swapped then flashAttnSource%[tl.float64,tl.float16,actualIndexCommuted] else flashAttnSource%[tl.float64,tl.float16,actualOriginal]
```
</details>

<details><summary><code>actual_f8e4</code></summary>

```lean
abbrev actual_f8e4 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e4nv,tl.float16,actualIndexCommuted] else flashAttnSource%[tl.float8e4nv,tl.float16,actualOriginal]
```
</details>

<details><summary><code>actual_f8e5</code></summary>

```lean
abbrev actual_f8e5 (swapped : Bool) := if swapped then flashAttnSource%[tl.float8e5,tl.float16,actualIndexCommuted] else flashAttnSource%[tl.float8e5,tl.float16,actualOriginal]
```
</details>

## Also present (pinned special-case summaries)
- `flash_attn_genuine_output_compute_correct_general`
- `flash_attn_genuine_l_compute_correct_general`
- `flash_attn_python_case1_genuine_compute_correct_general`
- `flash_attn_python_case2_genuine_compute_correct_general`
