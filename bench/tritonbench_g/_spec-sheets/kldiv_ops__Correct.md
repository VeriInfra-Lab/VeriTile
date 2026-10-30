# Spec sheet — `bench/tritonbench_g/kldiv_ops/Correct.lean`

**Python source:** `bench/tritonbench_g/kldiv_ops/kldiv_ops.py`

## Public theorem: `kldiv_backward_default_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the default backward one-block slice. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_default_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_default target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kldiv_backward_log_target_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the log-target backward one-block slice. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_log_target_compute_correct
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_log_target target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (new_grads_ptr, linearOffset s new_grads_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s new_grads_stride i)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kldiv_forward_default_none_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the forward `log_target=False`,
`reduction=0` one-block slice. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride eps i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kldiv_forward_log_target_none_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the forward log-target one-block slice. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kldiv_forward_log_target_none_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface**: the `log_target = True`,
`reduction = None` forward slice implements `exp(y_true) · (y_true − y)` on its
masked IO signature — for every disjoint flat placement of the three buffers,
every program id whose active lanes are in bounds, and every launch state whose
input windows hold `ys`/`gts` at the active lanes, the translated pointer kernel
terminates, every active output lane holds the spec value, and every other
memory cell is unchanged.

This is the audit-once surface; the `Realizes_without_Rounding` face above
states the same computation against a declared write map but does not pin the
flat placement. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_log_target_none_correctness
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    kldivForwardLogTargetIO y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE
      ⊨ fun ys gts i => Real.exp (gts i) * (gts i - ys i)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kldiv_forward_default_none_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface**: the `log_target = False` (default),
`reduction = None` forward slice implements `y_true · (log(max(y_true, eps)) − y)` on its
masked IO signature — for every disjoint flat placement of the three buffers,
every program id whose active lanes are in bounds, and every launch state whose
input windows hold `ys`/`gts` at the active lanes, the translated pointer kernel
terminates, every active output lane holds the spec value, and every other
memory cell is unchanged.

This is the audit-once surface; the `Realizes_without_Rounding` face above
states the same computation against a declared write map but does not pin the
flat placement. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_default_none_correctness
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) :
    kldivForwardDefaultIO y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE eps
      ⊨ fun ys gts i => gts i * (Real.log (max (gts i) eps) - ys i)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kldiv_backward_default_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for the `log_target = False` backward
slice: the gradient is `−target` on every active lane. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_default_correctness
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    kldivBackwardDefaultIO target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE
      ⊨ fun ts i => ts i * (0.0 - 1)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kldiv_backward_log_target_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for the `log_target = True` backward
slice: the gradient is `−exp(target)` on every active lane. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_log_target_correctness
    (target_ptr new_grads_ptr : RegionName)
    (target_stride new_grads_stride n_cols BLOCK_SIZE : Nat) :
    kldivBackwardLogTargetIO target_ptr new_grads_ptr
        target_stride new_grads_stride n_cols BLOCK_SIZE
      ⊨ fun ts i => 0.0 - Real.exp (ts i)
```

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `forward_default_element_correct`

<details><summary>docstring</summary>

```
/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
```
</details>

**Statement:**
```lean
specification forward_default_element_correct (pred target output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ)
    (_hSource : ForwardDefaultSupported pred target output 0)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N,
      s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardElementIO
      (forwardDefault pred target output v Y T O sy st so N B eps 0)
      A s Y T O sy st so N eps Bool.false)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hO : O ∈ A.regions`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ ↦ 0`
- `hB : 0 < B`
- `hYO : Y ≠ O`
- `hTO : T ≠ O`
- `hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y`
- `hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T`
- `hbo : ∀ k : Fin N,
      s.pids 0 * so + k.val < A.extent O`

**Closed-form spec defs (transitive):** `forwardDefault`, `forwardDefaultOriginal`, `commuteOutput`, `forwardDefault_fp32_fp32`, `forwardDefault_fp32_fp64`, `forwardDefault_fp32_fp16`, `forwardDefault_fp32_bf16`, `forwardDefault_fp32_f8e4`, `forwardDefault_fp32_f8e5`, `forwardDefault_fp64_fp32`, `forwardDefault_fp64_fp64`, `forwardDefault_fp64_fp16`, `forwardDefault_fp64_bf16`, `forwardDefault_fp64_f8e4`, `forwardDefault_fp64_f8e5`, `forwardDefault_fp16_fp32`, `forwardDefault_fp16_fp64`, `forwardDefault_fp16_fp16`, `forwardDefault_fp16_bf16`, `forwardDefault_fp16_f8e4`, `forwardDefault_fp16_f8e5`, `forwardDefault_bf16_fp32`, `forwardDefault_bf16_fp64`, `forwardDefault_bf16_fp16`, `forwardDefault_bf16_bf16`, `forwardDefault_bf16_f8e4`, `forwardDefault_bf16_f8e5`, `forwardDefault_f8e4_fp32`, `forwardDefault_f8e4_fp64`, `forwardDefault_f8e4_fp16`, `forwardDefault_f8e4_bf16`, `forwardDefault_f8e4_f8e4`, `forwardDefault_f8e4_f8e5`, `forwardDefault_f8e5_fp32`, `forwardDefault_f8e5_fp64`, `forwardDefault_f8e5_fp16`, `forwardDefault_f8e5_bf16`, `forwardDefault_f8e5_f8e4`, `forwardDefault_f8e5_f8e5`

<details><summary><code>forwardDefault</code></summary>

```lean
def forwardDefault (pred : Storage) (target : Storage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : ComputeKernel :=
  let source := forwardDefaultOriginal pred target output Y T O sy st so N B eps reduction
  if v then commuteOutput source 3 O so else source
```
</details>

<details><summary><code>forwardDefaultOriginal</code></summary>

```
/-- Complete forward source for the default target branch; all blocks and the final store remain. -/
```
```lean
def forwardDefaultOriginal (pred : Storage) (target : Storage) (output : Storage) : Forward :=
  match pred, target with
  | .fp32, .fp32 => forwardDefault_fp32_fp32 output
  | .fp32, .fp64 => forwardDefault_fp32_fp64 output
  | .fp32, .fp16 => forwardDefault_fp32_fp16 output
  | .fp32, .bf16 => forwardDefault_fp32_bf16 output
  | .fp32, .f8e4 => forwardDefault_fp32_f8e4 output
  | .fp32, .f8e5 => forwardDefault_fp32_f8e5 output
  | .fp64, .fp32 => forwardDefault_fp64_fp32 output
  | .fp64, .fp64 => forwardDefault_fp64_fp64 output
  | .fp64, .fp16 => forwardDefault_fp64_fp16 output
  | .fp64, .bf16 => forwardDefault_fp64_bf16 output
  | .fp64, .f8e4 => forwardDefault_fp64_f8e4 output
  | .fp64, .f8e5 => forwardDefault_fp64_f8e5 output
  | .fp16, .fp32 => forwardDefault_fp16_fp32 output
  | .fp16, .fp64 => forwardDefault_fp16_fp64 output
  | .fp16, .fp16 => forwardDefault_fp16_fp16 output
  | .fp16, .bf16 => forwardDefault_fp16_bf16 output
  | .fp16, .f8e4 => forwardDefault_fp16_f8e4 output
  | .fp16, .f8e5 => forwardDefault_fp16_f8e5 output
  | .bf16, .fp32 => forwardDefault_bf16_fp32 output
  | .bf16, .fp64 => forwardDefault_bf16_fp64 output
  | .bf16, .fp16 => forwardDefault_bf16_fp16 output
  | .bf16, .bf16 => forwardDefault_bf16_bf16 output
  | .bf16, .f8e4 => forwardDefault_bf16_f8e4 output
  | .bf16, .f8e5 => forwardDefault_bf16_f8e5 output
  | .f8e4, .fp32 => forwardDefault_f8e4_fp32 output
  | .f8e4, .fp64 => forwardDefault_f8e4_fp64 output
  | .f8e4, .fp16 => forwardDefault_f8e4_fp16 output
  | .f8e4, .bf16 => forwardDefault_f8e4_bf16 output
  | .f8e4, .f8e4 => forwardDefault_f8e4_f8e4 output
  | .f8e4, .f8e5 => forwardDefault_f8e4_f8e5 output
  | .f8e5, .fp32 => forwardDefault_f8e5_fp32 output
  | .f8e5, .fp64 => forwardDefault_f8e5_fp64 output
  | .f8e5, .fp16 => forwardDefault_f8e5_fp16 output
  | .f8e5, .bf16 => forwardDefault_f8e5_bf16 output
  | .f8e5, .f8e4 => forwardDefault_f8e5_f8e4 output
  | .f8e5, .f8e5 => forwardDefault_f8e5_f8e5 output
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate commutes only the exact signed output-row product. -/
```
```lean
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))
```
</details>

<details><summary><code>forwardDefault_fp32_fp32</code></summary>

```lean
def forwardDefault_fp32_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_fp64</code></summary>

```lean
def forwardDefault_fp32_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_fp16</code></summary>

```lean
def forwardDefault_fp32_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_bf16</code></summary>

```lean
def forwardDefault_fp32_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_f8e4</code></summary>

```lean
def forwardDefault_fp32_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_f8e5</code></summary>

```lean
def forwardDefault_fp32_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_fp32</code></summary>

```lean
def forwardDefault_fp64_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_fp64</code></summary>

```lean
def forwardDefault_fp64_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_fp16</code></summary>

```lean
def forwardDefault_fp64_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_bf16</code></summary>

```lean
def forwardDefault_fp64_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_f8e4</code></summary>

```lean
def forwardDefault_fp64_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_f8e5</code></summary>

```lean
def forwardDefault_fp64_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_fp32</code></summary>

```lean
def forwardDefault_fp16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_fp64</code></summary>

```lean
def forwardDefault_fp16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_fp16</code></summary>

```lean
def forwardDefault_fp16_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_bf16</code></summary>

```lean
def forwardDefault_fp16_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_f8e4</code></summary>

```lean
def forwardDefault_fp16_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_f8e5</code></summary>

```lean
def forwardDefault_fp16_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_fp32</code></summary>

```lean
def forwardDefault_bf16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_fp64</code></summary>

```lean
def forwardDefault_bf16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_fp16</code></summary>

```lean
def forwardDefault_bf16_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_bf16</code></summary>

```lean
def forwardDefault_bf16_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_f8e4</code></summary>

```lean
def forwardDefault_bf16_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_f8e5</code></summary>

```lean
def forwardDefault_bf16_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_fp32</code></summary>

```lean
def forwardDefault_f8e4_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_fp64</code></summary>

```lean
def forwardDefault_f8e4_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_fp16</code></summary>

```lean
def forwardDefault_f8e4_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_bf16</code></summary>

```lean
def forwardDefault_f8e4_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_f8e4</code></summary>

```lean
def forwardDefault_f8e4_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_f8e5</code></summary>

```lean
def forwardDefault_f8e4_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_fp32</code></summary>

```lean
def forwardDefault_f8e5_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_fp64</code></summary>

```lean
def forwardDefault_f8e5_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_fp16</code></summary>

```lean
def forwardDefault_f8e5_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_bf16</code></summary>

```lean
def forwardDefault_f8e5_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_f8e4</code></summary>

```lean
def forwardDefault_f8e5_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_f8e5</code></summary>

```lean
def forwardDefault_f8e5_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

## Public theorem: `forward_log_element_correct`

<details><summary>docstring</summary>

```
/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
```
</details>

**Statement:**
```lean
specification forward_log_element_correct (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ)
    (_hSource : ForwardLogSupported pred target output 0)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N,
      s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardElementIO
      (forwardLog pred target output v Y T O sy st so N B eps 0)
      A s Y T O sy st so N eps Bool.true)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hO : O ∈ A.regions`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ ↦ 0`
- `hB : 0 < B`
- `hYO : Y ≠ O`
- `hTO : T ≠ O`
- `hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y`
- `hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T`
- `hbo : ∀ k : Fin N,
      s.pids 0 * so + k.val < A.extent O`

**Closed-form spec defs (transitive):** `forwardLog`, `forwardLogOriginal`, `commuteOutput`, `forwardLog_fp32_fp32`, `forwardLog_fp32_fp64`, `forwardLog_fp64_fp32`, `forwardLog_fp64_fp64`, `forwardLog_fp16_fp32`, `forwardLog_fp16_fp64`, `forwardLog_bf16_fp32`, `forwardLog_bf16_fp64`, `forwardLog_f8e4_fp32`, `forwardLog_f8e4_fp64`, `forwardLog_f8e5_fp32`, `forwardLog_f8e5_fp64`

<details><summary><code>forwardLog</code></summary>

```lean
def forwardLog (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : ComputeKernel :=
  let source := forwardLogOriginal pred target output Y T O sy st so N B eps reduction
  if v then commuteOutput source 3 O so else source
```
</details>

<details><summary><code>forwardLogOriginal</code></summary>

```
/-- Complete forward source for the log target branch; all blocks and the final store remain. -/
```
```lean
def forwardLogOriginal (pred : Storage) (target : MathStorage) (output : Storage) : Forward :=
  match pred, target with
  | .fp32, .fp32 => forwardLog_fp32_fp32 output
  | .fp32, .fp64 => forwardLog_fp32_fp64 output
  | .fp64, .fp32 => forwardLog_fp64_fp32 output
  | .fp64, .fp64 => forwardLog_fp64_fp64 output
  | .fp16, .fp32 => forwardLog_fp16_fp32 output
  | .fp16, .fp64 => forwardLog_fp16_fp64 output
  | .bf16, .fp32 => forwardLog_bf16_fp32 output
  | .bf16, .fp64 => forwardLog_bf16_fp64 output
  | .f8e4, .fp32 => forwardLog_f8e4_fp32 output
  | .f8e4, .fp64 => forwardLog_f8e4_fp64 output
  | .f8e5, .fp32 => forwardLog_f8e5_fp32 output
  | .f8e5, .fp64 => forwardLog_f8e5_fp64 output
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate commutes only the exact signed output-row product. -/
```
```lean
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))
```
</details>

<details><summary><code>forwardLog_fp32_fp32</code></summary>

```lean
def forwardLog_fp32_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float32, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float32, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float32, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float32, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float32, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp32_fp64</code></summary>

```lean
def forwardLog_fp32_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float32, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float32, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float32, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float32, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float32, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp64_fp32</code></summary>

```lean
def forwardLog_fp64_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float64, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float64, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float64, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float64, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float64, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp64_fp64</code></summary>

```lean
def forwardLog_fp64_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float64, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float64, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float64, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float64, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float64, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp16_fp32</code></summary>

```lean
def forwardLog_fp16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float16, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float16, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float16, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float16, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float16, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp16_fp64</code></summary>

```lean
def forwardLog_fp16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float16, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float16, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float16, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float16, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float16, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_bf16_fp32</code></summary>

```lean
def forwardLog_bf16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_bf16_fp64</code></summary>

```lean
def forwardLog_bf16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e4_fp32</code></summary>

```lean
def forwardLog_f8e4_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e4_fp64</code></summary>

```lean
def forwardLog_f8e4_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e5_fp32</code></summary>

```lean
def forwardLog_f8e5_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e5_fp64</code></summary>

```lean
def forwardLog_f8e5_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e5, logarithmic]
```
</details>

## Public theorem: `forward_default_reduction_correct`

<details><summary>docstring</summary>

```
/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
```
</details>

**Statement:**
```lean
specification forward_default_reduction_correct (pred target output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (_hSource : ForwardDefaultSupported pred target output reduction)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hR : reduction ≠ 0)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : s.pids 0 * so < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardReductionIO
      (forwardDefault pred target output v Y T O sy st so N B eps reduction)
      A s Y T O sy st so N eps Bool.false)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hO : O ∈ A.regions`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ ↦ 0`
- `hB : 0 < B`
- `hR : reduction ≠ 0`
- `hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y`
- `hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T`
- `hbo : s.pids 0 * so < A.extent O`

**Closed-form spec defs (transitive):** `forwardDefault`, `forwardDefaultOriginal`, `commuteOutput`, `forwardDefault_fp32_fp32`, `forwardDefault_fp32_fp64`, `forwardDefault_fp32_fp16`, `forwardDefault_fp32_bf16`, `forwardDefault_fp32_f8e4`, `forwardDefault_fp32_f8e5`, `forwardDefault_fp64_fp32`, `forwardDefault_fp64_fp64`, `forwardDefault_fp64_fp16`, `forwardDefault_fp64_bf16`, `forwardDefault_fp64_f8e4`, `forwardDefault_fp64_f8e5`, `forwardDefault_fp16_fp32`, `forwardDefault_fp16_fp64`, `forwardDefault_fp16_fp16`, `forwardDefault_fp16_bf16`, `forwardDefault_fp16_f8e4`, `forwardDefault_fp16_f8e5`, `forwardDefault_bf16_fp32`, `forwardDefault_bf16_fp64`, `forwardDefault_bf16_fp16`, `forwardDefault_bf16_bf16`, `forwardDefault_bf16_f8e4`, `forwardDefault_bf16_f8e5`, `forwardDefault_f8e4_fp32`, `forwardDefault_f8e4_fp64`, `forwardDefault_f8e4_fp16`, `forwardDefault_f8e4_bf16`, `forwardDefault_f8e4_f8e4`, `forwardDefault_f8e4_f8e5`, `forwardDefault_f8e5_fp32`, `forwardDefault_f8e5_fp64`, `forwardDefault_f8e5_fp16`, `forwardDefault_f8e5_bf16`, `forwardDefault_f8e5_f8e4`, `forwardDefault_f8e5_f8e5`

<details><summary><code>forwardDefault</code></summary>

```lean
def forwardDefault (pred : Storage) (target : Storage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : ComputeKernel :=
  let source := forwardDefaultOriginal pred target output Y T O sy st so N B eps reduction
  if v then commuteOutput source 3 O so else source
```
</details>

<details><summary><code>forwardDefaultOriginal</code></summary>

```
/-- Complete forward source for the default target branch; all blocks and the final store remain. -/
```
```lean
def forwardDefaultOriginal (pred : Storage) (target : Storage) (output : Storage) : Forward :=
  match pred, target with
  | .fp32, .fp32 => forwardDefault_fp32_fp32 output
  | .fp32, .fp64 => forwardDefault_fp32_fp64 output
  | .fp32, .fp16 => forwardDefault_fp32_fp16 output
  | .fp32, .bf16 => forwardDefault_fp32_bf16 output
  | .fp32, .f8e4 => forwardDefault_fp32_f8e4 output
  | .fp32, .f8e5 => forwardDefault_fp32_f8e5 output
  | .fp64, .fp32 => forwardDefault_fp64_fp32 output
  | .fp64, .fp64 => forwardDefault_fp64_fp64 output
  | .fp64, .fp16 => forwardDefault_fp64_fp16 output
  | .fp64, .bf16 => forwardDefault_fp64_bf16 output
  | .fp64, .f8e4 => forwardDefault_fp64_f8e4 output
  | .fp64, .f8e5 => forwardDefault_fp64_f8e5 output
  | .fp16, .fp32 => forwardDefault_fp16_fp32 output
  | .fp16, .fp64 => forwardDefault_fp16_fp64 output
  | .fp16, .fp16 => forwardDefault_fp16_fp16 output
  | .fp16, .bf16 => forwardDefault_fp16_bf16 output
  | .fp16, .f8e4 => forwardDefault_fp16_f8e4 output
  | .fp16, .f8e5 => forwardDefault_fp16_f8e5 output
  | .bf16, .fp32 => forwardDefault_bf16_fp32 output
  | .bf16, .fp64 => forwardDefault_bf16_fp64 output
  | .bf16, .fp16 => forwardDefault_bf16_fp16 output
  | .bf16, .bf16 => forwardDefault_bf16_bf16 output
  | .bf16, .f8e4 => forwardDefault_bf16_f8e4 output
  | .bf16, .f8e5 => forwardDefault_bf16_f8e5 output
  | .f8e4, .fp32 => forwardDefault_f8e4_fp32 output
  | .f8e4, .fp64 => forwardDefault_f8e4_fp64 output
  | .f8e4, .fp16 => forwardDefault_f8e4_fp16 output
  | .f8e4, .bf16 => forwardDefault_f8e4_bf16 output
  | .f8e4, .f8e4 => forwardDefault_f8e4_f8e4 output
  | .f8e4, .f8e5 => forwardDefault_f8e4_f8e5 output
  | .f8e5, .fp32 => forwardDefault_f8e5_fp32 output
  | .f8e5, .fp64 => forwardDefault_f8e5_fp64 output
  | .f8e5, .fp16 => forwardDefault_f8e5_fp16 output
  | .f8e5, .bf16 => forwardDefault_f8e5_bf16 output
  | .f8e5, .f8e4 => forwardDefault_f8e5_f8e4 output
  | .f8e5, .f8e5 => forwardDefault_f8e5_f8e5 output
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate commutes only the exact signed output-row product. -/
```
```lean
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))
```
</details>

<details><summary><code>forwardDefault_fp32_fp32</code></summary>

```lean
def forwardDefault_fp32_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_fp64</code></summary>

```lean
def forwardDefault_fp32_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_fp16</code></summary>

```lean
def forwardDefault_fp32_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_bf16</code></summary>

```lean
def forwardDefault_fp32_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_f8e4</code></summary>

```lean
def forwardDefault_fp32_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp32_f8e5</code></summary>

```lean
def forwardDefault_fp32_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_fp32</code></summary>

```lean
def forwardDefault_fp64_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_fp64</code></summary>

```lean
def forwardDefault_fp64_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_fp16</code></summary>

```lean
def forwardDefault_fp64_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_bf16</code></summary>

```lean
def forwardDefault_fp64_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_f8e4</code></summary>

```lean
def forwardDefault_fp64_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp64_f8e5</code></summary>

```lean
def forwardDefault_fp64_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_fp32</code></summary>

```lean
def forwardDefault_fp16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_fp64</code></summary>

```lean
def forwardDefault_fp16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_fp16</code></summary>

```lean
def forwardDefault_fp16_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_bf16</code></summary>

```lean
def forwardDefault_fp16_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_f8e4</code></summary>

```lean
def forwardDefault_fp16_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_fp16_f8e5</code></summary>

```lean
def forwardDefault_fp16_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_fp32</code></summary>

```lean
def forwardDefault_bf16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_fp64</code></summary>

```lean
def forwardDefault_bf16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_fp16</code></summary>

```lean
def forwardDefault_bf16_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_bf16</code></summary>

```lean
def forwardDefault_bf16_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_f8e4</code></summary>

```lean
def forwardDefault_bf16_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_bf16_f8e5</code></summary>

```lean
def forwardDefault_bf16_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_fp32</code></summary>

```lean
def forwardDefault_f8e4_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_fp64</code></summary>

```lean
def forwardDefault_f8e4_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_fp16</code></summary>

```lean
def forwardDefault_f8e4_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_bf16</code></summary>

```lean
def forwardDefault_f8e4_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_f8e4</code></summary>

```lean
def forwardDefault_f8e4_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e4_f8e5</code></summary>

```lean
def forwardDefault_f8e4_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_fp32</code></summary>

```lean
def forwardDefault_f8e5_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_fp64</code></summary>

```lean
def forwardDefault_f8e5_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_fp16</code></summary>

```lean
def forwardDefault_f8e5_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_bf16</code></summary>

```lean
def forwardDefault_f8e5_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_f8e4</code></summary>

```lean
def forwardDefault_f8e5_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>forwardDefault_f8e5_f8e5</code></summary>

```lean
def forwardDefault_f8e5_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

## Public theorem: `forward_log_reduction_correct`

<details><summary>docstring</summary>

```
/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
```
</details>

**Statement:**
```lean
specification forward_log_reduction_correct (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (Y T O : RegionName)
    (sy st so N B : Nat) (eps : ℝ) (reduction : Nat)
    (_hSource : ForwardLogSupported pred target output reduction)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hR : reduction ≠ 0)
    (hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : s.pids 0 * so < A.extent O) :
    VeriTile.Spec.Real (Contracts.ForwardReductionIO
      (forwardLog pred target output v Y T O sy st so N B eps reduction)
      A s Y T O sy st so N eps Bool.true)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hO : O ∈ A.regions`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ ↦ 0`
- `hB : 0 < B`
- `hR : reduction ≠ 0`
- `hby : ∀ k : Fin N, s.pids 0 * sy + k.val < A.extent Y`
- `hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T`
- `hbo : s.pids 0 * so < A.extent O`

**Closed-form spec defs (transitive):** `forwardLog`, `forwardLogOriginal`, `commuteOutput`, `forwardLog_fp32_fp32`, `forwardLog_fp32_fp64`, `forwardLog_fp64_fp32`, `forwardLog_fp64_fp64`, `forwardLog_fp16_fp32`, `forwardLog_fp16_fp64`, `forwardLog_bf16_fp32`, `forwardLog_bf16_fp64`, `forwardLog_f8e4_fp32`, `forwardLog_f8e4_fp64`, `forwardLog_f8e5_fp32`, `forwardLog_f8e5_fp64`

<details><summary><code>forwardLog</code></summary>

```lean
def forwardLog (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : ComputeKernel :=
  let source := forwardLogOriginal pred target output Y T O sy st so N B eps reduction
  if v then commuteOutput source 3 O so else source
```
</details>

<details><summary><code>forwardLogOriginal</code></summary>

```
/-- Complete forward source for the log target branch; all blocks and the final store remain. -/
```
```lean
def forwardLogOriginal (pred : Storage) (target : MathStorage) (output : Storage) : Forward :=
  match pred, target with
  | .fp32, .fp32 => forwardLog_fp32_fp32 output
  | .fp32, .fp64 => forwardLog_fp32_fp64 output
  | .fp64, .fp32 => forwardLog_fp64_fp32 output
  | .fp64, .fp64 => forwardLog_fp64_fp64 output
  | .fp16, .fp32 => forwardLog_fp16_fp32 output
  | .fp16, .fp64 => forwardLog_fp16_fp64 output
  | .bf16, .fp32 => forwardLog_bf16_fp32 output
  | .bf16, .fp64 => forwardLog_bf16_fp64 output
  | .f8e4, .fp32 => forwardLog_f8e4_fp32 output
  | .f8e4, .fp64 => forwardLog_f8e4_fp64 output
  | .f8e5, .fp32 => forwardLog_f8e5_fp32 output
  | .f8e5, .fp64 => forwardLog_f8e5_fp64 output
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate commutes only the exact signed output-row product. -/
```
```lean
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))
```
</details>

<details><summary><code>forwardLog_fp32_fp32</code></summary>

```lean
def forwardLog_fp32_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float32, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float32, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float32, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float32, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float32, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp32_fp64</code></summary>

```lean
def forwardLog_fp32_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float32, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float32, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float32, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float32, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float32, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp64_fp32</code></summary>

```lean
def forwardLog_fp64_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float64, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float64, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float64, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float64, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float64, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp64_fp64</code></summary>

```lean
def forwardLog_fp64_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float64, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float64, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float64, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float64, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float64, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp16_fp32</code></summary>

```lean
def forwardLog_fp16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float16, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float16, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float16, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float16, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float16, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_fp16_fp64</code></summary>

```lean
def forwardLog_fp16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float16, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float16, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float16, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float16, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float16, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_bf16_fp32</code></summary>

```lean
def forwardLog_bf16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_bf16_fp64</code></summary>

```lean
def forwardLog_bf16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e4_fp32</code></summary>

```lean
def forwardLog_f8e4_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e4_fp64</code></summary>

```lean
def forwardLog_f8e4_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e5_fp32</code></summary>

```lean
def forwardLog_f8e5_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>forwardLog_f8e5_fp64</code></summary>

```lean
def forwardLog_f8e5_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e5, logarithmic]
```
</details>

## Public theorem: `backward_default_correct`

<details><summary>docstring</summary>

```
/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
```
</details>

**Statement:**
```lean
specification backward_default_correct (target output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (T O : RegionName)
    (st so N B : Nat)
    (_hSource : BackwardSourceSupported target output Bool.false)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.BackwardIO
      (backwardDefault target output v T O st so N B)
      A s T O st so N Bool.false)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hO : O ∈ A.regions`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ ↦ 0`
- `hB : 0 < B`
- `hTO : T ≠ O`
- `hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T`
- `hbo : ∀ k : Fin N, s.pids 0 * so + k.val < A.extent O`

**Closed-form spec defs (transitive):** `backwardDefault`, `backwardDefaultOriginal`, `commuteOutput`

<details><summary><code>backwardDefault</code></summary>

```lean
def backwardDefault (target : Storage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  let source := backwardDefaultOriginal target output T O st so N B
  if v then commuteOutput source 2 O so else source
```
</details>

<details><summary><code>backwardDefaultOriginal</code></summary>

```
/-- Complete backward default branch, retaining T then O pointer updates and the pre-loop mask. -/
```
```lean
def backwardDefaultOriginal (target : Storage) (output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klOpsBackward%[tl.float32, tl.float32, defaultTarget]
  | .fp32, .fp64 => klOpsBackward%[tl.float32, tl.float64, defaultTarget]
  | .fp32, .fp16 => klOpsBackward%[tl.float32, tl.float16, defaultTarget]
  | .fp32, .bf16 => klOpsBackward%[tl.float32, tl.bfloat16, defaultTarget]
  | .fp32, .f8e4 => klOpsBackward%[tl.float32, tl.float8e4nv, defaultTarget]
  | .fp32, .f8e5 => klOpsBackward%[tl.float32, tl.float8e5, defaultTarget]
  | .fp64, .fp32 => klOpsBackward%[tl.float64, tl.float32, defaultTarget]
  | .fp64, .fp64 => klOpsBackward%[tl.float64, tl.float64, defaultTarget]
  | .fp64, .fp16 => klOpsBackward%[tl.float64, tl.float16, defaultTarget]
  | .fp64, .bf16 => klOpsBackward%[tl.float64, tl.bfloat16, defaultTarget]
  | .fp64, .f8e4 => klOpsBackward%[tl.float64, tl.float8e4nv, defaultTarget]
  | .fp64, .f8e5 => klOpsBackward%[tl.float64, tl.float8e5, defaultTarget]
  | .fp16, .fp32 => klOpsBackward%[tl.float16, tl.float32, defaultTarget]
  | .fp16, .fp64 => klOpsBackward%[tl.float16, tl.float64, defaultTarget]
  | .fp16, .fp16 => klOpsBackward%[tl.float16, tl.float16, defaultTarget]
  | .fp16, .bf16 => klOpsBackward%[tl.float16, tl.bfloat16, defaultTarget]
  | .fp16, .f8e4 => klOpsBackward%[tl.float16, tl.float8e4nv, defaultTarget]
  | .fp16, .f8e5 => klOpsBackward%[tl.float16, tl.float8e5, defaultTarget]
  | .bf16, .fp32 => klOpsBackward%[tl.bfloat16, tl.float32, defaultTarget]
  | .bf16, .fp64 => klOpsBackward%[tl.bfloat16, tl.float64, defaultTarget]
  | .bf16, .fp16 => klOpsBackward%[tl.bfloat16, tl.float16, defaultTarget]
  | .bf16, .bf16 => klOpsBackward%[tl.bfloat16, tl.bfloat16, defaultTarget]
  | .bf16, .f8e4 => klOpsBackward%[tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .bf16, .f8e5 => klOpsBackward%[tl.bfloat16, tl.float8e5, defaultTarget]
  | .f8e4, .fp32 => klOpsBackward%[tl.float8e4nv, tl.float32, defaultTarget]
  | .f8e4, .fp64 => klOpsBackward%[tl.float8e4nv, tl.float64, defaultTarget]
  | .f8e4, .fp16 => klOpsBackward%[tl.float8e4nv, tl.float16, defaultTarget]
  | .f8e4, .bf16 => klOpsBackward%[tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4, .f8e4 => klOpsBackward%[tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e4, .f8e5 => klOpsBackward%[tl.float8e4nv, tl.float8e5, defaultTarget]
  | .f8e5, .fp32 => klOpsBackward%[tl.float8e5, tl.float32, defaultTarget]
  | .f8e5, .fp64 => klOpsBackward%[tl.float8e5, tl.float64, defaultTarget]
  | .f8e5, .fp16 => klOpsBackward%[tl.float8e5, tl.float16, defaultTarget]
  | .f8e5, .bf16 => klOpsBackward%[tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e5, .f8e4 => klOpsBackward%[tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5, .f8e5 => klOpsBackward%[tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate commutes only the exact signed output-row product. -/
```
```lean
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))
```
</details>

## Public theorem: `backward_log_correct`

<details><summary>docstring</summary>

```
/-- Both variants execute the entire source loop and satisfy the independent
formula, actual source trace safety and the complete raw-memory frame. Dtype erasure is explicit. -/
```
</details>

**Statement:**
```lean
specification backward_log_correct (target : MathStorage) (output : Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (T O : RegionName)
    (st so N B : Nat)
    (_hSource : BackwardSourceSupported target.storage output Bool.true)
    (hd : A.Disjoint) (hO : O ∈ A.regions)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = fun _ _ ↦ 0)
    (hB : 0 < B) (hTO : T ≠ O)
    (hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T)
    (hbo : ∀ k : Fin N, s.pids 0 * so + k.val < A.extent O) :
    VeriTile.Spec.Real (Contracts.BackwardIO
      (backwardLog target output v T O st so N B)
      A s T O st so N Bool.true)
```

**Assumptions / layout contracts:**
- `hd : A.Disjoint`
- `hO : O ∈ A.regions`
- `hc : ∀ r, r ∉ A.regions → A.extent r = 0`
- `hu : s.undef = fun _ _ ↦ 0`
- `hB : 0 < B`
- `hTO : T ≠ O`
- `hbt : ∀ k : Fin N, s.pids 0 * st + k.val < A.extent T`
- `hbo : ∀ k : Fin N, s.pids 0 * so + k.val < A.extent O`

**Closed-form spec defs (transitive):** `backwardLog`, `backwardLogOriginal`, `commuteOutput`

<details><summary><code>backwardLog</code></summary>

```lean
def backwardLog (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  let source := backwardLogOriginal target output T O st so N B
  if v then commuteOutput source 2 O so else source
```
</details>

<details><summary><code>backwardLogOriginal</code></summary>

```
/-- Complete backward log branch, retaining T then O pointer updates and the pre-loop mask. -/
```
```lean
def backwardLogOriginal (target : MathStorage) (output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klOpsBackward%[tl.float32, tl.float32, logarithmic]
  | .fp32, .fp64 => klOpsBackward%[tl.float32, tl.float64, logarithmic]
  | .fp32, .fp16 => klOpsBackward%[tl.float32, tl.float16, logarithmic]
  | .fp32, .bf16 => klOpsBackward%[tl.float32, tl.bfloat16, logarithmic]
  | .fp32, .f8e4 => klOpsBackward%[tl.float32, tl.float8e4nv, logarithmic]
  | .fp32, .f8e5 => klOpsBackward%[tl.float32, tl.float8e5, logarithmic]
  | .fp64, .fp32 => klOpsBackward%[tl.float64, tl.float32, logarithmic]
  | .fp64, .fp64 => klOpsBackward%[tl.float64, tl.float64, logarithmic]
  | .fp64, .fp16 => klOpsBackward%[tl.float64, tl.float16, logarithmic]
  | .fp64, .bf16 => klOpsBackward%[tl.float64, tl.bfloat16, logarithmic]
  | .fp64, .f8e4 => klOpsBackward%[tl.float64, tl.float8e4nv, logarithmic]
  | .fp64, .f8e5 => klOpsBackward%[tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate commutes only the exact signed output-row product. -/
```
```lean
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))
```
</details>
