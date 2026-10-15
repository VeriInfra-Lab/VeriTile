# Spec sheet — `bench/tritonbench_g/kldiv_ops/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/kldiv_ops/kldiv_ops.py`

## Public theorem: `forward_default_equivalence`

<details><summary>docstring</summary>

```
/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
```
</details>

**Statement:**
```lean
specification forward_default_equivalence (pred target output : Storage)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (_hSource : ForwardDefaultSupported pred target output reduction) :
    forward_default_program pred target output Bool.false Y T O sy st so N B eps reduction ≡[assumptions]
      forward_default_program pred target output Bool.true Y T O sy st so N B eps reduction
```

**Closed-form spec defs (transitive):** `forward_default_program`, `assumptions`, `forwardDefault`, `forwardDefaultOriginal`, `commuteOutput`, `forwardDefault_fp32_fp32`, `forwardDefault_fp32_fp64`, `forwardDefault_fp32_fp16`, `forwardDefault_fp32_bf16`, `forwardDefault_fp32_f8e4`, `forwardDefault_fp32_f8e5`, `forwardDefault_fp64_fp32`, `forwardDefault_fp64_fp64`, `forwardDefault_fp64_fp16`, `forwardDefault_fp64_bf16`, `forwardDefault_fp64_f8e4`, `forwardDefault_fp64_f8e5`, `forwardDefault_fp16_fp32`, `forwardDefault_fp16_fp64`, `forwardDefault_fp16_fp16`, `forwardDefault_fp16_bf16`, `forwardDefault_fp16_f8e4`, `forwardDefault_fp16_f8e5`, `forwardDefault_bf16_fp32`, `forwardDefault_bf16_fp64`, `forwardDefault_bf16_fp16`, `forwardDefault_bf16_bf16`, `forwardDefault_bf16_f8e4`, `forwardDefault_bf16_f8e5`, `forwardDefault_f8e4_fp32`, `forwardDefault_f8e4_fp64`, `forwardDefault_f8e4_fp16`, `forwardDefault_f8e4_bf16`, `forwardDefault_f8e4_f8e4`, `forwardDefault_f8e4_f8e5`, `forwardDefault_f8e5_fp32`, `forwardDefault_f8e5_fp64`, `forwardDefault_f8e5_fp16`, `forwardDefault_f8e5_bf16`, `forwardDefault_f8e5_f8e4`, `forwardDefault_f8e5_f8e5`

<details><summary><code>forward_default_program</code></summary>

```lean
def forward_default_program (pred target output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : Program where
  kernel := forwardDefault pred target output v Y T O sy st so N B eps reduction
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

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

## Public theorem: `forward_log_equivalence`

<details><summary>docstring</summary>

```
/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
```
</details>

**Statement:**
```lean
specification forward_log_equivalence (pred : Storage) (target : MathStorage) (output : Storage)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) (_hSource : ForwardLogSupported pred target output reduction) :
    forward_log_program pred target output Bool.false Y T O sy st so N B eps reduction ≡[assumptions]
      forward_log_program pred target output Bool.true Y T O sy st so N B eps reduction
```

**Closed-form spec defs (transitive):** `forward_log_program`, `assumptions`, `forwardLog`, `forwardLogOriginal`, `commuteOutput`, `forwardLog_fp32_fp32`, `forwardLog_fp32_fp64`, `forwardLog_fp64_fp32`, `forwardLog_fp64_fp64`, `forwardLog_fp16_fp32`, `forwardLog_fp16_fp64`, `forwardLog_bf16_fp32`, `forwardLog_bf16_fp64`, `forwardLog_f8e4_fp32`, `forwardLog_f8e4_fp64`, `forwardLog_f8e5_fp32`, `forwardLog_f8e5_fp64`

<details><summary><code>forward_log_program</code></summary>

```lean
def forward_log_program (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : Program where
  kernel := forwardLog pred target output v Y T O sy st so N B eps reduction
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

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

## Public theorem: `backward_default_equivalence`

<details><summary>docstring</summary>

```
/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
```
</details>

**Statement:**
```lean
specification backward_default_equivalence (target output : Storage)
    (T O : RegionName) (st so N B : Nat) (_hSource : BackwardSourceSupported target output Bool.false) :
    backward_default_program target output Bool.false T O st so N B ≡[assumptions]
      backward_default_program target output Bool.true T O st so N B
```

**Closed-form spec defs (transitive):** `backward_default_program`, `assumptions`, `backwardDefault`, `backwardDefaultOriginal`, `commuteOutput`

<details><summary><code>backward_default_program</code></summary>

```lean
def backward_default_program (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : Program where
  kernel := backwardDefault target output v T O st so N B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

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

## Public theorem: `backward_log_equivalence`

<details><summary>docstring</summary>

```
/-- The full source and its existing exact-address candidate have identical typed
executions. The empty assumption list is justified because floating arithmetic is unchanged. -/
```
</details>

**Statement:**
```lean
specification backward_log_equivalence (target : MathStorage) (output : Storage)
    (T O : RegionName) (st so N B : Nat) (_hSource : BackwardSourceSupported target.storage output Bool.true) :
    backward_log_program target output Bool.false T O st so N B ≡[assumptions]
      backward_log_program target output Bool.true T O st so N B
```

**Closed-form spec defs (transitive):** `backward_log_program`, `assumptions`, `backwardLog`, `backwardLogOriginal`, `commuteOutput`

<details><summary><code>backward_log_program</code></summary>

```lean
def backward_log_program (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : Program where
  kernel := backwardLog target output v T O st so N B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

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
