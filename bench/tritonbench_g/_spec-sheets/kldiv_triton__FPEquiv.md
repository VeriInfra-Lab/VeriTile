# Spec sheet — `bench/tritonbench_g/kldiv_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/kldiv_triton/kldiv_triton.py`

## Public theorem: `forward_equivalence`

<details><summary>docstring</summary>

```
/-- Full forward state/failure equivalence retains every chunk store and both source
branches. It imposes no floating law, successful-execution or output-value premise. -/
```
</details>

**Statement:**
```lean
specification forward_equivalence (pred : Storage) (target : MathStorage) (output : Storage)
    (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    forwardProgram pred target output Bool.false Y T O sy st so N B logTarget reduction
      ≡[assumptions]
    forwardProgram pred target output Bool.true Y T O sy st so N B logTarget reduction
```

**Closed-form spec defs (transitive):** `forwardProgram`, `assumptions`, `forward`, `forwardOriginal`, `commuteOutput`

<details><summary><code>forwardProgram</code></summary>

```lean
def forwardProgram (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    Program where
  kernel := forward pred target output v Y T O sy st so N B logTarget reduction
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>forward</code></summary>

```lean
def forward (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    ComputeKernel :=
  let source := forwardOriginal pred target output Y T O sy st so N B logTarget reduction
  if v then commuteOutput source 3 O so else source
```
</details>

<details><summary><code>forwardOriginal</code></summary>

```
/-- Every chunk is retained. Nonzero reduction writes one scalar per chunk and
advances the destination pointer; no whole-row reduction is substituted. -/
```
```lean
def forwardOriginal (pred : Storage) (target : MathStorage) (output : Storage) : Forward :=
  match pred, target, output with
  | .fp32, .fp32, .fp32 => klForward%[tl.float32, tl.float32, tl.float32]
  | .fp32, .fp32, .fp64 => klForward%[tl.float32, tl.float32, tl.float64]
  | .fp32, .fp32, .fp16 => klForward%[tl.float32, tl.float32, tl.float16]
  | .fp32, .fp32, .bf16 => klForward%[tl.float32, tl.float32, tl.bfloat16]
  | .fp32, .fp32, .f8e4 => klForward%[tl.float32, tl.float32, tl.float8e4nv]
  | .fp32, .fp32, .f8e5 => klForward%[tl.float32, tl.float32, tl.float8e5]
  | .fp32, .fp64, .fp32 => klForward%[tl.float32, tl.float64, tl.float32]
  | .fp32, .fp64, .fp64 => klForward%[tl.float32, tl.float64, tl.float64]
  | .fp32, .fp64, .fp16 => klForward%[tl.float32, tl.float64, tl.float16]
  | .fp32, .fp64, .bf16 => klForward%[tl.float32, tl.float64, tl.bfloat16]
  | .fp32, .fp64, .f8e4 => klForward%[tl.float32, tl.float64, tl.float8e4nv]
  | .fp32, .fp64, .f8e5 => klForward%[tl.float32, tl.float64, tl.float8e5]
  | .fp64, .fp32, .fp32 => klForward%[tl.float64, tl.float32, tl.float32]
  | .fp64, .fp32, .fp64 => klForward%[tl.float64, tl.float32, tl.float64]
  | .fp64, .fp32, .fp16 => klForward%[tl.float64, tl.float32, tl.float16]
  | .fp64, .fp32, .bf16 => klForward%[tl.float64, tl.float32, tl.bfloat16]
  | .fp64, .fp32, .f8e4 => klForward%[tl.float64, tl.float32, tl.float8e4nv]
  | .fp64, .fp32, .f8e5 => klForward%[tl.float64, tl.float32, tl.float8e5]
  | .fp64, .fp64, .fp32 => klForward%[tl.float64, tl.float64, tl.float32]
  | .fp64, .fp64, .fp64 => klForward%[tl.float64, tl.float64, tl.float64]
  | .fp64, .fp64, .fp16 => klForward%[tl.float64, tl.float64, tl.float16]
  | .fp64, .fp64, .bf16 => klForward%[tl.float64, tl.float64, tl.bfloat16]
  | .fp64, .fp64, .f8e4 => klForward%[tl.float64, tl.float64, tl.float8e4nv]
  | .fp64, .fp64, .f8e5 => klForward%[tl.float64, tl.float64, tl.float8e5]
  | .fp16, .fp32, .fp32 => klForward%[tl.float16, tl.float32, tl.float32]
  | .fp16, .fp32, .fp64 => klForward%[tl.float16, tl.float32, tl.float64]
  | .fp16, .fp32, .fp16 => klForward%[tl.float16, tl.float32, tl.float16]
  | .fp16, .fp32, .bf16 => klForward%[tl.float16, tl.float32, tl.bfloat16]
  | .fp16, .fp32, .f8e4 => klForward%[tl.float16, tl.float32, tl.float8e4nv]
  | .fp16, .fp32, .f8e5 => klForward%[tl.float16, tl.float32, tl.float8e5]
  | .fp16, .fp64, .fp32 => klForward%[tl.float16, tl.float64, tl.float32]
  | .fp16, .fp64, .fp64 => klForward%[tl.float16, tl.float64, tl.float64]
  | .fp16, .fp64, .fp16 => klForward%[tl.float16, tl.float64, tl.float16]
  | .fp16, .fp64, .bf16 => klForward%[tl.float16, tl.float64, tl.bfloat16]
  | .fp16, .fp64, .f8e4 => klForward%[tl.float16, tl.float64, tl.float8e4nv]
  | .fp16, .fp64, .f8e5 => klForward%[tl.float16, tl.float64, tl.float8e5]
  | .bf16, .fp32, .fp32 => klForward%[tl.bfloat16, tl.float32, tl.float32]
  | .bf16, .fp32, .fp64 => klForward%[tl.bfloat16, tl.float32, tl.float64]
  | .bf16, .fp32, .fp16 => klForward%[tl.bfloat16, tl.float32, tl.float16]
  | .bf16, .fp32, .bf16 => klForward%[tl.bfloat16, tl.float32, tl.bfloat16]
  | .bf16, .fp32, .f8e4 => klForward%[tl.bfloat16, tl.float32, tl.float8e4nv]
  | .bf16, .fp32, .f8e5 => klForward%[tl.bfloat16, tl.float32, tl.float8e5]
  | .bf16, .fp64, .fp32 => klForward%[tl.bfloat16, tl.float64, tl.float32]
  | .bf16, .fp64, .fp64 => klForward%[tl.bfloat16, tl.float64, tl.float64]
  | .bf16, .fp64, .fp16 => klForward%[tl.bfloat16, tl.float64, tl.float16]
  | .bf16, .fp64, .bf16 => klForward%[tl.bfloat16, tl.float64, tl.bfloat16]
  | .bf16, .fp64, .f8e4 => klForward%[tl.bfloat16, tl.float64, tl.float8e4nv]
  | .bf16, .fp64, .f8e5 => klForward%[tl.bfloat16, tl.float64, tl.float8e5]
  | .f8e4, .fp32, .fp32 => klForward%[tl.float8e4nv, tl.float32, tl.float32]
  | .f8e4, .fp32, .fp64 => klForward%[tl.float8e4nv, tl.float32, tl.float64]
  | .f8e4, .fp32, .fp16 => klForward%[tl.float8e4nv, tl.float32, tl.float16]
  | .f8e4, .fp32, .bf16 => klForward%[tl.float8e4nv, tl.float32, tl.bfloat16]
  | .f8e4, .fp32, .f8e4 => klForward%[tl.float8e4nv, tl.float32, tl.float8e4nv]
  | .f8e4, .fp32, .f8e5 => klForward%[tl.float8e4nv, tl.float32, tl.float8e5]
  | .f8e4, .fp64, .fp32 => klForward%[tl.float8e4nv, tl.float64, tl.float32]
  | .f8e4, .fp64, .fp64 => klForward%[tl.float8e4nv, tl.float64, tl.float64]
  | .f8e4, .fp64, .fp16 => klForward%[tl.float8e4nv, tl.float64, tl.float16]
  | .f8e4, .fp64, .bf16 => klForward%[tl.float8e4nv, tl.float64, tl.bfloat16]
  | .f8e4, .fp64, .f8e4 => klForward%[tl.float8e4nv, tl.float64, tl.float8e4nv]
  | .f8e4, .fp64, .f8e5 => klForward%[tl.float8e4nv, tl.float64, tl.float8e5]
  | .f8e5, .fp32, .fp32 => klForward%[tl.float8e5, tl.float32, tl.float32]
  | .f8e5, .fp32, .fp64 => klForward%[tl.float8e5, tl.float32, tl.float64]
  | .f8e5, .fp32, .fp16 => klForward%[tl.float8e5, tl.float32, tl.float16]
  | .f8e5, .fp32, .bf16 => klForward%[tl.float8e5, tl.float32, tl.bfloat16]
  | .f8e5, .fp32, .f8e4 => klForward%[tl.float8e5, tl.float32, tl.float8e4nv]
  | .f8e5, .fp32, .f8e5 => klForward%[tl.float8e5, tl.float32, tl.float8e5]
  | .f8e5, .fp64, .fp32 => klForward%[tl.float8e5, tl.float64, tl.float32]
  | .f8e5, .fp64, .fp64 => klForward%[tl.float8e5, tl.float64, tl.float64]
  | .f8e5, .fp64, .fp16 => klForward%[tl.float8e5, tl.float64, tl.float16]
  | .f8e5, .fp64, .bf16 => klForward%[tl.float8e5, tl.float64, tl.bfloat16]
  | .f8e5, .fp64, .f8e4 => klForward%[tl.float8e5, tl.float64, tl.float8e4nv]
  | .f8e5, .fp64, .f8e5 => klForward%[tl.float8e5, tl.float64, tl.float8e5]
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate changes only the order of the exact signed row-address product. -/
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

## Public theorem: `backward_default_source_equivalence`

<details><summary>docstring</summary>

```
/-- Source-facing default backward equivalence retains the pinned frontend's
nonzero-literal restriction. The broader interpreter lemma above also covers
abstract fp8 programs, without certifying that their Python source compiles. -/
```
</details>

**Statement:**
```lean
specification backward_default_source_equivalence (target output : Storage)
    (T O : RegionName) (st so N B : Nat) (_hSource : BackwardSourceSupported target) :
    backwardDefaultProgram target output Bool.false T O st so N B ≡[assumptions]
      backwardDefaultProgram target output Bool.true T O st so N B
```

**Closed-form spec defs (transitive):** `backwardDefaultProgram`, `assumptions`, `backwardDefault`, `backwardDefaultOriginal`, `commuteOutput`

<details><summary><code>backwardDefaultProgram</code></summary>

```lean
def backwardDefaultProgram (target output : Storage) (v : Bool)
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
def backwardDefault (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  let source := backwardDefaultOriginal target output T O st so N B
  if v then commuteOutput source 1 O so else source
```
</details>

<details><summary><code>backwardDefaultOriginal</code></summary>

```
/-- Default-target backward keeps the weak scalar multiplication by -1. -/
```
```lean
def backwardDefaultOriginal (target output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klBackward%[tl.float32, tl.float32, defaultTarget]
  | .fp32, .fp64 => klBackward%[tl.float32, tl.float64, defaultTarget]
  | .fp32, .fp16 => klBackward%[tl.float32, tl.float16, defaultTarget]
  | .fp32, .bf16 => klBackward%[tl.float32, tl.bfloat16, defaultTarget]
  | .fp32, .f8e4 => klBackward%[tl.float32, tl.float8e4nv, defaultTarget]
  | .fp32, .f8e5 => klBackward%[tl.float32, tl.float8e5, defaultTarget]
  | .fp64, .fp32 => klBackward%[tl.float64, tl.float32, defaultTarget]
  | .fp64, .fp64 => klBackward%[tl.float64, tl.float64, defaultTarget]
  | .fp64, .fp16 => klBackward%[tl.float64, tl.float16, defaultTarget]
  | .fp64, .bf16 => klBackward%[tl.float64, tl.bfloat16, defaultTarget]
  | .fp64, .f8e4 => klBackward%[tl.float64, tl.float8e4nv, defaultTarget]
  | .fp64, .f8e5 => klBackward%[tl.float64, tl.float8e5, defaultTarget]
  | .fp16, .fp32 => klBackward%[tl.float16, tl.float32, defaultTarget]
  | .fp16, .fp64 => klBackward%[tl.float16, tl.float64, defaultTarget]
  | .fp16, .fp16 => klBackward%[tl.float16, tl.float16, defaultTarget]
  | .fp16, .bf16 => klBackward%[tl.float16, tl.bfloat16, defaultTarget]
  | .fp16, .f8e4 => klBackward%[tl.float16, tl.float8e4nv, defaultTarget]
  | .fp16, .f8e5 => klBackward%[tl.float16, tl.float8e5, defaultTarget]
  | .bf16, .fp32 => klBackward%[tl.bfloat16, tl.float32, defaultTarget]
  | .bf16, .fp64 => klBackward%[tl.bfloat16, tl.float64, defaultTarget]
  | .bf16, .fp16 => klBackward%[tl.bfloat16, tl.float16, defaultTarget]
  | .bf16, .bf16 => klBackward%[tl.bfloat16, tl.bfloat16, defaultTarget]
  | .bf16, .f8e4 => klBackward%[tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .bf16, .f8e5 => klBackward%[tl.bfloat16, tl.float8e5, defaultTarget]
  | .f8e4, .fp32 => klBackward%[tl.float8e4nv, tl.float32, defaultTarget]
  | .f8e4, .fp64 => klBackward%[tl.float8e4nv, tl.float64, defaultTarget]
  | .f8e4, .fp16 => klBackward%[tl.float8e4nv, tl.float16, defaultTarget]
  | .f8e4, .bf16 => klBackward%[tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4, .f8e4 => klBackward%[tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e4, .f8e5 => klBackward%[tl.float8e4nv, tl.float8e5, defaultTarget]
  | .f8e5, .fp32 => klBackward%[tl.float8e5, tl.float32, defaultTarget]
  | .f8e5, .fp64 => klBackward%[tl.float8e5, tl.float64, defaultTarget]
  | .f8e5, .fp16 => klBackward%[tl.float8e5, tl.float16, defaultTarget]
  | .f8e5, .bf16 => klBackward%[tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e5, .f8e4 => klBackward%[tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5, .f8e5 => klBackward%[tl.float8e5, tl.float8e5, defaultTarget]
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate changes only the order of the exact signed row-address product. -/
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
/-- Complete log-target backward source; the only rewrite is signed address multiplication. -/
```
</details>

**Statement:**
```lean
specification backward_log_equivalence (target : MathStorage) (output : Storage)
    (T O : RegionName) (st so N B : Nat) :
    backwardLogProgram target output Bool.false T O st so N B ≡[assumptions]
      backwardLogProgram target output Bool.true T O st so N B
```

**Closed-form spec defs (transitive):** `backwardLogProgram`, `assumptions`, `backwardLog`, `backwardLogOriginal`, `commuteOutput`

<details><summary><code>backwardLogProgram</code></summary>

```lean
def backwardLogProgram (target : MathStorage) (output : Storage) (v : Bool)
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
  if v then commuteOutput source 1 O so else source
```
</details>

<details><summary><code>backwardLogOriginal</code></summary>

```
/-- Log-target backward retains the raw exp input dtype and negation order. -/
```
```lean
def backwardLogOriginal (target : MathStorage) (output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klBackward%[tl.float32, tl.float32, logarithmic]
  | .fp32, .fp64 => klBackward%[tl.float32, tl.float64, logarithmic]
  | .fp32, .fp16 => klBackward%[tl.float32, tl.float16, logarithmic]
  | .fp32, .bf16 => klBackward%[tl.float32, tl.bfloat16, logarithmic]
  | .fp32, .f8e4 => klBackward%[tl.float32, tl.float8e4nv, logarithmic]
  | .fp32, .f8e5 => klBackward%[tl.float32, tl.float8e5, logarithmic]
  | .fp64, .fp32 => klBackward%[tl.float64, tl.float32, logarithmic]
  | .fp64, .fp64 => klBackward%[tl.float64, tl.float64, logarithmic]
  | .fp64, .fp16 => klBackward%[tl.float64, tl.float16, logarithmic]
  | .fp64, .bf16 => klBackward%[tl.float64, tl.bfloat16, logarithmic]
  | .fp64, .f8e4 => klBackward%[tl.float64, tl.float8e4nv, logarithmic]
  | .fp64, .f8e5 => klBackward%[tl.float64, tl.float8e5, logarithmic]
```
</details>

<details><summary><code>commuteOutput</code></summary>

```
/-- Candidate changes only the order of the exact signed row-address product. -/
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
