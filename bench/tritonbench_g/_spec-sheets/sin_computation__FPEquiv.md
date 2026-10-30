# Spec sheet — `bench/tritonbench_g/sin_computation/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/sin_computation/sin_computation.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Full floating execution equivalence from arbitrary states: same signature,
registers, memory and failure behavior; no numerical atom or domain restriction. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (input : InputStorage) (output : OutputStorage)
    (X Y : RegionName) (N B : Nat) :
    program input output Bool.false X Y N B ≡[assumptions]
      program input output Bool.true X Y N B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`

<details><summary><code>program</code></summary>

```lean
def program (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : Program where
  kernel := typedKernel input output v X Y N B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>typedKernel</code></summary>

```
/-- Complete typed source and its natural-index commutation candidate. Explicit
store casts expose Triton's implicit conversion to the output pointer dtype. -/
```
```lean
def typedKernel (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp32, .fp32, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp32, .fp64, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float64
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float64), mask=mask)
    }).surfaceBody
  | .fp32, .fp64, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float64
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float64), mask=mask)
    }).surfaceBody
  | .fp32, .fp16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp32, .fp16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp32, .bf16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp32, .bf16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp32, .f8e4, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp32, .f8e4, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp32, .f8e5, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody
  | .fp32, .f8e5, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody
  | .fp64, .fp32, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float32
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float32), mask=mask)
    }).surfaceBody
  | .fp64, .fp32, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float32
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float32), mask=mask)
    }).surfaceBody
  | .fp64, .fp64, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, .fp64, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, .fp16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp64, .fp16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp64, .bf16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp64, .bf16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp64, .f8e4, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp64, .f8e4, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp64, .f8e5, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody
  | .fp64, .f8e5, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody
```
</details>
