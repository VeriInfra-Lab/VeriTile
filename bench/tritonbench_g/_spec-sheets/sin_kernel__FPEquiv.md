# Spec sheet — `bench/tritonbench_g/sin_kernel/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/sin_kernel/sin_kernel.py`

## Public theorem: `source_equivalence`

<details><summary>docstring</summary>

```
/-- Full floating execution equivalence from arbitrary states: same signature,
registers, memory and failure behavior; no numerical atom or domain restriction. -/
```
</details>

**Statement:**
```lean
specification source_equivalence (input : InputStorage)
    (X Y : RegionName) (N B : Nat) :
    program input Bool.false X Y N B ≡[assumptions]
      program input Bool.true X Y N B
```

**Closed-form spec defs (transitive):** `program`, `assumptions`, `typedKernel`

<details><summary><code>program</code></summary>

```lean
def program (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : Program where
  kernel := typedKernel input v X Y N B
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
/-- Complete typed source and its natural-index commutation candidate. The
wrapper's empty_like output has the same dtype as its input. -/
```
```lean
def typedKernel (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  match input, v with
  | .fp32, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp32, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
```
</details>
