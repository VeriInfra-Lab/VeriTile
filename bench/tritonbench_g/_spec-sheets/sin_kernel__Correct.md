# Spec sheet — `bench/tritonbench_g/sin_kernel/Correct.lean`

**Python source:** `bench/tritonbench_g/sin_kernel/sin_kernel.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Independent Real.sin formula for every active lane of either complete typed
source program, including successful flat execution, active bounds and full frame. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    VeriTile.Spec.Real (typedIO input v X Y N B ⊨ fun xs i => Real.sin (xs i))
```

**Closed-form spec defs (transitive):** `typedIO`, `typedKernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : MaskedKernelIO₁ :=
  { sinIO X Y N B with
    kernel := (typedKernel input v X Y N B).eraseDType
    projection := typed_supported input v X Y N B }
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
