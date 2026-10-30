# Spec sheet — `bench/tritonbench_g/relu_triton_kernel/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py`

## Public theorem: `relu_fused_equiv`

**Statement:**
```lean
specification relu_fused_equiv (x y : RegionName) (N B : Nat) (_hB : 0 < B) :
    original x y N B ≡[assumptions] fused x y N B
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`

**Closed-form spec defs (transitive):** `original`, `assumptions`, `fused`, `originalIO`, `fusedIO`, `relu_kernel`, `relu_fused_kernel`

<details><summary><code>original</code></summary>

```lean
def original (x y : RegionName) (N B : Nat) : FP.Kernel₁ where
  io := originalIO x y N B
  profile := FP.fp32
  domain := fun _ => .top
  comparisons := [.le .fp32 .real]
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>fused</code></summary>

```lean
def fused (x y : RegionName) (N B : Nat) : FP.Kernel₁ :=
  { original x y N B with io := fusedIO x y N B }
```
</details>

<details><summary><code>originalIO</code></summary>

```lean
def originalIO (x y : RegionName) (N B : Nat) : KernelIO₁ where
  kernel := relu_kernel x y N B
  inp := x
  out := y
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B
```
</details>

<details><summary><code>fusedIO</code></summary>

```lean
def fusedIO (x y : RegionName) (N B : Nat) : KernelIO₁ :=
  (originalIO x y N B).withKernel (relu_fused_kernel x y N B)
```
</details>

<details><summary><code>relu_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `relu_triton_kernel.py`'s `relu_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N: tl.constexpr` / `block_size: tl.constexpr` → Lean `Nat`.
- Python `if cond: body` → `if cond { body }`, the DSL-side gate (block syntax
  uses braces because Lean is whitespace-insensitive; the `if` keyword itself
  is shared). -/
```
```lean
def relu_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  result = tl.where(x >= 0, x, 0.0)
  if pid == 0 {
    tl.store(out_ptr + offsets, result, mask=mask)
  }
}
```
</details>

<details><summary><code>relu_fused_kernel</code></summary>

```
/-- Inline ReLU into the guarded store, avoiding work for nonzero programs. -/
```
```lean
def relu_fused_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  if pid == 0 {
    tl.store(out_ptr + offsets, tl.where(x >= 0, x, 0.0), mask=mask)
  }
}
```
</details>
