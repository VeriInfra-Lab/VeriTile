# Spec sheet — `bench/tritonbench_g/fused_activation/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/fused_activation/fused_activation.py`

## Public theorem: `fused_activation_equiv`

**Statement:**
```lean
specification fused_activation_equiv (x b i : RegionName) (W N B : Nat)
    (m : ℝ) (a : Bool) (_hW : 0 < W) (R : Rules) :
    original x b i W N B m a ≡[R] commuted x b i W N B m a
```

**Assumptions / layout contracts:**
- `_hW : 0 < W`

**Closed-form spec defs (transitive):** `Rules`, `original`, `commuted`, `fused_add_mul_activation_kernel`, `additionSite`, `fused_add_mul_activation_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) : Program where
  kernel := fused_add_mul_activation_kernel x b i W N B m a
  domain := [additionSite x b i W N B m a]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) : Program :=
  { original x b i W N B m a with
    kernel := fused_add_mul_activation_commuted_kernel x b i W N B m a }
```
</details>

<details><summary><code>fused_add_mul_activation_kernel</code></summary>

```
/-- Faithful transcription of `fused_activation.py`'s
`fused_add_mul_activation_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `num_weights/xnumel/multiplier/activation/BLOCK_SIZE: tl.constexpr` ->
  Lean parameters.
- Python `activation == "sigmoid"` / `elif activation == "relu"` -> Lean
  Boolean `activation`; `true` selects sigmoid and `false` selects ReLU. -/
```
```lean
def fused_add_mul_activation_kernel
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (activation : Bool) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(BLOCK_SIZE)
  index = xoffset + tl.arange(0, $(BLOCK_SIZE))[:]
  mask = index < $(xnumel)
  bias_index = index % $(num_weights)
  tmp0 = tl.load(x_ptr + index, mask)
  tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy="evict_last")
  tmp3 = tl.load(in_ptr + index, mask)
  activ_input = $(multiplier) * tmp3 + tmp0 + tmp1
  if activation {
    ma_result = tl.sigmoid(activ_input)
  } else {
    ma_result = tl.maximum(0, activ_input)
  }
  tl.store(x_ptr + index, ma_result, mask)
}
```
</details>

<details><summary><code>additionSite</code></summary>

```
/-- The actual combined input (multiplier * input + x) and bias must be finite.
This includes inactive lanes since arithmetic executes before the masked store.
No input distribution, positivity or nonzero restriction is required. -/
```
```lean
def additionSite (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) : Site :=
  .add (Source.prefixBefore
    (fused_add_mul_activation_kernel x b i W N B m a).surfaceBody "activ_input")
    [B] (.add .real (.consSame .nil)
      (.mul .real .scalarL (.const m) (.ref .real [B] "tmp3"))
      (.ref .real [B] "tmp0")) (.ref .real [B] "tmp1")
```
</details>

<details><summary><code>fused_add_mul_activation_commuted_kernel</code></summary>

```
/-- Commute the combined input and the loaded bias, retaining all memory operations. -/
```
```lean
def fused_add_mul_activation_commuted_kernel
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (activation : Bool) :
    ComputeKernel := triton {
  xoffset = tl.program_id(0) * $(BLOCK_SIZE)
  index = xoffset + tl.arange(0, $(BLOCK_SIZE))[:]
  mask = index < $(xnumel)
  bias_index = index % $(num_weights)
  tmp0 = tl.load(x_ptr + index, mask)
  tmp1 = tl.load(bias_ptr + bias_index, mask, eviction_policy="evict_last")
  tmp3 = tl.load(in_ptr + index, mask)
  activ_input = tmp1 + ($(multiplier) * tmp3 + tmp0)
  if activation {
    ma_result = tl.sigmoid(activ_input)
  } else {
    ma_result = tl.maximum(0, activ_input)
  }
  tl.store(x_ptr + index, ma_result, mask)
}
```
</details>
