# Spec sheet — `bench/tritonbench_g/fused_activation/Correct.lean`

**Python source:** `bench/tritonbench_g/fused_activation/fused_activation.py`

## Public theorem: `original_real_correct`

<details><summary>docstring</summary>

```
/-- Independent real formula for the original, with full grouped flat-memory IO. -/
```
</details>

**Statement:**
```lean
specification original_real_correct (x b i : RegionName) (W N B : Nat)
    (m : ℝ) (a : Bool) (_hW : 0 < W) :
    VeriTile.Spec.Real (fusedActivationIO x b i W N B m a
      ⊨ fun _ _ xs _ j => activationFormula a
        (xs (⟨0, by decide⟩ : Fin 3) j) (xs (⟨1, by decide⟩ : Fin 3) j)
        (xs (⟨2, by decide⟩ : Fin 3) j) m)
```

**Assumptions / layout contracts:**
- `_hW : 0 < W`

**Closed-form spec defs (transitive):** `fusedActivationIO`, `activationFormula`, `fused_add_mul_activation_kernel`

<details><summary><code>fusedActivationIO</code></summary>

```
/-- `fused_add_mul_activation_kernel`'s grouped masked **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`GroupedMasked2DKernelIO`, the vector-channel genre; the general per-lane
windows are what the broadcast bias read needs):

* `bufs = [x_ptr, bias_ptr, in_ptr]` — every buffer once; the output channel
  names `x_ptr` again, i.e. the update is **in place**;
* `nIn = 3` — channel 0 = `x_ptr` at `pid₀·BLOCK_SIZE + j`, channel 1 =
  `bias_ptr` at `(pid₀·BLOCK_SIZE + j) % num_weights` (the broadcast bias),
  channel 2 = `in_ptr` at `pid₀·BLOCK_SIZE + j`;
* `nOut = 1` — `x_ptr` at `pid₀·BLOCK_SIZE + j`;
* every read/write gate is the kernel's single mask `pid₀·BLOCK_SIZE + j <
  xnumel`;
* `B = BLOCK_SIZE`; the kernel is a 1-D launch, so the family's second program
  id is ignored by every field.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the declared lanes. -/
```
```lean
def fusedActivationIO
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) : GroupedMasked2DKernelIO where
  kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights
    xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
  nIn := 3
  nOut := 1
  bufs := [x_ptr, bias_ptr, in_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => x_ptr
    | ⟨1, _⟩ => bias_ptr
    | ⟨_ + 2, _⟩ => in_ptr
  out := fun _ => x_ptr
  B := BLOCK_SIZE
  read := fun i pid₀ _ j => match i with
    | ⟨0, _⟩ => pid₀ * BLOCK_SIZE + j.val
    | ⟨1, _⟩ => (pid₀ * BLOCK_SIZE + j.val) % num_weights
    | ⟨_ + 2, _⟩ => pid₀ * BLOCK_SIZE + j.val
  readMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
  write := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  writeMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
```
</details>

<details><summary><code>activationFormula</code></summary>

```
/-- Mathematical scalar operation, independent of either source program. -/
```
```lean
noncomputable def activationFormula (a : Bool) (x bias input multiplier : ℝ) : ℝ :=
  let z := multiplier * input + x + bias
  if a then Real.sigmoid z else max 0 z
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

## Public theorem: `commuted_real_correct`

<details><summary>docstring</summary>

```
/-- The candidate implements the same mathematical sigmoid/ReLU formula.
The proof uses real addition commutativity, independently of FP rules. -/
```
</details>

**Statement:**
```lean
specification commuted_real_correct (x b i : RegionName) (W N B : Nat)
    (m : ℝ) (a : Bool) (_hW : 0 < W) :
    VeriTile.Spec.Real (commutedIO x b i W N B m a
      ⊨ fun _ _ xs _ j => activationFormula a
        (xs (⟨0, by decide⟩ : Fin 3) j) (xs (⟨1, by decide⟩ : Fin 3) j)
        (xs (⟨2, by decide⟩ : Fin 3) j) m)
```

**Assumptions / layout contracts:**
- `_hW : 0 < W`

**Closed-form spec defs (transitive):** `commutedIO`, `activationFormula`, `fusedActivationIO`, `fused_add_mul_activation_commuted_kernel`, `fused_add_mul_activation_kernel`

<details><summary><code>commutedIO</code></summary>

```
/-- The candidate retains all three read windows and the in-place write window. -/
```
```lean
def commutedIO (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) :
    GroupedMasked2DKernelIO :=
  { fusedActivationIO x b i W N B m a with
    kernel := fused_add_mul_activation_commuted_kernel x b i W N B m a
    projection := by cases a <;> rfl }
```
</details>

<details><summary><code>activationFormula</code></summary>

```
/-- Mathematical scalar operation, independent of either source program. -/
```
```lean
noncomputable def activationFormula (a : Bool) (x bias input multiplier : ℝ) : ℝ :=
  let z := multiplier * input + x + bias
  if a then Real.sigmoid z else max 0 z
```
</details>

<details><summary><code>fusedActivationIO</code></summary>

```
/-- `fused_add_mul_activation_kernel`'s grouped masked **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`GroupedMasked2DKernelIO`, the vector-channel genre; the general per-lane
windows are what the broadcast bias read needs):

* `bufs = [x_ptr, bias_ptr, in_ptr]` — every buffer once; the output channel
  names `x_ptr` again, i.e. the update is **in place**;
* `nIn = 3` — channel 0 = `x_ptr` at `pid₀·BLOCK_SIZE + j`, channel 1 =
  `bias_ptr` at `(pid₀·BLOCK_SIZE + j) % num_weights` (the broadcast bias),
  channel 2 = `in_ptr` at `pid₀·BLOCK_SIZE + j`;
* `nOut = 1` — `x_ptr` at `pid₀·BLOCK_SIZE + j`;
* every read/write gate is the kernel's single mask `pid₀·BLOCK_SIZE + j <
  xnumel`;
* `B = BLOCK_SIZE`; the kernel is a 1-D launch, so the family's second program
  id is ignored by every field.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the declared lanes. -/
```
```lean
def fusedActivationIO
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (ACTIVATION_SIGMOID : Bool) : GroupedMasked2DKernelIO where
  kernel := fused_add_mul_activation_kernel x_ptr bias_ptr in_ptr num_weights
    xnumel BLOCK_SIZE multiplier ACTIVATION_SIGMOID
  nIn := 3
  nOut := 1
  bufs := [x_ptr, bias_ptr, in_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => x_ptr
    | ⟨1, _⟩ => bias_ptr
    | ⟨_ + 2, _⟩ => in_ptr
  out := fun _ => x_ptr
  B := BLOCK_SIZE
  read := fun i pid₀ _ j => match i with
    | ⟨0, _⟩ => pid₀ * BLOCK_SIZE + j.val
    | ⟨1, _⟩ => (pid₀ * BLOCK_SIZE + j.val) % num_weights
    | ⟨_ + 2, _⟩ => pid₀ * BLOCK_SIZE + j.val
  readMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
  write := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val
  writeMask := fun _ pid₀ _ j => pid₀ * BLOCK_SIZE + j.val < xnumel
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
