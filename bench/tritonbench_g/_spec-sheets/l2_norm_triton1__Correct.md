# Spec sheet — `bench/tritonbench_g/l2_norm_triton1/Correct.lean`

**Python source:** `bench/tritonbench_g/l2_norm_triton1/l2_norm_triton1.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Complete real execution of either typed program, with exact active-lane
bounds, success and the full flat-memory frame. No positivity restriction on
epsilon is needed for Lean's total real sqrt/division semantics. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat)
    (_hB : 0 < B) (_hN : N ≤ B) :
    Spec.Real (typedIO storage commuted X Y stride N eps B ⊨ rowFormula N B eps)
```

**Assumptions / layout contracts:**
- `_hB : 0 < B`
- `_hN : N ≤ B`

**Closed-form spec defs (transitive):** `typedIO`, `l2NormIO`, `typedKernel`, `l2_norm_fwd_1pass_kernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (storage : Storage) (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₁ :=
  { l2NormIO X Y stride N eps B with
    kernel := (typedKernel storage commuted X Y stride N eps B).eraseDType
    projection := typed_supported storage commuted X Y stride N eps B }
```
</details>

<details><summary><code>l2NormIO</code></summary>

```
/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `row` reads its row of `X` and writes its row of
  `Y`, both at `row * stride_x_row` (the host passes the same row stride for
  both buffers; the one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))`) carry no obligations on
  either side. The store mask equals the load mask, so `writeMask` keeps its
  default.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def l2NormIO (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₁ where
  kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N
  inp := X
  out := Y
  B := BLOCK_N
  read := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, commuted with
  | .fp32, Bool.false => l2Source%[tl.float32, original] X Y stride_x_row N eps BLOCK_N
  | .fp32, Bool.true => l2Source%[tl.float32, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.false => l2Source%[tl.float16, original] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.true => l2Source%[tl.float16, commuted] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.false => l2Source%[tl.bfloat16, original] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.true => l2Source%[tl.bfloat16, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.false => l2Source%[tl.float64, original] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.true => l2Source%[tl.float64, commuted] X Y stride_x_row N eps BLOCK_N

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>l2_norm_fwd_1pass_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

/-! Typed full-source family. The runtime epsilon is fp32. The explicit output
cast spells out the conversion performed by tl.store after pointer rebinding.
The candidate commutes only the two natural-number row-base products. -/
open Lean
syntax "l2Source%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(l2Source%[$dtype:tritonDType, $version:ident]) => do
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += row * $(stride_x_row))
      else `(tritonStmt| X += $(stride_x_row) * row)
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += row * $(stride_x_row))
      else `(tritonStmt| Y += $(stride_x_row) * row)
    let write : TSyntax `tritonStmt ← match dtype with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Y + cols, y, mask=mask))
      | _ => `(tritonStmt| tl.store(Y + cols, y.to($dtype), mask=mask))
    `(fun (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) =>
      ComputeKernel.mk [X] [Y] (triton {
  tl.region X = $dtype, Y = $dtype
  row = tl.program_id(0)
  $xbase
  $ybase
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  mask = cols < $(N)
  y = x * rstd
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr
```
</details>

## Public theorem: `original_real_correct`

**Statement:**
```lean
specification original_real_correct (storage : Storage) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (hB : 0 < B) (hN : N ≤ B) :
    Spec.Real (typedIO storage Bool.false X Y stride N eps B ⊨ rowFormula N B eps)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hN : N ≤ B`

**Closed-form spec defs (transitive):** `typedIO`, `l2NormIO`, `typedKernel`, `l2_norm_fwd_1pass_kernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (storage : Storage) (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₁ :=
  { l2NormIO X Y stride N eps B with
    kernel := (typedKernel storage commuted X Y stride N eps B).eraseDType
    projection := typed_supported storage commuted X Y stride N eps B }
```
</details>

<details><summary><code>l2NormIO</code></summary>

```
/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `row` reads its row of `X` and writes its row of
  `Y`, both at `row * stride_x_row` (the host passes the same row stride for
  both buffers; the one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))`) carry no obligations on
  either side. The store mask equals the load mask, so `writeMask` keeps its
  default.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def l2NormIO (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₁ where
  kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N
  inp := X
  out := Y
  B := BLOCK_N
  read := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, commuted with
  | .fp32, Bool.false => l2Source%[tl.float32, original] X Y stride_x_row N eps BLOCK_N
  | .fp32, Bool.true => l2Source%[tl.float32, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.false => l2Source%[tl.float16, original] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.true => l2Source%[tl.float16, commuted] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.false => l2Source%[tl.bfloat16, original] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.true => l2Source%[tl.bfloat16, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.false => l2Source%[tl.float64, original] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.true => l2Source%[tl.float64, commuted] X Y stride_x_row N eps BLOCK_N

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>l2_norm_fwd_1pass_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

/-! Typed full-source family. The runtime epsilon is fp32. The explicit output
cast spells out the conversion performed by tl.store after pointer rebinding.
The candidate commutes only the two natural-number row-base products. -/
open Lean
syntax "l2Source%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(l2Source%[$dtype:tritonDType, $version:ident]) => do
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += row * $(stride_x_row))
      else `(tritonStmt| X += $(stride_x_row) * row)
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += row * $(stride_x_row))
      else `(tritonStmt| Y += $(stride_x_row) * row)
    let write : TSyntax `tritonStmt ← match dtype with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Y + cols, y, mask=mask))
      | _ => `(tritonStmt| tl.store(Y + cols, y.to($dtype), mask=mask))
    `(fun (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) =>
      ComputeKernel.mk [X] [Y] (triton {
  tl.region X = $dtype, Y = $dtype
  row = tl.program_id(0)
  $xbase
  $ybase
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  mask = cols < $(N)
  y = x * rstd
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr
```
</details>

## Public theorem: `candidate_real_correct`

**Statement:**
```lean
specification candidate_real_correct (storage : Storage) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (hB : 0 < B) (hN : N ≤ B) :
    Spec.Real (typedIO storage Bool.true X Y stride N eps B ⊨ rowFormula N B eps)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hN : N ≤ B`

**Closed-form spec defs (transitive):** `typedIO`, `l2NormIO`, `typedKernel`, `l2_norm_fwd_1pass_kernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (storage : Storage) (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₁ :=
  { l2NormIO X Y stride N eps B with
    kernel := (typedKernel storage commuted X Y stride N eps B).eraseDType
    projection := typed_supported storage commuted X Y stride N eps B }
```
</details>

<details><summary><code>l2NormIO</code></summary>

```
/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `row` reads its row of `X` and writes its row of
  `Y`, both at `row * stride_x_row` (the host passes the same row stride for
  both buffers; the one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))`) carry no obligations on
  either side. The store mask equals the load mask, so `writeMask` keeps its
  default.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def l2NormIO (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₁ where
  kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N
  inp := X
  out := Y
  B := BLOCK_N
  read := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, commuted with
  | .fp32, Bool.false => l2Source%[tl.float32, original] X Y stride_x_row N eps BLOCK_N
  | .fp32, Bool.true => l2Source%[tl.float32, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.false => l2Source%[tl.float16, original] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.true => l2Source%[tl.float16, commuted] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.false => l2Source%[tl.bfloat16, original] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.true => l2Source%[tl.bfloat16, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.false => l2Source%[tl.float64, original] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.true => l2Source%[tl.float64, commuted] X Y stride_x_row N eps BLOCK_N

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>l2_norm_fwd_1pass_kernel</code></summary>

```
/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

/-! Typed full-source family. The runtime epsilon is fp32. The explicit output
cast spells out the conversion performed by tl.store after pointer rebinding.
The candidate commutes only the two natural-number row-base products. -/
open Lean
syntax "l2Source%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(l2Source%[$dtype:tritonDType, $version:ident]) => do
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += row * $(stride_x_row))
      else `(tritonStmt| X += $(stride_x_row) * row)
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += row * $(stride_x_row))
      else `(tritonStmt| Y += $(stride_x_row) * row)
    let write : TSyntax `tritonStmt ← match dtype with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Y + cols, y, mask=mask))
      | _ => `(tritonStmt| tl.store(Y + cols, y.to($dtype), mask=mask))
    `(fun (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) =>
      ComputeKernel.mk [X] [Y] (triton {
  tl.region X = $dtype, Y = $dtype
  row = tl.program_id(0)
  $xbase
  $ybase
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  mask = cols < $(N)
  y = x * rstd
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr
```
</details>
