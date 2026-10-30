# Spec sheet — `bench/tritonbench_g/rms_norm_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/rms_norm_triton/rms_norm_triton.py`

## Public theorem: `typed_real_correct`

<details><summary>docstring</summary>

```
/-- Complete strided-row real execution and flat memory frame for either typed
source. The positive row length and complete tile describe a nonempty wrapper
row; the historical empty-tile theorem above is preserved unchanged. -/
```
</details>

**Statement:**
```lean
specification typed_real_correct (storage weight : Storage) (commuted : Bool)
    (Y X W : RegionName) (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat)
    (_hN : 0 < N) (_hNB : N ≤ B) (hyc : 0 < ysc) :
    Spec.Real (typedIO storage weight commuted Y X W ysr ysc xsr xsc N eps B ⊨
      fun _ _ xs ws i => rowFormula N B eps xs ws i)
```

**Assumptions / layout contracts:**
- `_hN : 0 < N`
- `_hNB : N ≤ B`
- `hyc : 0 < ysc`

**Closed-form spec defs (transitive):** `typedIO`, `rmsNormIO`, `typed`, `rms_norm_kernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (storage weight : Storage) (commuted : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) : Masked2DKernelIO₂ :=
  { rmsNormIO Y X W ysr ysc xsr xsc N eps B with
    kernel := (typed storage weight commuted Y X W ysr ysc xsr xsc N eps B).eraseDType
    projection := typed_supported storage weight commuted Y X W ysr ysc xsr xsc N eps B }
```
</details>

<details><summary><code>rmsNormIO</code></summary>

```
/-- `rms_norm_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring): the input
  matrix `X`, the per-column weights `W`, the output matrix `Y`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write` — **per-lane strided windows**: program `pid` reads its `X`
  row at `pid * x_stride_r + j * x_stride_c` and writes its `Y` row at
  `pid * y_stride_r + j * y_stride_c` (the host-side one-program-per-row
  launch convention with general row/column strides);
* `read2` — the weight window is **absolute and pid-independent**: every
  program reads `W[j]` (the host passes the same weight vector to all rows);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(N)`) carry no obligations on either side. The
  store mask equals the load mask, so `writeMask` keeps its default.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and mask are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and mask are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def rmsNormIO (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ where
  kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
    N eps BLOCK_SIZE
  in1 := X
  in2 := W
  out := Y
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * x_stride_r + j.val * x_stride_c
  read2 := fun _ _ j => j.val
  write := fun pid _ j => pid * y_stride_r + j.val * y_stride_c
  mask := fun _ _ j => j.val < N
```
</details>

<details><summary><code>typed</code></summary>

```lean
def typed (storage weight : Storage) (commuted : Bool)
    (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, weight, commuted with
  | .fp32, .fp32, Bool.false =>
    rmsTyped%[tl.float32, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    rmsTyped%[tl.float32, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    rmsTyped%[tl.float32, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    rmsTyped%[tl.float32, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    rmsTyped%[tl.float32, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    rmsTyped%[tl.float32, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    rmsTyped%[tl.float32, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    rmsTyped%[tl.float32, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    rmsTyped%[tl.float16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    rmsTyped%[tl.float16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    rmsTyped%[tl.float16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    rmsTyped%[tl.float16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    rmsTyped%[tl.float16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    rmsTyped%[tl.float16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    rmsTyped%[tl.float16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    rmsTyped%[tl.float16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    rmsTyped%[tl.float64, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    rmsTyped%[tl.float64, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    rmsTyped%[tl.float64, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    rmsTyped%[tl.float64, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    rmsTyped%[tl.float64, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    rmsTyped%[tl.float64, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    rmsTyped%[tl.float64, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    rmsTyped%[tl.float64, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>rms_norm_kernel</code></summary>

```
/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(Y.dtype.element_ty) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}


/-! The complete floating storage family retains the cast before multiplication
by W. Runtime eps is fp32; the final cast models the implicit store conversion
only when multiplication promoted the value beyond Y's storage type. -/
open Lean
syntax "rmsTyped%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(rmsTyped%[$dtype:tritonDType, $weight:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += pid * $(y_stride_r))
      else `(tritonStmt| Y += $(y_stride_r) * pid)
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += pid * $(x_stride_r))
      else `(tritonStmt| X += $(x_stride_r) * pid)
    let narrow := match dtype, weight with
      | `(tritonDType| tl.float64), _ => Bool.false
      | `(tritonDType| tl.float32), `(tritonDType| tl.float64) => Bool.true
      | `(tritonDType| tl.float32), _ => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.float16) => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.bfloat16) => Bool.false
      | `(tritonDType| tl.bfloat16), `(tritonDType| tl.bfloat16) => Bool.false
      | _, _ => Bool.true
    let write : TSyntax `tritonStmt ← if narrow then
      `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y.to($dtype:tritonDType), mask=mask))
      else `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y, mask=mask))
    `(fun (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
        (eps : ℝ) (BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [X, W] [Y] (triton {
  tl.region Y = $dtype, X = $dtype, W = $weight
  pid = tl.program_id(0)
  $ybase
  $xbase
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to($dtype:tritonDType) * w
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
specification original_real_correct (storage weight : Storage) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat)
    (hN : 0 < N) (hNB : N ≤ B) (hyc : 0 < ysc) :
    Spec.Real (typedIO storage weight Bool.false Y X W ysr ysc xsr xsc N eps B ⊨
      fun _ _ xs ws i => rowFormula N B eps xs ws i)
```

**Assumptions / layout contracts:**
- `hN : 0 < N`
- `hNB : N ≤ B`
- `hyc : 0 < ysc`

**Closed-form spec defs (transitive):** `typedIO`, `rmsNormIO`, `typed`, `rms_norm_kernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (storage weight : Storage) (commuted : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) : Masked2DKernelIO₂ :=
  { rmsNormIO Y X W ysr ysc xsr xsc N eps B with
    kernel := (typed storage weight commuted Y X W ysr ysc xsr xsc N eps B).eraseDType
    projection := typed_supported storage weight commuted Y X W ysr ysc xsr xsc N eps B }
```
</details>

<details><summary><code>rmsNormIO</code></summary>

```
/-- `rms_norm_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring): the input
  matrix `X`, the per-column weights `W`, the output matrix `Y`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write` — **per-lane strided windows**: program `pid` reads its `X`
  row at `pid * x_stride_r + j * x_stride_c` and writes its `Y` row at
  `pid * y_stride_r + j * y_stride_c` (the host-side one-program-per-row
  launch convention with general row/column strides);
* `read2` — the weight window is **absolute and pid-independent**: every
  program reads `W[j]` (the host passes the same weight vector to all rows);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(N)`) carry no obligations on either side. The
  store mask equals the load mask, so `writeMask` keeps its default.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and mask are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and mask are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def rmsNormIO (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ where
  kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
    N eps BLOCK_SIZE
  in1 := X
  in2 := W
  out := Y
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * x_stride_r + j.val * x_stride_c
  read2 := fun _ _ j => j.val
  write := fun pid _ j => pid * y_stride_r + j.val * y_stride_c
  mask := fun _ _ j => j.val < N
```
</details>

<details><summary><code>typed</code></summary>

```lean
def typed (storage weight : Storage) (commuted : Bool)
    (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, weight, commuted with
  | .fp32, .fp32, Bool.false =>
    rmsTyped%[tl.float32, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    rmsTyped%[tl.float32, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    rmsTyped%[tl.float32, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    rmsTyped%[tl.float32, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    rmsTyped%[tl.float32, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    rmsTyped%[tl.float32, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    rmsTyped%[tl.float32, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    rmsTyped%[tl.float32, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    rmsTyped%[tl.float16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    rmsTyped%[tl.float16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    rmsTyped%[tl.float16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    rmsTyped%[tl.float16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    rmsTyped%[tl.float16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    rmsTyped%[tl.float16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    rmsTyped%[tl.float16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    rmsTyped%[tl.float16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    rmsTyped%[tl.float64, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    rmsTyped%[tl.float64, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    rmsTyped%[tl.float64, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    rmsTyped%[tl.float64, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    rmsTyped%[tl.float64, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    rmsTyped%[tl.float64, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    rmsTyped%[tl.float64, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    rmsTyped%[tl.float64, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>rms_norm_kernel</code></summary>

```
/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(Y.dtype.element_ty) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}


/-! The complete floating storage family retains the cast before multiplication
by W. Runtime eps is fp32; the final cast models the implicit store conversion
only when multiplication promoted the value beyond Y's storage type. -/
open Lean
syntax "rmsTyped%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(rmsTyped%[$dtype:tritonDType, $weight:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += pid * $(y_stride_r))
      else `(tritonStmt| Y += $(y_stride_r) * pid)
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += pid * $(x_stride_r))
      else `(tritonStmt| X += $(x_stride_r) * pid)
    let narrow := match dtype, weight with
      | `(tritonDType| tl.float64), _ => Bool.false
      | `(tritonDType| tl.float32), `(tritonDType| tl.float64) => Bool.true
      | `(tritonDType| tl.float32), _ => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.float16) => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.bfloat16) => Bool.false
      | `(tritonDType| tl.bfloat16), `(tritonDType| tl.bfloat16) => Bool.false
      | _, _ => Bool.true
    let write : TSyntax `tritonStmt ← if narrow then
      `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y.to($dtype:tritonDType), mask=mask))
      else `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y, mask=mask))
    `(fun (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
        (eps : ℝ) (BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [X, W] [Y] (triton {
  tl.region Y = $dtype, X = $dtype, W = $weight
  pid = tl.program_id(0)
  $ybase
  $xbase
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to($dtype:tritonDType) * w
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
specification candidate_real_correct (storage weight : Storage) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat)
    (hN : 0 < N) (hNB : N ≤ B) (hyc : 0 < ysc) :
    Spec.Real (typedIO storage weight Bool.true Y X W ysr ysc xsr xsc N eps B ⊨
      fun _ _ xs ws i => rowFormula N B eps xs ws i)
```

**Assumptions / layout contracts:**
- `hN : 0 < N`
- `hNB : N ≤ B`
- `hyc : 0 < ysc`

**Closed-form spec defs (transitive):** `typedIO`, `rmsNormIO`, `typed`, `rms_norm_kernel`

<details><summary><code>typedIO</code></summary>

```lean
def typedIO (storage weight : Storage) (commuted : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) : Masked2DKernelIO₂ :=
  { rmsNormIO Y X W ysr ysc xsr xsc N eps B with
    kernel := (typed storage weight commuted Y X W ysr ysc xsr xsc N eps B).eraseDType
    projection := typed_supported storage weight commuted Y X W ysr ysc xsr xsc N eps B }
```
</details>

<details><summary><code>rmsNormIO</code></summary>

```
/-- `rms_norm_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring): the input
  matrix `X`, the per-column weights `W`, the output matrix `Y`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write` — **per-lane strided windows**: program `pid` reads its `X`
  row at `pid * x_stride_r + j * x_stride_c` and writes its `Y` row at
  `pid * y_stride_r + j * y_stride_c` (the host-side one-program-per-row
  launch convention with general row/column strides);
* `read2` — the weight window is **absolute and pid-independent**: every
  program reads `W[j]` (the host passes the same weight vector to all rows);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(N)`) carry no obligations on either side. The
  store mask equals the load mask, so `writeMask` keeps its default.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and mask are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and mask are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
```
```lean
def rmsNormIO (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ where
  kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
    N eps BLOCK_SIZE
  in1 := X
  in2 := W
  out := Y
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * x_stride_r + j.val * x_stride_c
  read2 := fun _ _ j => j.val
  write := fun pid _ j => pid * y_stride_r + j.val * y_stride_c
  mask := fun _ _ j => j.val < N
```
</details>

<details><summary><code>typed</code></summary>

```lean
def typed (storage weight : Storage) (commuted : Bool)
    (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, weight, commuted with
  | .fp32, .fp32, Bool.false =>
    rmsTyped%[tl.float32, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    rmsTyped%[tl.float32, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    rmsTyped%[tl.float32, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    rmsTyped%[tl.float32, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    rmsTyped%[tl.float32, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    rmsTyped%[tl.float32, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    rmsTyped%[tl.float32, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    rmsTyped%[tl.float32, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    rmsTyped%[tl.float16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    rmsTyped%[tl.float16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    rmsTyped%[tl.float16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    rmsTyped%[tl.float16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    rmsTyped%[tl.float16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    rmsTyped%[tl.float16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    rmsTyped%[tl.float16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    rmsTyped%[tl.float16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    rmsTyped%[tl.bfloat16, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    rmsTyped%[tl.bfloat16, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    rmsTyped%[tl.float64, tl.float32, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    rmsTyped%[tl.float64, tl.float32, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    rmsTyped%[tl.float64, tl.float16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    rmsTyped%[tl.float64, tl.float16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    rmsTyped%[tl.float64, tl.bfloat16, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    rmsTyped%[tl.float64, tl.bfloat16, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    rmsTyped%[tl.float64, tl.float64, original] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    rmsTyped%[tl.float64, tl.float64, commuted] Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
```
</details>

<details><summary><code>rms_norm_kernel</code></summary>

```
/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(Y.dtype.element_ty) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}


/-! The complete floating storage family retains the cast before multiplication
by W. Runtime eps is fp32; the final cast models the implicit store conversion
only when multiplication promoted the value beyond Y's storage type. -/
open Lean
syntax "rmsTyped%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(rmsTyped%[$dtype:tritonDType, $weight:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += pid * $(y_stride_r))
      else `(tritonStmt| Y += $(y_stride_r) * pid)
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += pid * $(x_stride_r))
      else `(tritonStmt| X += $(x_stride_r) * pid)
    let narrow := match dtype, weight with
      | `(tritonDType| tl.float64), _ => Bool.false
      | `(tritonDType| tl.float32), `(tritonDType| tl.float64) => Bool.true
      | `(tritonDType| tl.float32), _ => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.float16) => Bool.false
      | `(tritonDType| tl.float16), `(tritonDType| tl.bfloat16) => Bool.false
      | `(tritonDType| tl.bfloat16), `(tritonDType| tl.bfloat16) => Bool.false
      | _, _ => Bool.true
    let write : TSyntax `tritonStmt ← if narrow then
      `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y.to($dtype:tritonDType), mask=mask))
      else `(tritonStmt| tl.store(Y + cols * $(y_stride_c), y, mask=mask))
    `(fun (Y X W : RegionName) (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
        (eps : ℝ) (BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [X, W] [Y] (triton {
  tl.region Y = $dtype, X = $dtype, W = $weight
  pid = tl.program_id(0)
  $ybase
  $xbase
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to($dtype:tritonDType) * w
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr
```
</details>
