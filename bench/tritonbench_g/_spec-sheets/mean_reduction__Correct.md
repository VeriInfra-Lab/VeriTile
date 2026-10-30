# Spec sheet — `bench/tritonbench_g/mean_reduction/Correct.lean`

**Python source:** `bench/tritonbench_g/mean_reduction/mean_reduction.py`

## Public theorem: `mean_dim_kernel_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre; single-stream
skin).** For every rounding model `R`, the faithful `mean_dim_kernel`
surface implements, on its `StreamMasked2DKernelIO₁` signature, the **ideal
ℝ row-mean fold** over the streamed masked tiles: output lane `i` holds
`(∑ t, ∑ j, in-window X-tile[t](i, j)) / N` — the spec `f` is exact real
arithmetic; the kernel is entirely cast-free and its store is `.real`-typed,
so the skin's readback contract at the default `outDType := .real` grid is
exact for every `R` (`R.round .real = id`): this is the exact streaming
genre carried on the single rounding surface.

Layer map: the pre-loop, the whole `for off` loop and the store tail are
cast-free, so under `execR R` they collapse verbatim onto the exact stepper
and the proven `meanPreLoop` / `meanLoopContextInvariant` / `forRange_inv` /
`meanPostLoop` stack above is reused unchanged; the `⊨[R]` face adds only
the `TraceSafeR` walk, the memory frame, and the stream-lane spec bridge.

The single hypothesis `hStepNe : BLOCK_N ≠ 0` is truth-forced: it is the
exact stack's own side condition (`mean_dim_kernel_output_summary`) — with
`BLOCK_N = 0` and `N > 0` the `for off in range(0, N, 0)` loop does not
terminate, `execR` returns `none`, and the statement is false. The output
window `pid·BLOCK_M + i` on `Fin BLOCK_M` is injective outright
(`meanOutOffset_injective_col1`), so no width hypothesis is needed. -/
```
</details>

**Statement:**
```lean
specification mean_dim_kernel_io_correctness (R : RoundingModel)
    (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat)
    (hStepNe : BLOCK_N ≠ 0) :
    meanKernelIO X Mean M N BLOCK_M BLOCK_N ⊨[R] fun _ _ xs i =>
      meanStreamSpec N BLOCK_M BLOCK_N xs i
```

**Assumptions / layout contracts:**
- `hStepNe : BLOCK_N ≠ 0`

**Closed-form spec defs (transitive):** `meanKernelIO`, `meanStreamSpec`, `mean_dim_kernel`, `meanNumSteps`, `meanReadAddr`, `meanWriteAddr`, `meanReadActive`, `meanWriteActive`, `meanStreamTerm`, `meanXLane`

<details><summary><code>meanKernelIO</code></summary>

```
/-- **Streaming IO signature** of `mean_dim_kernel` on the single-stream fold
skin (S1: fold + terminal store). Step `t` of the `for off` loop (at
`off = t·BLOCK_N`) reads the masked `[BLOCK_M, BLOCK_N]` `X`-tile; after the
loop one `BLOCK_M`-lane row-mean tile is stored to `Mean` at the `.real`
grid (the kernel's store is untyped/`.real`, so the default
`outDType := .real` applies and the readback is exact). The windows
transcribe the kernel's pointer arithmetic and masks exactly (pid axis 0;
lanes row-major over `[BLOCK_M, BLOCK_N]` via `Lane2D`). -/
```
```lean
def meanKernelIO (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) :
    StreamMasked2DKernelIO₁ where
  kernel := mean_dim_kernel X Mean M N BLOCK_M BLOCK_N
  inp1 := X
  out := Mean
  T := meanNumSteps N BLOCK_N
  B1 := BLOCK_M * BLOCK_N
  C := BLOCK_M
  read1 := fun p₀ _ t l => meanReadAddr N BLOCK_M BLOCK_N p₀ t.val l
  write := fun p₀ _ i => meanWriteAddr BLOCK_M p₀ i
  mask1 := fun p₀ _ t l => meanReadActive M N BLOCK_M BLOCK_N p₀ t.val l
  writeMask := fun p₀ _ i => meanWriteActive M BLOCK_M p₀ i
```
</details>

<details><summary><code>meanStreamSpec</code></summary>

```
/-- The stream-level row-mean spec: output lane `i` holds the double fold
`(∑ t, ∑ j, in-window xs) / N` over the whole curried stream. -/
```
```lean
noncomputable def meanStreamSpec (N BLOCK_M BLOCK_N : Nat)
    (xs : Fin (meanNumSteps N BLOCK_N) → Fin (BLOCK_M * BLOCK_N) → ℝ)
    (i : Fin BLOCK_M) : ℝ :=
  (∑ t : Fin (meanNumSteps N BLOCK_N), ∑ j : Fin BLOCK_N,
    meanStreamTerm N BLOCK_M BLOCK_N xs i t j) / (N : ℝ)
```
</details>

<details><summary><code>mean_dim_kernel</code></summary>

```
/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters.

The proof below connects the full-row spec to a loop invariant for the
`for off in range(...)` accumulation.
-/
```
```lean
def mean_dim_kernel
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None]
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean, mean, row_mask)
}

/-! Typed source family. The candidate commutes only the natural-number
program-base product. Loaded values still explicitly convert to fp32 before
the complete accumulation loop; the output conversion makes the store dtype
explicit after pointer rebinding. -/
syntax "meanSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(meanSource%[$input:tritonDType, $output:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let pid : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None])
      else
        `(tritonStmt| pid = $(BLOCK_M) * tl.program_id(0) + tl.arange(0, $(BLOCK_M))[:, None])
    let write : TSyntax `tritonStmt ←
      match output with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Mean, mean, row_mask))
      | _ => `(tritonStmt| tl.store(Mean, mean.to($output), row_mask))
    `(fun (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [X] [Mean] (triton {
  tl.region X = $input, Mean = $output
  $pid
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq
```
</details>

<details><summary><code>meanNumSteps</code></summary>

```
/-- Trip count of the `for off in range(0, N, BLOCK_N)` stream:
`⌈N / BLOCK_N⌉`. -/
```
```lean
def meanNumSteps (N BLOCK_N : Nat) : Nat := (N + BLOCK_N - 1) / BLOCK_N
```
</details>

<details><summary><code>meanReadAddr</code></summary>

```
/-- Step `t`, lane `l = (i, j)`'s `X` read address: row `pid·BLOCK_M + i`
(row stride `N`), global column `t·BLOCK_N + j`. -/
```
```lean
def meanReadAddr (N BLOCK_M BLOCK_N p₀ t : Nat)
    (l : Fin (BLOCK_M * BLOCK_N)) : Nat :=
  (p₀ * BLOCK_M + l.val / BLOCK_N) * N + (t * BLOCK_N + l.val % BLOCK_N)
```
</details>

<details><summary><code>meanWriteAddr</code></summary>

```
/-- Output lane `i`'s terminal write address: `Mean + pid·BLOCK_M + i`. -/
```
```lean
def meanWriteAddr (BLOCK_M p₀ : Nat) (i : Fin BLOCK_M) : Nat :=
  p₀ * BLOCK_M + i.val
```
</details>

<details><summary><code>meanReadActive</code></summary>

```
/-- Step `t`, lane `l = (i, j)`'s read-active window: the kernel's
`row_mask and col_mask` — row in `[0, M)`, global column in `[0, N)`. -/
```
```lean
def meanReadActive (M N BLOCK_M BLOCK_N p₀ t : Nat)
    (l : Fin (BLOCK_M * BLOCK_N)) : Prop :=
  p₀ * BLOCK_M + l.val / BLOCK_N < M ∧ t * BLOCK_N + l.val % BLOCK_N < N
```
</details>

<details><summary><code>meanWriteActive</code></summary>

```
/-- The terminal store's write-active window: the kernel's `row_mask`. -/
```
```lean
def meanWriteActive (M BLOCK_M p₀ : Nat) (i : Fin BLOCK_M) : Prop :=
  p₀ * BLOCK_M + i.val < M
```
</details>

<details><summary><code>meanStreamTerm</code></summary>

```
/-- The guarded per-step summand of the stream spec: lane `(i, j)` of step
`t` contributes its streamed value inside the column window
`t·BLOCK_N + j < N`, and `0` outside (the kernel's `other=0.0`). -/
```
```lean
noncomputable def meanStreamTerm (N BLOCK_M BLOCK_N : Nat)
    (xs : Fin (meanNumSteps N BLOCK_N) → Fin (BLOCK_M * BLOCK_N) → ℝ)
    (i : Fin BLOCK_M) (t : Fin (meanNumSteps N BLOCK_N)) (j : Fin BLOCK_N) : ℝ :=
  if t.val * BLOCK_N + j.val < N then xs t (meanXLane BLOCK_M BLOCK_N i j) else 0
```
</details>

<details><summary><code>meanXLane</code></summary>

```
/-- The `X`-stream lane holding row `i`, in-block column `j` of the
`[BLOCK_M, BLOCK_N]` per-step tile (row-major, via the shared `Lane2D`
bridge). -/
```
```lean
def meanXLane (BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) (j : Fin BLOCK_N) :
    Fin (BLOCK_M * BLOCK_N) :=
  Lane2D.encode (i, j, PUnit.unit)
```
</details>

## Public theorem: `mean_typed_real_correct`

<details><summary>docstring</summary>

```
/-- Both complete typed programs implement the independent streamed row mean
under real semantics. Success, active bounds and the complete flat-memory frame
are retained. Positive dimensions describe legal nonempty reduction launches. -/
```
</details>

**Statement:**
```lean
specification mean_typed_real_correct (input output : Storage) (commuted : Bool)
    (X Mean : RegionName) (M N BM BN : Nat) (_hN : 0 < N) (_hM : 0 < BM)
    (hB : 0 < BN) :
    VeriTile.Spec.Real (typedIO input output commuted X Mean M N BM BN
      ⊨[RoundingModel.triv] fun _ _ xs i => meanStreamSpec N BM BN xs i)
```

**Assumptions / layout contracts:**
- `_hN : 0 < N`
- `_hM : 0 < BM`
- `hB : 0 < BN`

**Closed-form spec defs (transitive):** `typedIO`, `meanStreamSpec`, `meanKernelIO`, `typedKernel`, `meanNumSteps`, `meanStreamTerm`, `mean_dim_kernel`, `meanReadAddr`, `meanWriteAddr`, `meanReadActive`, `meanWriteActive`, `meanXLane`

<details><summary><code>typedIO</code></summary>

```
/-- Checked real projection of the complete typed stream, with the original
input windows and all active row stores. -/
```
```lean
def typedIO (input output : Storage) (commuted : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) : StreamMasked2DKernelIO₁ :=
  { meanKernelIO X Mean M N BM BN with
    kernel := (typedKernel input output commuted X Mean M N BM BN).eraseDType
    projection := typed_supported input output commuted X Mean M N BM BN }
```
</details>

<details><summary><code>meanStreamSpec</code></summary>

```
/-- The stream-level row-mean spec: output lane `i` holds the double fold
`(∑ t, ∑ j, in-window xs) / N` over the whole curried stream. -/
```
```lean
noncomputable def meanStreamSpec (N BLOCK_M BLOCK_N : Nat)
    (xs : Fin (meanNumSteps N BLOCK_N) → Fin (BLOCK_M * BLOCK_N) → ℝ)
    (i : Fin BLOCK_M) : ℝ :=
  (∑ t : Fin (meanNumSteps N BLOCK_N), ∑ j : Fin BLOCK_N,
    meanStreamTerm N BLOCK_M BLOCK_N xs i t j) / (N : ℝ)
```
</details>

<details><summary><code>meanKernelIO</code></summary>

```
/-- **Streaming IO signature** of `mean_dim_kernel` on the single-stream fold
skin (S1: fold + terminal store). Step `t` of the `for off` loop (at
`off = t·BLOCK_N`) reads the masked `[BLOCK_M, BLOCK_N]` `X`-tile; after the
loop one `BLOCK_M`-lane row-mean tile is stored to `Mean` at the `.real`
grid (the kernel's store is untyped/`.real`, so the default
`outDType := .real` applies and the readback is exact). The windows
transcribe the kernel's pointer arithmetic and masks exactly (pid axis 0;
lanes row-major over `[BLOCK_M, BLOCK_N]` via `Lane2D`). -/
```
```lean
def meanKernelIO (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) :
    StreamMasked2DKernelIO₁ where
  kernel := mean_dim_kernel X Mean M N BLOCK_M BLOCK_N
  inp1 := X
  out := Mean
  T := meanNumSteps N BLOCK_N
  B1 := BLOCK_M * BLOCK_N
  C := BLOCK_M
  read1 := fun p₀ _ t l => meanReadAddr N BLOCK_M BLOCK_N p₀ t.val l
  write := fun p₀ _ i => meanWriteAddr BLOCK_M p₀ i
  mask1 := fun p₀ _ t l => meanReadActive M N BLOCK_M BLOCK_N p₀ t.val l
  writeMask := fun p₀ _ i => meanWriteActive M BLOCK_M p₀ i
```
</details>

<details><summary><code>typedKernel</code></summary>

```lean
def typedKernel (input output : Storage) (commuted : Bool) : Source :=
  match input, commuted with
  | .fp32, Bool.false => meanOutputs%[tl.float32, original] output
  | .fp32, Bool.true => meanOutputs%[tl.float32, indexCommuted] output
  | .fp16, Bool.false => meanOutputs%[tl.float16, original] output
  | .fp16, Bool.true => meanOutputs%[tl.float16, indexCommuted] output
  | .bf16, Bool.false => meanOutputs%[tl.bfloat16, original] output
  | .bf16, Bool.true => meanOutputs%[tl.bfloat16, indexCommuted] output
  | .fp64, Bool.false => meanOutputs%[tl.float64, original] output
  | .fp64, Bool.true => meanOutputs%[tl.float64, indexCommuted] output
```
</details>

<details><summary><code>meanNumSteps</code></summary>

```
/-- Trip count of the `for off in range(0, N, BLOCK_N)` stream:
`⌈N / BLOCK_N⌉`. -/
```
```lean
def meanNumSteps (N BLOCK_N : Nat) : Nat := (N + BLOCK_N - 1) / BLOCK_N
```
</details>

<details><summary><code>meanStreamTerm</code></summary>

```
/-- The guarded per-step summand of the stream spec: lane `(i, j)` of step
`t` contributes its streamed value inside the column window
`t·BLOCK_N + j < N`, and `0` outside (the kernel's `other=0.0`). -/
```
```lean
noncomputable def meanStreamTerm (N BLOCK_M BLOCK_N : Nat)
    (xs : Fin (meanNumSteps N BLOCK_N) → Fin (BLOCK_M * BLOCK_N) → ℝ)
    (i : Fin BLOCK_M) (t : Fin (meanNumSteps N BLOCK_N)) (j : Fin BLOCK_N) : ℝ :=
  if t.val * BLOCK_N + j.val < N then xs t (meanXLane BLOCK_M BLOCK_N i j) else 0
```
</details>

<details><summary><code>mean_dim_kernel</code></summary>

```
/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters.

The proof below connects the full-row spec to a loop invariant for the
`for off in range(...)` accumulation.
-/
```
```lean
def mean_dim_kernel
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None]
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean, mean, row_mask)
}

/-! Typed source family. The candidate commutes only the natural-number
program-base product. Loaded values still explicitly convert to fp32 before
the complete accumulation loop; the output conversion makes the store dtype
explicit after pointer rebinding. -/
syntax "meanSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(meanSource%[$input:tritonDType, $output:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let pid : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None])
      else
        `(tritonStmt| pid = $(BLOCK_M) * tl.program_id(0) + tl.arange(0, $(BLOCK_M))[:, None])
    let write : TSyntax `tritonStmt ←
      match output with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Mean, mean, row_mask))
      | _ => `(tritonStmt| tl.store(Mean, mean.to($output), row_mask))
    `(fun (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [X] [Mean] (triton {
  tl.region X = $input, Mean = $output
  $pid
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  $write
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq
```
</details>

<details><summary><code>meanReadAddr</code></summary>

```
/-- Step `t`, lane `l = (i, j)`'s `X` read address: row `pid·BLOCK_M + i`
(row stride `N`), global column `t·BLOCK_N + j`. -/
```
```lean
def meanReadAddr (N BLOCK_M BLOCK_N p₀ t : Nat)
    (l : Fin (BLOCK_M * BLOCK_N)) : Nat :=
  (p₀ * BLOCK_M + l.val / BLOCK_N) * N + (t * BLOCK_N + l.val % BLOCK_N)
```
</details>

<details><summary><code>meanWriteAddr</code></summary>

```
/-- Output lane `i`'s terminal write address: `Mean + pid·BLOCK_M + i`. -/
```
```lean
def meanWriteAddr (BLOCK_M p₀ : Nat) (i : Fin BLOCK_M) : Nat :=
  p₀ * BLOCK_M + i.val
```
</details>

<details><summary><code>meanReadActive</code></summary>

```
/-- Step `t`, lane `l = (i, j)`'s read-active window: the kernel's
`row_mask and col_mask` — row in `[0, M)`, global column in `[0, N)`. -/
```
```lean
def meanReadActive (M N BLOCK_M BLOCK_N p₀ t : Nat)
    (l : Fin (BLOCK_M * BLOCK_N)) : Prop :=
  p₀ * BLOCK_M + l.val / BLOCK_N < M ∧ t * BLOCK_N + l.val % BLOCK_N < N
```
</details>

<details><summary><code>meanWriteActive</code></summary>

```
/-- The terminal store's write-active window: the kernel's `row_mask`. -/
```
```lean
def meanWriteActive (M BLOCK_M p₀ : Nat) (i : Fin BLOCK_M) : Prop :=
  p₀ * BLOCK_M + i.val < M
```
</details>

<details><summary><code>meanXLane</code></summary>

```
/-- The `X`-stream lane holding row `i`, in-block column `j` of the
`[BLOCK_M, BLOCK_N]` per-step tile (row-major, via the shared `Lane2D`
bridge). -/
```
```lean
def meanXLane (BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) (j : Fin BLOCK_N) :
    Fin (BLOCK_M * BLOCK_N) :=
  Lane2D.encode (i, j, PUnit.unit)
```
</details>

## Also present (pinned special-case summaries)
- `mean_dim_kernel_compute_correct`
