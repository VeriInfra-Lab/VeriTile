# Spec sheet — `bench/tritonbench_g/kldiv_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/kldiv_triton/kldiv_triton.py`

## Public theorem: `kldiv_backward_default_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the default backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_default_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_default input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => defaultSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)`

**Closed-form spec defs (transitive):** `kldiv_backward_default`

<details><summary><code>kldiv_backward_default</code></summary>

```
/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_backward`
for the `log_target = False` constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop. Proofs
target the Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`.

Allowed mechanical Lean-syntax-only changes:
- Python `log_target: tl.constexpr` → separate kernel defs per branch. -/
```
```lean
def kldiv_backward_default
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    res = target * -1
    tl.store(input_ptr + offsets, res, mask=mask)
  }
}
```
</details>

## Public theorem: `kldiv_backward_log_target_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the log-target backward kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_backward_log_target_compute_correct
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_backward_log_target input_ptr target_ptr
        input_stride target_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (input_ptr, linearOffset s input_stride i)))
      (expected := fun i => logTargetSpec s target_ptr target_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s input_stride i)`

**Closed-form spec defs (transitive):** `kldiv_backward_log_target`

<details><summary><code>kldiv_backward_log_target</code></summary>

```
/-- Faithful transcription of `_kldiv_kernel_backward` for the
`log_target = True` constexpr branch. Includes the Python
`for i in range(0, n_cols, BLOCK_SIZE)` loop; proofs target the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
```lean
def kldiv_backward_log_target
    (input_ptr target_ptr : RegionName)
    (input_stride target_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  input_ptr += pid * $(input_stride)
  target_ptr += pid * $(target_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    res = -tl.exp(target)
    tl.store(input_ptr + offsets, res, mask=mask)
  }
}



namespace TypedSource
open Lean

/-- Floating memory carriers. This is a frontend model, not a native-device support claim. -/
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Direct `tl.log` and `tl.exp` accept fp32/fp64 in pinned Triton 3.5.1. -/
inductive MathStorage where
  | fp32 | fp64
  deriving DecidableEq, Repr

syntax "klForward%" "[" tritonDType "," tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(klForward%[$pred:tritonDType, $target:tritonDType, $output:tritonDType]) => do
    let wide := match pred, target with
      | `(tritonDType| tl.float64), _ | _, `(tritonDType| tl.float64) => Bool.true
      | _, _ => Bool.false
    let same := match output with
      | `(tritonDType| tl.float64) => wide
      | `(tritonDType| tl.float32) => !wide
      | _ => Bool.false
    let elementStore : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O + offsets, loss, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, loss.to($output), mask=mask))
    let chunkStore : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O, loss))
      else `(tritonStmt| tl.store(O, loss.to($output)))
    `(fun (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) =>
      ComputeKernel.mk [Y, T] [O] (triton {
        tl.region Y = $pred, T = $target, O = $output
        pid = tl.program_id(0).to(tl.int64)
        Y += pid * $(sy)
        T += pid * $(st)
        O += pid * $(so)
        base_offsets = tl.arange(0, $(B))
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + base_offsets
          mask = offsets < $((N : Int))
          y = tl.load(Y + offsets, mask=mask, other=0.0)
          y_true = tl.load(T + offsets, mask=mask, other=0.0)
          if not logTarget {
            loss = y_true * (tl.log(y_true) - y)
          } else {
            loss = tl.exp(y_true) * (y_true - y)
          }
          if $(reduction) == $(0) {
            $elementStore
          } else {
            loss = tl.sum(loss, axis=0)
            $chunkStore
            O += $(1)
          }
        }
      }).surfaceBody)

syntax "klBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(klBackward%[$target:tritonDType, $output:tritonDType, $branch:ident]) => do
    let logarithmic := branch.getId == `logarithmic
    unless logarithmic || branch.getId == `defaultTarget do
      Macro.throwError "expected logarithmic or defaultTarget"
    let compute : TSyntax `tritonStmt ← if logarithmic then
      `(tritonStmt| res = -tl.exp(target))
      else `(tritonStmt| res = target * -1)
    let same := match target, output with
      | `(tritonDType| tl.float32), `(tritonDType| tl.float32)
      | `(tritonDType| tl.float64), `(tritonDType| tl.float64)
      | `(tritonDType| tl.float16), `(tritonDType| tl.float16)
      | `(tritonDType| tl.bfloat16), `(tritonDType| tl.bfloat16)
      | `(tritonDType| tl.float8e4nv), `(tritonDType| tl.float8e4nv)
      | `(tritonDType| tl.float8e5), `(tritonDType| tl.float8e5) => Bool.true
      | _, _ => Bool.false
    let store : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O + offsets, res, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, res.to($output), mask=mask))
    `(fun (T O : RegionName) (st so N B : Nat) =>
      ComputeKernel.mk [T] [O] (triton {
        tl.region T = $target, O = $output
        pid = tl.program_id(0).to(tl.int64)
        O += pid * $(so)
        T += pid * $(st)
        offsets = tl.arange(0, $(B))
        mask = offsets < $((N : Int))
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + tl.arange(0, $(B))
          mask = offsets < $((N : Int))
          target = tl.load(T + offsets, mask=mask, other=0.0)
          $compute
          $store
        }
      }).surfaceBody)
```
</details>

## Public theorem: `kldiv_forward_default_none_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the forward `log_target=False`,
`reduction=0` kernel under the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_default_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_default_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardDefaultSpec s y_ptr gt_ptr y_stride gt_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`

**Closed-form spec defs (transitive):** `kldiv_forward_default_none`

<details><summary><code>kldiv_forward_default_none</code></summary>

```
/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`
for the `log_target = False`, `reduction = 0` (None) constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop; proofs
target the Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`.
Mirrors the Python `loss = y_true * (tl.log(y_true) - y)` body; no `eps`
clamp (unlike the kldiv_ops port). -/
```
```lean
def kldiv_forward_default_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    loss = y_true * (tl.log(y_true) - y)
    tl.store(loss_ptr + offsets, loss, mask=mask)
  }
}
```
</details>

## Public theorem: `kldiv_forward_log_target_none_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the forward log-target kernel under the
Python-tested single-chunk regime `0 < n_cols ≤ BLOCK_SIZE`. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_log_target_none_compute_correct
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen : n_cols ≤ BLOCK_SIZE)
    (hLenPos : 0 < n_cols)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := kldiv_forward_log_target_none y_ptr gt_ptr loss_ptr
        y_stride gt_stride loss_stride n_cols BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < n_cols)
        (fun i => (loss_ptr, linearOffset s loss_stride i)))
      (expected := fun i =>
        forwardLogTargetSpec s y_ptr gt_ptr y_stride gt_stride i)
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : n_cols ≤ BLOCK_SIZE`
- `hLenPos : 0 < n_cols`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => linearOffset s loss_stride i)`

**Closed-form spec defs (transitive):** `kldiv_forward_log_target_none`

<details><summary><code>kldiv_forward_log_target_none</code></summary>

```
/-- Faithful transcription of `kldiv_triton.py`'s `_kldiv_kernel_forward`
for the `log_target = True`, `reduction = 0` (None) constexpr branch.

Includes the Python `for i in range(0, n_cols, BLOCK_SIZE)` loop with the
elementwise-store path; proofs target the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`. -/
```
```lean
def kldiv_forward_log_target_none
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0).to(tl.int64)
  y_ptr += pid * $(y_stride)
  gt_ptr += pid * $(gt_stride)
  loss_ptr += pid * $(loss_stride)
  base_offsets = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), $(n_cols), $(BLOCK_SIZE)) {
    offsets = i + base_offsets
    mask = offsets < $(n_cols)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
    loss = tl.exp(y_true) * (y_true - y)
    tl.store(loss_ptr + offsets, loss, mask=mask)
  }
}
```
</details>

## Public theorem: `kldiv_forward_default_none_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `kldiv_forward_default_none` surface
(the `log_target = False`, `reduction = 0` (None) forward branch) implements,
on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ elementwise
forward KL divergence** over the streamed tiles: emitted window `(t, j)` holds
`y_true[t,j] · (log y_true[t,j] − y[t,j])`, with `y` the log-space prediction
stream and `y_true` the ground-truth stream. The spec `f` is exact real
arithmetic — this port has **no `eps` clamp** (unlike the `kldiv_ops` port).

Rounding story: the kernel has **zero rounding events**. Both `tl.load`s and
the in-loop `tl.store` are at `.real` (the lowered store is
`Stmt.store TileDType.real … (MaskOpt.mask …)` — no cast on the stored value),
and the kernel contains no `castFloat` at all; the only casts are the
`int`-flavoured `pid = tl.program_id(0).to(tl.int64)` pointer arithmetic, which
is exact. So the skin's boundary quantization degenerates: at the declared grid
`outDType := .real` the readback contract's `R.round .real` is the identity by
the model's defining `round_real`, and the `.real` in-loop stores are exact
under `execR R`. The ∀-`R` face therefore holds via the `RoundingModel` `.real`
identity fields, **not** as a `.triv` special case.

Layer map: the prologue and the loop body are cast-free
(`klPrefix_castFree` / `klLossBody_castFree`), so under `execR R` they collapse
verbatim onto the exact stepper; the loop is driven by the `forRange`
invariant `klInv` (`kl_preLoop` + `klInv_step` + `kl_runR`), the safety walk by
`Stmt.forRangeTraceSafeR_inv` over the same invariant (`kl_lossBodySafeR` +
`kldiv_traceSafeR`), and the stream indices are matched to the memory-level
closed form by `klStreamSpec_eq_klCell`.

**Scope: this face is dimension-general — the full multi-chunk stream.** `T`
is `⌈n_cols / BLOCK_SIZE⌉`, so the loop is proved for an arbitrary number of
iterations; there is **no** `n_cols ≤ BLOCK_SIZE` regime hypothesis and no
`0 < n_cols` hypothesis. (The retained exact surfaces below/above are the
narrower single-chunk statements; see "Relation to the exact surfaces".)

Every hypothesis is truth-forced:

* `hBS : 0 < BLOCK_SIZE` — the loop steps by `BLOCK_SIZE`
  (`range(0, n_cols, BLOCK_SIZE)`); at `BLOCK_SIZE = 0` with `n_cols > 0` the
  `forRange` fold does not advance, `execR` cannot terminate the way the
  invariant requires, and the step index `i / BLOCK_SIZE` is meaningless. It
  holds for every real launch (the host picks
  `BLOCK_SIZE = min(16384, next_power_of_2(S))`).
* `hly : loss_ptr ≠ y_ptr`, `hlg : loss_ptr ≠ gt_ptr` — the store sits
  **inside** the loop, so iteration `t + 1` loads *after* iteration `t` has
  already stored. If the loss buffer aliased either input buffer, later chunks
  would re-read overwritten values and the closed form would be false. (In the
  single-chunk regime these are vacuous; they are exactly the price of the
  general multi-chunk face.) The host allocates `output_tensor` fresh, so both
  hold for every real launch.

Relation to the exact surfaces: the file's exact headline
`kldiv_forward_default_none_compute_correct`
(`Realizes_without_Rounding`, scoped to the Python-tested single-chunk regime
`0 < n_cols ≤ BLOCK_SIZE`, with the offset-injectivity side condition) is
retained unchanged; this `⊨[R]` face restates the same KL content on the
streaming emit skin, for every `R` at once and for an arbitrary number of
chunks (at the `.real` grid the two faces carry the same exact cell on the
single-chunk overlap). Both faces are kept per the rounding-as-default
doctrine. The `log_target = True` branch and the `reduction ≠ 0` reduction
modes keep their existing (exact, partial) treatment — the reduction-mode
surface `kldiv_forward_surface` is still only proved to lower. -/
```
</details>

**Statement:**
```lean
specification kldiv_forward_default_none_io_correctness (R : RoundingModel)
    (y_ptr gt_ptr loss_ptr : RegionName)
    (y_stride gt_stride loss_stride n_cols BLOCK_SIZE : Nat)
    (hBS : 0 < BLOCK_SIZE)
    (hly : loss_ptr ≠ y_ptr) (hlg : loss_ptr ≠ gt_ptr) :
    kldivForwardDefaultNoneIO y_ptr gt_ptr loss_ptr y_stride gt_stride loss_stride
        n_cols BLOCK_SIZE ⊨[R]
      fun _ _ xs ys t j => klStreamSpec n_cols BLOCK_SIZE xs ys t j
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hly : loss_ptr ≠ y_ptr`
- `hlg : loss_ptr ≠ gt_ptr`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `typed_forward_real_correct`

<details><summary>docstring</summary>

```
/-- Independent full forward loss formula, successful flat IO and exact raw-cell frame.
Every nonzero reduction mode writes one sum per executed chunk; N=0 writes nothing.
Both source variants are proved separately through their mathematical projection. -/
```
</details>

**Statement:**
```lean
specification typed_forward_real_correct
    (pred : TypedSource.Storage) (target : TypedSource.MathStorage)
    (output : TypedSource.Storage) (v : Bool) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O) :
    Contracts.ForwardIO
      (TypedSource.forward pred target output v Y T O sy st so N B flag reduction)
      Y T O sy st so N B flag reduction
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hYO : Y ≠ O`
- `hTO : T ≠ O`

**Closed-form spec defs (transitive):** `forward`, `forwardOriginal`, `commuteOutput`

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

## Public theorem: `typed_backward_default_real_correct`

<details><summary>docstring</summary>

```
/-- Full default-backward gradients and flat-memory frame for the source-supported
weak-literal target types. Output storage remains independently selectable. -/
```
</details>

**Statement:**
```lean
specification typed_backward_default_real_correct
    (target output : TypedSource.Storage) (v : Bool) (T O : RegionName)
    (st so N B : Nat) (hSource : TypedSource.BackwardSourceSupported target)
    (hB : 0 < B) (hTO : T ≠ O) :
    Contracts.BackwardIO (TypedSource.backwardDefault target output v T O st so N B)
      T O st so N Bool.false
```

**Assumptions / layout contracts:**
- `hSource : TypedSource.BackwardSourceSupported target`
- `hB : 0 < B`
- `hTO : T ≠ O`

**Closed-form spec defs (transitive):** `backwardDefault`, `backwardDefaultOriginal`, `commuteOutput`

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

## Public theorem: `typed_backward_log_real_correct`

<details><summary>docstring</summary>

```
/-- Full logarithmic-target backward gradients, successful flat execution and raw frame. -/
```
</details>

**Statement:**
```lean
specification typed_backward_log_real_correct
    (target : TypedSource.MathStorage) (output : TypedSource.Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) (hB : 0 < B) (hTO : T ≠ O) :
    Contracts.BackwardIO (TypedSource.backwardLog target output v T O st so N B)
      T O st so N Bool.true
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hTO : T ≠ O`

**Closed-form spec defs (transitive):** `backwardLog`, `backwardLogOriginal`, `commuteOutput`

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
