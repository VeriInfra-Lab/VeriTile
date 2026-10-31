import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-!
# `fused_recurrent_hgrn` — strict per-kernel correctness

`fused_recurrent_hgrn_fwd_kernel` is the HGRN forward recurrent state scan:
each program carries a hidden-state vector `b_h` across a `0..T` time loop,
updating `b_h = g_t * b_h + x_t` per step, optionally
seeded by an initial state and optionally storing the final state. The
companion backward kernel performs the reverse-time gradient scan.

## Scope

This file verifies **four hand-cut slices** of the two `@triton.jit` bodies: one
forward loop body, the backward loop body's `dx` and `dg` writebacks, and a
two-consecutive-iterations `dx` slice that executes the backward scan's carry
fold. It does **not** verify the launched kernels: the forward surface
`fused_recurrent_hgrn_fwd_surface` is only shown to lower to the algorithm
layer, and the backward surface `fused_recurrent_hgrn_bwd_surface` appears in no
correctness face at all. The host launch (grid shape, `@triton.autotune` config
selection over `BD`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*. Because the program ids are universally
quantified, each per-program statement covers every program of the grid.

## Proof architecture

```
fused_recurrent_hgrn_output_summary_general               ← TOP THEOREM
  ├─ fused_recurrent_hgrn_fwd_surface_toAlgorithm_supported       (lowering only)
  ├─ fused_recurrent_hgrn_forward_step_closed_form                (hgrnStateClosed forward)
  ├─ fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct
  ├─ fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct (bwdPrevOut branch)
  ├─ fused_recurrent_hgrn_bwd_dx_two_step_closed_form             (hgrnBwdDx scan)
  │     └─ bwdDxTwoStepValue_eq_hgrnBwdDx
  │           ├─ hgrnBwdDx_eq_carry_add_do   (b_dh = b_dh + b_do)
  │           └─ hgrnBwdCarry_pred           (b_dh = b_dh * b_g  ← the scan)
  └─ fused_recurrent_hgrn_bwd_dx_two_step_closed_form_init
        └─ hgrnBwdCarry_init                 (the `tl.zeros` seed)
```

## Modeling boundary — read before trusting anything below

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity post-erasure. `@triton.autotune` (the `BD` config sweep)
is not modeled — `BD` is a free parameter. The step slices reproduce Python's
`mask = o_d < D` / `other = 0` loads faithfully, so no full-tile side condition
is needed. What is **outside** every claim in this file:

* **The backward cross-step fold.** The *forward* `range(0, T)` fold is no
  longer in this list: `hgrnFwdOuterLoop_run` runs the kernel's **own**
  `Stmt.forRange` under `forRange_inv`, carrying `b_h` in a *register* across
  all `T` iterations — the way Python carries it — and lands
  `hgrnStateClosed T` with no per-step hypothesis — **and** it states the per-row
  `O` history: every time row `t < T` holds `hgrnStateClosed (t+1)` on every
  active lane, which is the kernel's actual output, not just its final carry.
  It needs only `x ≠ o` and `g ≠ o`. The headline's clause 2 is still the
  single-step face over the materialized fiction region `BHPrev`, so *that
  clause's* scope is unchanged; the fold stands beside it, not through it.

  The backward reverse-time fold threading `b_dh` is still unmodeled: its carry
  is presented to its slice as a materialized fiction region `DHPrev`
  constrained by an *assumed* hypothesis.
  On the backward side the carry *value* is pinned down: `hgrnBwdCarry` is the
  closed form of `b_dh`, `hgrnBwdCarry_pred` proves the `b_dh = b_dh * b_g` fold
  and `hgrnBwdCarry_init` the `tl.zeros` seed, and the two-step slice executes
  one fold. What is still assumed is only that the fiction region `DHPrev` holds
  that value (and at the loop's first fold even that degenerates to "the seed row
  reads `0`"). Chaining the fold across all `T` reverse steps is not modeled.
* **The `STORE_FINAL_STATE` writeback.** No correctness face. The former
  `final_state_store_slice` face was a masked memcpy (load address
  `BHFinal + i_bh·D + offs_d`, store address `Ht + i_bh·D + offs_d` — identical
  under the same mask) whose only content was the assumption
  `BHFinal = hgrnStateClosed(T)`; it has been deleted rather than presented as a
  result about `ht`.
* **The `dg` face's `O` row is the forward kernel's output only by the autograd
  contract.** The `dg` slice now reads Python's actual `b_o` operands — region
  `O` at the *previous* row `i_t − 1` and region `H0` at `i_bh·D + offs_d`, under
  Python's three-way branch (`bwdPrevOut`). That `O` in fact holds the forward
  pass's output rows is `ctx.save_for_backward(g, o, initial_state)`, i.e. the
  host-side autograd plumbing, which is outside this file; no theorem here
  connects the two kernels.
* **Region distinctness.** No `≠` hypotheses are stated, and none are needed:
  each slice performs all of its loads before its single store, and every
  `expected` is a function of the *initial* state, so aliasing cannot falsify
  any face below.
-/

namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileShapedKernelIO₂

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `fused_recurrent_hgrn_output_summary_general` — shape-general,
genuine closed form `hgrnStateClosed` over the input regions, but scoped to three
single-step **slices** (one forward body, the backward `dx`/`dg` writebacks), not
to the launched kernels. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Algorithm-layer form of the forward surface (a lowering identity)

`fused_recurrent_hgrn_fwd_body_split` below is **not a correctness result**. It
is the exact-lowering counterpart of `*_toAlgorithm_supported`: where that says
*some* algorithm exists, this says *which* one, by naming the erased statement
list and checking it with `rfl`.

Its purpose is to make the forward loop attackable as the kernel's **own**
`Stmt.forRange`, so that a cross-step fold can be run with `forRange_inv`
(`VeriTile.Triton.LoopInvariant`) with the carry `b_h` staying in a *register*
across iterations — the way the Python loop actually carries it, and the route
`chunk_delta_fwd` takes via `cdfOuterLoop_run`.  That fold is **not** in this
file yet; until it lands, the port's cross-step story is still the pinned
`hPrev` hypothesis of `fused_recurrent_hgrn_forward_step_closed_form`, and
nothing here changes that.

Two facts the split makes visible, both needed by the eventual invariant:

* the loop body advances `p_x`/`p_g`/`p_o` by `D` *each iteration*, so the
  invariant must pin all three pointer registers at `base + i·D`, not just the
  carry `b_h`;
* the loads carry `MaskOpt.maskOther … 0` and the store `MaskOpt.mask`, i.e. the
  `mask = o_d < D` lane predicate is applied on both sides. -/

/-- The `p_x`/`p_g`/`p_o` row-base pointer expression (`R + i_bh·T·D + o_d`). -/
def hgrnRowPtr (R : RegionName) (T D BD : Nat) : Op .ptr [BD] :=
  Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
    (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T))
        (Op.constNat D))
      (Op.ref .nat [BD] "o_d"))

/-- The `p_h0`/`p_ht` state pointer expression (`R + i_bh·D + o_d`). -/
def hgrnStatePtr (R : RegionName) (D BD : Nat) : Op .ptr [BD] :=
  Op.ptrAdd Broadcast.scalarL (Op.ptrBase R)
    (Op.add .nat Broadcast.scalarL
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat D))
      (Op.ref .nat [BD] "o_d"))

/-- The shared `mask=mask, other=0` load option. -/
def hgrnMaskOther (BD : Nat) : MaskOpt .real [BD] :=
  MaskOpt.maskOther (Op.ref .bool [BD] "mask") ((Op.const 0).broadcast [BD])

/-! ## One forward loop body, walked at the register level

`hgrnFwdOuterBody_step` executes the body `hgrnFwdOuterBody` — the very
statement list `fused_recurrent_hgrn_fwd_body_split` identifies as the kernel's
own `forRange` body — from a state whose registers are pinned at iteration `i`,
and pins them again at `i + 1`. It is the `hStep` obligation of
`VeriTile.Triton.forRange_inv`, minus the memory half.

Still **not** a correctness result: it says nothing about the `o` region the body
stores into, so the port's cross-step story remains the pinned `hPrev`
hypothesis of `fused_recurrent_hgrn_forward_step_closed_form`. What it does is
retire the hard half of the register walk.

Two modeling points the statement makes explicit:

* `b_h` is pinned as `some (if lane active then … else 0)`. Off-mask lanes
  really do hold `0`, not garbage: the seed is `Op.full 0` and both loads carry
  `other = 0`, so an inactive lane computes `0 · prev + 0`. Writing the carry as
  a bare closed form would be **false** there — `gVal`/`xVal` read out-of-range
  addresses off-mask.
* all three pointer registers advance by `D` per iteration, so all three are
  invariant clauses, not just the carry. -/

/-- Per-lane global channel index `i_d·BD + j`. -/
def hgrnChan (s0 : BlockState) (BD : Nat) (j : TileIndex [BD]) : Nat :=
  s0.pids 0 * BD + j.1.val

/-- Flat address of lane `j` at forward loop iteration `i`. -/
def hgrnAddr (s0 : BlockState) (T D BD i : Nat) (j : TileIndex [BD]) : Nat :=
  s0.pids 1 * T * D + hgrnChan s0 BD j + i * D

/-! ## The loop body's memory frame

`hgrnFwdOuterBody_step` pins the registers; this section pins what the body does
*not* touch. `forRange_inv` needs it because the closed form's `gVal`/`xVal` read
the `x` and `g` regions of the **initial** state, and those reads have to stay
valid across every iteration — which they do exactly when the body's only store
target `o` is a different region.

The body's single store lowers to a masked `foldl` of `writeMemTyped .real`,
which normalizes to plain `writeMem` at the `.real` channel (the carrier's `⊥`
fallback collapses to `WithBot.unbotD 0`), so the library's
`BlockState.foldl_writeMem_prop_masked_mem_preserve_other_region` applies
directly — no typed re-derivation is needed. -/

def dIndex (s : BlockState) (BD : Nat) (i : Fin BD) : Nat :=
  s.pids 0 * BD + i.val

def active (s : BlockState) (D BD : Nat) (i : Fin BD) : Prop :=
  dIndex s BD i < D

instance activeDecidable (s : BlockState) (D BD : Nat) (i : Fin BD) :
    Decidable (active s D BD i) := by
  unfold active
  infer_instance

def bhOffset (s : BlockState) (D BD : Nat) (i : Fin BD) : Nat :=
  s.pids 1 * D + dIndex s BD i

def outOffset (s : BlockState) (i_t T D BD : Nat) (i : Fin BD) : Nat :=
  (s.pids 1 * T + i_t) * D + dIndex s BD i

noncomputable def forwardStepValue
    (s : BlockState) (BHPrev X G : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  s.readMem G (outOffset s i_t T D BD i) *
    s.readMem BHPrev (bhOffset s D BD i) +
  s.readMem X (outOffset s i_t T D BD i)

/-! ## Genuine closed-form forward state (over the *input* regions)

The forward loop body is `b_h = g_t · b_h + x_t`, output `o_t = b_h`. Unrolling
the recurrence gives, for the state after processing time steps `0,1,…,n-1`,

```
b_h^(n)[i] = seed[i] · ∏_{j<n} g_j[i]  +  Σ_{t<n} x_t[i] · ∏_{t<j<n} g_j[i]
```

with `seed = h0` when `USE_INITIAL_STATE` else `0` — `hgrnStateClosed` below, a
standalone specification over the *input* regions `x, g, h0`, never a read-back
of the kernel's own output. The kernel's output at time row `i_t` is the
*post-update* state `b_h^(i_t + 1)`, and the optional final state is
`hgrnStateClosed` at `n = T`. -/

/-- `g_t[i]` at the kernel's exact time-row layout. -/
noncomputable def gVal (s : BlockState) (g : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem g (outOffset s t T D BD i)

/-- `x_t[i]` at the kernel's exact time-row layout. -/
noncomputable def xVal (s : BlockState) (x : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem x (outOffset s t T D BD i)

/-- Seeded initial state `b_h^(0)`: `h0` if `USE_INITIAL_STATE` else `0`. -/
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (D BD : Nat) (i : Fin BD) : ℝ :=
  if USE_INITIAL_STATE then s.readMem h0 (bhOffset s D BD i) else 0

/-- **Genuine closed form for the forward state after `n` steps**, channel `i`:
`seed · ∏_{j<n} g_j + Σ_{t<n} x_t · ∏_{t<j<n} g_j`. A standalone specification
over the input regions `x, g, h0` — never a read-back of the kernel's own
output. -/
noncomputable def hgrnStateClosed
    (s : BlockState) (x g h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (T D BD n : Nat) (i : Fin BD) : ℝ :=
  stateSeed s h0 USE_INITIAL_STATE D BD i *
      (∏ j ∈ Finset.range n, gVal s g T D BD j i) +
    ∑ t ∈ Finset.range n,
      xVal s x T D BD t i *
        (∏ j ∈ Finset.Ico (t + 1) n, gVal s g T D BD j i)

/-- **The forward state carry-fold recurrence.** Unrolling one step:
`b_h^(n+1) = g_n · b_h^(n) + x_n`. This is the exact closed-form counterpart of
the Python loop body `b_h = b_g * b_h + b_x`. -/
theorem hgrnStateClosed_succ
    (s : BlockState) (x g h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (T D BD n : Nat) (i : Fin BD) :
    hgrnStateClosed s x g h0 USE_INITIAL_STATE T D BD (n + 1) i
      = gVal s g T D BD n i *
          hgrnStateClosed s x g h0 USE_INITIAL_STATE T D BD n i
        + xVal s x T D BD n i := by
  unfold hgrnStateClosed
  rw [Finset.prod_range_succ, Finset.sum_range_succ]
  rw [show Finset.Ico (n + 1) (n + 1) = (∅ : Finset Nat) from by simp,
      Finset.prod_empty, mul_one]
  have hsum :
      (∑ t ∈ Finset.range n,
          xVal s x T D BD t i *
            ∏ j ∈ Finset.Ico (t + 1) (n + 1), gVal s g T D BD j i)
        = (∑ t ∈ Finset.range n,
            xVal s x T D BD t i *
              ∏ j ∈ Finset.Ico (t + 1) n, gVal s g T D BD j i)
          * gVal s g T D BD n i := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t ht
    simp only [Finset.mem_range] at ht
    rw [Finset.prod_Ico_succ_top (by omega : t + 1 ≤ n)]
    ring
  rw [hsum]; ring

/-- **Forward carry-fold step (genuine).** If the materialized previous-state
buffer `BHPrev` holds the genuine `i_t`-step folded state
`hgrnStateClosed(i_t)`, then one loop body — `forwardStepValue`, i.e.
`g_{i_t} · BHPrev + x_{i_t}` — produces exactly the genuine `(i_t+1)`-step
folded state `hgrnStateClosed(i_t + 1)`, the value stored to the output row. -/
theorem forwardStepValue_eq_hgrnStateClosed_succ
    (s : BlockState) (BHPrev X G h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (i_t T D BD : Nat)
    (hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G h0 USE_INITIAL_STATE T D BD i_t i)
    (i : Fin BD) :
    forwardStepValue s BHPrev X G i_t T D BD i
      = hgrnStateClosed s X G h0 USE_INITIAL_STATE T D BD (i_t + 1) i := by
  rw [hgrnStateClosed_succ]
  unfold forwardStepValue
  rw [hPrev i]
  simp only [gVal, xVal]

/-! ## Cross-step fold of the forward loop (genuine, register carry)

This is the piece the port's step face could not reach: the cross-step
induction, run over the kernel's **own** `Stmt.forRange` with the carry `b_h`
living in a *register* the whole way, exactly as the Python loop carries it.
No materialized `BHPrev` region, no per-step hypothesis. -/

/-- The loop's flat lane address is the `outOffset` the closed form reads. -/
theorem hgrnAddr_eq_outOffset (s0 : BlockState) (T D BD t : Nat)
    (j : TileIndex [BD]) :
    hgrnAddr s0 T D BD t j = outOffset s0 t T D BD j.1 := by
  simp only [hgrnAddr, hgrnChan, outOffset, dIndex]; ring

/-- Memory agreement transports to the real-channel readback. -/
theorem readMem_of_mem_eq {s t : BlockState} {r : RegionName} {off : Nat}
    (h : s.mem r off = t.mem r off) : s.readMem r off = t.readMem r off := by
  unfold BlockState.readMem; rw [h]

/-- Masked companion of `BlockState.foldl_writeMem_mem_preserve_unhit`: a masked
scatter leaves an address no *enabled* lane targets alone. The library ships
only the unmasked form. -/
theorem foldl_writeMem_masked_preserve_unhit {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (l : List α) (off : Nat) (ho : ∀ k ∈ l, offsetFn k ≠ off) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
          s).mem region off)
        = s.mem region off := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons, ih (fun k hk => ho k (List.mem_cons_of_mem hd hk))]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem,
          if_neg (fun hc => ho hd (List.mem_cons_self) hc.2.symm)]
      · rw [if_neg hP]

/-- The per-iteration lane address is injective in the lane. No side condition:
`hgrnChan` is `i_d·BD + j` with `j` ranging over `Fin BD`. -/
theorem hgrnAddr_injective (s0 : BlockState) (T D BD i : Nat) :
    Function.Injective (fun j : TileIndex [BD] => hgrnAddr s0 T D BD i j) := by
  rintro ⟨a, ⟨⟩⟩ ⟨b, ⟨⟩⟩ hab
  simp only [hgrnAddr, hgrnChan] at hab
  have : a.val = b.val := by omega
  simpa using Fin.ext this

/-- A row written at iteration `i` is never an *active* row of an earlier
iteration `t < i`: the active lanes of a row span less than one `D`-stride. -/
theorem hgrnAddr_ne_of_lt (s0 : BlockState) (T D BD t i : Nat)
    (ht : t < i) (j k : TileIndex [BD]) (hj : hgrnChan s0 BD j < D) :
    hgrnAddr s0 T D BD i k ≠ hgrnAddr s0 T D BD t j := by
  have h1 : t * D + D ≤ i * D := by
    have h2 : (t + 1) * D ≤ i * D := Nat.mul_le_mul_right D ht
    rwa [Nat.succ_mul] at h2
  simp only [hgrnAddr] at *
  omega

/-- **Forward loop invariant.** At iteration `i`: the lane mask and the three
row pointers are pinned at `i`, the register carry `b_h` holds the genuine
closed form `hgrnStateClosed i` on active lanes (`0` off-mask — see the step
lemma's note), and nothing outside the output region `o` has moved. -/
def HgrnFwdInv (s0 : BlockState) (x g o h0 : RegionName) (U : Bool)
    (T D BD n : Nat) (i : Nat) (s : BlockState) : Prop :=
  i ≤ n
  ∧ s.regs .bool [BD] "mask" = some ⟨fun j => decide (hgrnChan s0 BD j < D)⟩
  ∧ s.regs .ptr [BD] "p_x" = some ⟨fun j => (x, hgrnAddr s0 T D BD i j)⟩
  ∧ s.regs .ptr [BD] "p_g" = some ⟨fun j => (g, hgrnAddr s0 T D BD i j)⟩
  ∧ s.regs .ptr [BD] "p_o" = some ⟨fun j => (o, hgrnAddr s0 T D BD i j)⟩
  ∧ s.regs .real [BD] "b_h" = some ⟨fun j => some (if hgrnChan s0 BD j < D then
      hgrnStateClosed s0 x g h0 U T D BD i j.1 else 0)⟩
  ∧ (∀ r, r ≠ o → ∀ off, s.mem r off = s0.mem r off)
  ∧ (∀ t, t < i → ∀ j : TileIndex [BD], hgrnChan s0 BD j < D →
      s.readMem o (hgrnAddr s0 T D BD t j)
        = hgrnStateClosed s0 x g h0 U T D BD (t + 1) j.1)

/-! ## Offset-injectivity side conditions (dimension-general + Python shape)

The per-time output address `(i_bh·T + i_t)·D + i_d·BD + i` and the state
address `i_bh·D + i_d·BD + i` are injective in the lane `i` whenever `0 < BD`
(lanes are contiguous in the low digit). Honest structural side conditions; the
Python regression (`BD = 32`) satisfies them. -/

/-- **Dimension-general** per-time output address injectivity, given `0 < BD`. -/
theorem fused_recurrent_hgrn_out_offset_injective_general
    (s : BlockState) (i_t T D BD : Nat) (_hBD : 0 < BD) :
    Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i) := by
  intro a b h
  apply Fin.ext
  simp [outOffset, dIndex] at h
  omega

/-! ## Genuine closed-form backward scan (over the *input* regions)

The backward loop body (`fused_recurrent_hgrn_bwd_kernel`, py:105–126) is a
*reverse-time* scan: `b_dh` is seeded to `0` before the `range(T-1, -1, -1)`
loop and then, at reverse step `i`,

```
b_dh  ← b_dh + do_i      (py:115)      b_dx_i = b_dh            (py:116)
b_dg_i = b_dh · b_o(i)   (py:117)      b_dh  ← b_dh · g_i       (py:118)
```

The last line — the multiplication by the gate row — is what makes the backward
pass a scan. Unrolling the carry from the seed, the value of `b_dh` *entering*
reverse step `i` (before `b_dh += b_do`) is

```
carry(i)[d] = Σ_{i < t < T} do_t[d] · ∏_{i < j ≤ t} g_j[d]
```

(`hgrnBwdCarry`), so the row the kernel stores into `dx` is

```
dx_i[d] = carry(i)[d] + do_i[d] = Σ_{i ≤ t < T} do_t[d] · ∏_{i < j ≤ t} g_j[d]
```

(`hgrnBwdDx`) — a standalone specification over the *input* regions `do` and
`g`, never a read-back of the kernel's own output. `hgrnBwdCarry_pred` is the
exact closed-form counterpart of py:118, `hgrnBwdCarry_init` of the `tl.zeros`
seed. -/

/-- `do_t[d]` at the kernel's exact time-row layout. -/
noncomputable def doVal (s : BlockState) (DO : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem DO (outOffset s t T D BD i)

/-- **Genuine closed form of the backward carry entering reverse step `i_t`**:
`Σ_{i_t < t < T} do_t · ∏_{i_t < j ≤ t} g_j`. A specification over the input
regions `do`, `g` only. -/
noncomputable def hgrnBwdCarry (s : BlockState) (DO G : RegionName)
    (T D BD : Nat) (i_t : Nat) (i : Fin BD) : ℝ :=
  ∑ t ∈ Finset.Ico (i_t + 1) T,
    doVal s DO T D BD t i *
      (∏ j ∈ Finset.Ico (i_t + 1) (t + 1), gVal s G T D BD j i)

/-- **Genuine closed form of the `dx` row stored at reverse step `i_t`**:
`Σ_{i_t ≤ t < T} do_t · ∏_{i_t < j ≤ t} g_j` — the carry plus this step's
`do`. -/
noncomputable def hgrnBwdDx (s : BlockState) (DO G : RegionName)
    (T D BD : Nat) (i_t : Nat) (i : Fin BD) : ℝ :=
  ∑ t ∈ Finset.Ico i_t T,
    doVal s DO T D BD t i *
      (∏ j ∈ Finset.Ico (i_t + 1) (t + 1), gVal s G T D BD j i)

/-- `b_dh = b_dh + b_do` (py:115) at the closed-form level: the stored `dx` row
is the incoming carry plus this step's `do` row. -/
theorem hgrnBwdDx_eq_carry_add_do
    (s : BlockState) (DO G : RegionName) (T D BD i_t : Nat) (i : Fin BD)
    (hlt : i_t < T) :
    hgrnBwdDx s DO G T D BD i_t i
      = hgrnBwdCarry s DO G T D BD i_t i + doVal s DO T D BD i_t i := by
  unfold hgrnBwdDx hgrnBwdCarry
  rw [Finset.sum_eq_sum_Ico_succ_bot hlt,
      show Finset.Ico (i_t + 1) (i_t + 1) = (∅ : Finset Nat) from by simp,
      Finset.prod_empty, mul_one, add_comm]

/-- **★ The backward scan step (py:118, `b_dh = b_dh * b_g`) at the closed-form
level.** The carry entering reverse step `i_t` is the *next* step's stored `dx`
row multiplied by that step's gate row — the multiplication that makes the
backward pass a scan. -/
theorem hgrnBwdCarry_pred
    (s : BlockState) (DO G : RegionName) (T D BD i_t : Nat) (i : Fin BD) :
    hgrnBwdCarry s DO G T D BD i_t i
      = hgrnBwdDx s DO G T D BD (i_t + 1) i * gVal s G T D BD (i_t + 1) i := by
  unfold hgrnBwdCarry hgrnBwdDx
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro t ht
  simp only [Finset.mem_Ico] at ht
  rw [Finset.prod_eq_prod_Ico_succ_bot (by omega : i_t + 1 < t + 1)]
  ring

/-- The `b_dh = tl.zeros([BD])` seed (py:104) at the closed-form level: the
carry entering the loop's *first* reverse step `i = T − 1` is `0`. -/
theorem hgrnBwdCarry_init
    (s : BlockState) (DO G : RegionName) (T D BD : Nat) (i : Fin BD)
    (hT : 0 < T) :
    hgrnBwdCarry s DO G T D BD (T - 1) i = 0 := by
  unfold hgrnBwdCarry
  rw [show T - 1 + 1 = T from by omega, Finset.Ico_self]
  simp

noncomputable def bwdDxStepValue
    (s : BlockState) (DHPrev DO : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  s.readMem DHPrev (outOffset s i_t T D BD i) +
    s.readMem DO (outOffset s i_t T D BD i)

/-! ## The backward scan step as an executed slice

The single-step `dx` slice above takes the incoming carry from the fiction
region `DHPrev` and never touches the gate region, so `b_dh = b_dh * b_g`
(py:118) is invisible to it. The **two-step** slice below closes that hole: it
transcribes reverse iteration `i_t + 1`'s carry fold
(`b_dh = b_dh + b_do`, then `b_dh = b_dh * b_g`, both at row `i_t + 1`) followed
by iteration `i_t`'s `b_dh = b_dh + b_do` and `dx` store, so the scan
multiplication is *executed* and the gate region `G` enters the compute face.
Iteration `i_t + 1`'s own `dx`/`dg` stores are omitted from the slice: they
write `DX`/`DG`, which the loop never reads back, so their omission cannot
change the `dx` row this face is about. -/

/-- The arithmetic spec of the two-step `dx` body:
`(carry_{i_t+1} + do_{i_t+1}) · g_{i_t+1} + do_{i_t}`. -/
noncomputable def bwdDxTwoStepValue
    (s : BlockState) (DHPrev DO G : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  (s.readMem DHPrev (outOffset s (i_t + 1) T D BD i) +
        doVal s DO T D BD (i_t + 1) i) *
      gVal s G T D BD (i_t + 1) i
    + doVal s DO T D BD i_t i

/-- **Backward carry-fold step (genuine).** If the fiction carry region holds the
genuine carry entering reverse step `i_t + 1`, then executing that step's scan
fold (`+ do`, then `· g`) and step `i_t`'s `+ do` stores exactly the genuine
closed form `hgrnBwdDx(i_t)` — `Σ_{i_t ≤ t < T} do_t · ∏_{i_t < j ≤ t} g_j`,
over the input regions `do`, `g` only. -/
theorem bwdDxTwoStepValue_eq_hgrnBwdDx
    (s : BlockState) (DHPrev DO G : RegionName) (i_t T D BD : Nat)
    (hlt : i_t + 1 < T)
    (hCarry : ∀ i : Fin BD,
      s.readMem DHPrev (outOffset s (i_t + 1) T D BD i)
        = hgrnBwdCarry s DO G T D BD (i_t + 1) i)
    (i : Fin BD) :
    bwdDxTwoStepValue s DHPrev DO G i_t T D BD i
      = hgrnBwdDx s DO G T D BD i_t i := by
  unfold bwdDxTwoStepValue
  rw [hCarry i, ← hgrnBwdDx_eq_carry_add_do s DO G T D BD (i_t + 1) i hlt,
      ← hgrnBwdCarry_pred s DO G T D BD i_t i,
      hgrnBwdDx_eq_carry_add_do s DO G T D BD i_t i (by omega)]

/-- `b_o` at reverse step `i_t` — Python's three-way branch (py:108–113): the
**previous** output row `i_t − 1` when `i_t > 0`, else `h0` when
`USE_INITIAL_STATE`, else `0`. -/
noncomputable def bwdPrevOut (s : BlockState) (O H0 : RegionName)
    (USE_INITIAL_STATE : Bool) (i_t T D BD : Nat) (i : Fin BD) : ℝ :=
  if 0 < i_t then s.readMem O (outOffset s (i_t - 1) T D BD i)
  else if USE_INITIAL_STATE then s.readMem H0 (bhOffset s D BD i) else 0

noncomputable def bwdDgStepValue
    (s : BlockState) (DHPrev DO O H0 : RegionName) (USE_INITIAL_STATE : Bool)
    (i_t T D BD : Nat) (i : Fin BD) : ℝ :=
  (s.readMem DHPrev (outOffset s i_t T D BD i) +
      s.readMem DO (outOffset s i_t T D BD i)) *
    bwdPrevOut s O H0 USE_INITIAL_STATE i_t T D BD i

/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about four hand-cut slices, not about the launched
kernels.** Clauses 2–6 are `Realizes` facts about
`fused_recurrent_hgrn_forward_step_store_slice`,
`fused_recurrent_hgrn_bwd_dx_step_store_slice`,
`fused_recurrent_hgrn_bwd_dg_step_store_slice` and
`fused_recurrent_hgrn_bwd_dx_two_step_store_slice`; the launched forward surface
appears only in clause 1, which says nothing more than "it lowers to the
algorithm layer", and the launched backward surface does not appear at all. The
`STORE_FINAL_STATE` writeback has **no** correctness face here (see the module
docstring).

Parameterized over the symbolic time/feature/tile sizes `T D BD`, the step index
`i_t`, and both flags `USE_INITIAL_STATE STORE_FINAL_STATE`. The forward face is
realized against the closed form `hgrnStateClosed`, the backward `dx` scan face
against `hgrnBwdDx`, both over the *input* regions (never a read-back of the
kernel's own output):

1. the full HGRN forward surface lowers to the algorithm layer;
2. one forward **output** body realizes `hgrnStateClosed(i_t + 1)` — the unrolled
   recurrence `b_h = g·b_h + x` — given the *assumed* carry invariant
   `BHPrev = hgrnStateClosed(i_t)`;
3. one backward `dx` body realizes the genuine `bwdDxStepValue` (`dh_prev + do`);
4. one backward `dg` body realizes the genuine `bwdDgStepValue`
   `(dh_prev + do) · b_o`, with `b_o` Python's **three-way previous-row branch**
   (`bwdPrevOut`: output row `i_t − 1`, or `h0` at `i_t = 0` under
   `USE_INITIAL_STATE`, else `0`) transcribed as a DSL `if`;
5. **the backward scan step, executed.** Two consecutive reverse iterations —
   iteration `i_t + 1`'s `b_dh = (b_dh + b_do) · b_g` (py:115+118) then iteration
   `i_t`'s `b_dh + b_do` and `dx` store — realize the genuine closed form
   `hgrnBwdDx(i_t) = Σ_{i_t ≤ t < T} do_t · ∏_{i_t < j ≤ t} g_j` over the input
   regions `do`, `g`, given the *assumed* carry entering step `i_t + 1`
   (antecedents local to the clause: `i_t + 1 < T` and that carry);
6. the same two-step body at `i_t = T − 2`, where the assumed carry is only the
   `tl.zeros` **seed** (`DHPrev` row `T − 1` reads `0`, py:104) — so the `dx` row
   `do_{T-2} + do_{T-1}·g_{T-1}` is closed over `do`, `g` with no carry-value
   fiction left (antecedents local to the clause: `2 ≤ T` and the zero seed).

Honest structural side condition: `0 < BD` (contiguous lanes, giving offset
injectivity for every face). The flags flow through verbatim; clauses 2–6 hold
for every flag setting.

**The carry invariants are assumptions, not conclusions.** Nothing in this file
propagates the forward one: clause 2 writes `O` at the time-indexed
`outOffset s i_t`, whereas `hPrev` constrains `BHPrev` at the time-free
`bhOffset` — a different region at a different offset, with no bridging lemma;
likewise nothing proves the base case `hgrnStateClosed(0) = seed`. On the
backward side the carry *values* are now pinned to the closed form
`hgrnBwdCarry` and the fold `hgrnBwdCarry_pred`/`hgrnBwdCarry_init` is proved,
but the region `DHPrev` still only *holds* that value by hypothesis (clause 5) —
except at the loop's first fold, where the hypothesis degenerates to the literal
`tl.zeros` seed (clause 6). Chaining clause 5 across all `T` steps is the
unmodelled reverse fold. -/

end Correct_without_Rounding

/-! ## ════════ `⊨` IO face for the backward `dx` step store ════════

The summary above is stated per *declared write map*. This section restates the
backward `dx` step store on the audit-once IO surface
`MaskedTileShapedKernelIO₂.Implements` (`⊨`), which additionally pins the **flat
memory** placement.

Zero new library surface: the slice reads two `[BD]` tiles (`DHPrev` and `DO`) at the
same row window, adds them, and stores the sum to `DX` at that window, gated by the
kernel's own `offs_d < D`. Two inputs of the same shape is exactly what the
per-channel-shape skin's degenerate case is. -/

section IOFace

/-- `Op.SafeAt` of a masked region load, unfolded once. Used as a *term*: the
per-case equation does not fire under `simp` on a `.load` node whose dtype is the
computed `ComputeDType.fp32.eraseDType`. -/
private theorem safeAt_load_region_maskOther {d : TileDType} {shape : TileShape}
    (bounds : RegionBounds) (s : BlockState) (R : RegionName)
    (off : Op .nat shape) (m : Op .bool shape) (other : Op d shape)
    (h1 : Op.SafeAt bounds s off) (h2 : Op.SafeAt bounds s m)
    (h3 : Op.SafeAt bounds s other)
    (h4 : MemAccess.ActiveAddressSafe bounds (MemAccess.region R off) s
      ((MaskOpt.maskOther m other).Active s)) :
    Op.SafeAt bounds s
      (Op.load d (MemAccess.region R off) (MaskOpt.maskOther m other)) := by
  rw [Op.SafeAt]
  exact ⟨h1, ⟨h2, h3⟩, h4⟩

/-- `Op.SafeAt` of a register reference is vacuous. -/
private theorem safeAt_ref (bounds : RegionBounds) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    Op.SafeAt bounds s (Op.ref dtype shape name) := by
  rw [Op.SafeAt]
  trivial

/-- The row window is injective outright — it is `base + lane` — so the readback's
`hOutInj` precondition is a theorem here, not a hypothesis the headline carries. -/
private theorem outOffset_inj (s : BlockState) (i_t T D BD : Nat) :
    Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i) := by
  intro a b h
  simp only [outOffset, dIndex] at h
  exact Fin.ext (Nat.add_left_cancel (Nat.add_left_cancel h))

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

end IOFace

end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn

#guard_msgs (drop info) in
#auditModuleAxioms
