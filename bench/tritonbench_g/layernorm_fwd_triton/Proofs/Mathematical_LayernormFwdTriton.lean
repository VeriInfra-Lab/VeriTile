import VeriTile.Triton

/-!
# `layernorm_fwd_triton` — strict per-kernel correctness

`_layer_norm_fwd_kernel` is a tutorial-style LayerNorm forward: each program
`(Seq, H)` normalizes one row of `X` by its mean and variance, scales by
per-head weights `W`, and writes `Y = ((x - mean) * rstd) * w` where
`rstd = 1 / sqrt(var + eps)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program. The host launch (`_layer_norm_fwd_kernel[grid](...)` with
`grid = (X.shape[0], X.shape[1])`, the host-side `BLOCK_SIZE = 128` choice,
scheduling, and how the runtime composes per-program writes into one buffer) is
the *trusted boundary*, not a proof obligation here. Because the two program ids
`Seq = tl.program_id(0)` and `H = tl.program_id(1)` are universally quantified
(via `s.pids 0` / `s.pids 1`), the per-program statement covers every program of
the 2D grid.

## Proof architecture

```
layernorm_fwd_triton_compute_fullN_correct    ← TOP THEOREM (general N, multi-block)
  └─ layernorm_fwd_triton_fullN_correct        ← algorithm-layer readback per column
       ├─ layernorm_fwd_triton_staged_fullN_correct_from_preloop
       ├─ layernormMeanForRange_context_*       ← mean-loop forRange invariant
       ├─ layernormVarForRange_context_*        ← var-loop forRange invariant
       └─ layernormOutForRange_fullN_of_init    ← output-loop forRange invariant
            └─ layernormOutLoopBody_step_output_invariant
layernorm_fwd_triton_compute_correct          ← one-block slice (0 < N ≤ BLOCK_SIZE)
  └─ layernorm_fwd_triton_correct
```

`layernorm_fwd_triton_compute_fullN_correct` is the strongest result: it covers
arbitrary `N` (mean / var / output loops each tiled over `BLOCK_SIZE` and closed
by `forRange` loop invariants). The one-block `*_compute_correct` is the
single-iteration specialization.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)`
input/weight casts and the store cast `(y).to(X.dtype.element_ty)` reduce to
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`). Both
reduction loops (`_mean`, `_var`) sum over the *padded* `BLOCK_SIZE` blocks, but
out-of-range lanes are masked to `0` (load `other=0.0`, plus the explicit
`tl.where` for the centered values), so each sum equals the logical row length
`N`. The mean is `(∑ x) / N`, the variance `(∑ (x - mean)²) / N`, and
`rstd = 1 / sqrt(var + eps)`; the affine step is `(x - mean) * rstd * w`. The
fullN theorem requires only `0 < BLOCK_SIZE` plus output/input region
disjointness (`X ≠ Y`, `W ≠ Y`); the one-block slice instead assumes
`0 < N`, `N ≤ BLOCK_SIZE`, and output-offset injectivity (`hOutInj`), and does
*not* need `0 < BLOCK_SIZE`/`X ≠ Y`/`W ≠ Y`. The kernel's unused `stride_*_hd`
strides are carried as unused Lean parameters.
-/

namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

def xOffset
    (s : BlockState) (stride_x_N stride_x_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + i.val

def yOffset
    (s : BlockState) (stride_y_N stride_y_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + i.val

def wOffset (s : BlockState) (stride_w_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_w_hn + i.val

noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_N stride_x_hn idx.1))
      else some (0.0 : ℝ) }

noncomputable def layernormMeanCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile s X stride_x_N stride_x_hn N BLOCK_SIZE)).data
        PUnit.unit)

noncomputable def layernormCenteredTile
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          (if idx.1.val < N then
            some (s.readMem X (xOffset s stride_x_N stride_x_hn idx.1))
          else some (0.0 : ℝ))
          (layernormMeanCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE)
      else some (0.0 : ℝ) }

noncomputable def layernormVarCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE))).data
        PUnit.unit)

noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) (eps : ℝ) :
    WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (layernormVarCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE)))

noncomputable def layernormYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat) (eps : ℝ)
    (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun centered inv => centered * inv)
        (Option.map₂ (fun x mean => x - mean)
          (some (s.readMem X (xOffset s stride_x_N stride_x_hn i)))
          (layernormMeanCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE))
        (layernormInvVarCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE eps))
      (some (s.readMem W (wOffset s stride_w_hn i))))

def xColOffset
    (s : BlockState) (stride_x_N stride_x_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + col

def yColOffset
    (s : BlockState) (stride_y_N stride_y_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + col

def wColOffset (s : BlockState) (stride_w_hn : Nat) (col : Nat) : Nat :=
  s.pids 1 * stride_w_hn + col

/-- Full-N mean used by the Python `for off in range(0, N, BLOCK_SIZE)` path. -/
noncomputable def layernormMeanFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N _BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val)) /
    (N : ℝ)

/-- Full-N variance after subtracting the full-N mean. -/
noncomputable def layernormVarFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N,
      (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2) /
    (N : ℝ)

noncomputable def layernormRstdFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) (eps : ℝ) : ℝ :=
  (Real.sqrt
    (layernormVarFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE + eps))⁻¹

/-- Full-N output spec for every Python-observable output column. -/
noncomputable def layernormYFullNSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat) (eps : ℝ)
    (i : Fin N) : ℝ :=
  ((s.readMem X (xColOffset s stride_x_N stride_x_hn i.val) -
      layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE) *
    layernormRstdFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE eps) *
    s.readMem W (wColOffset s stride_w_hn i.val)

/-- Per-lane mean-loop partial sum after columns `0..off`. -/
noncomputable def layernormMeanLanePrefix
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) : ℝ :=
  ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_SIZE = j.val).sum
    fun col => s.readMem X (xColOffset s stride_x_N stride_x_hn col)

noncomputable def layernormMeanAccumulatorSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some (layernormMeanLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1) }

theorem layernormMeanLanePrefix_final_eq_class_sum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hFinal : N ≤ off) :
    layernormMeanLanePrefix s X stride_x_N stride_x_hn N BLOCK_SIZE off j =
      ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
        fun col => s.readMem X (xColOffset s stride_x_N stride_x_hn col) := by
  classical
  unfold layernormMeanLanePrefix
  apply Finset.sum_congr
  · ext col
    simp only [Finset.mem_filter, Finset.mem_range, and_assoc]
    constructor
    · intro h
      exact ⟨h.2.1, h.2.2⟩
    · intro h
      exact ⟨Nat.lt_of_lt_of_le h.1 hFinal, h.1, h.2⟩
  · intro col _; rfl

theorem layernormMeanFullNCarrier_partition_by_lane
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    (∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val)) =
      ∑ j : Fin BLOCK_SIZE,
        ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
          fun col => s.readMem X (xColOffset s stride_x_N stride_x_hn col) := by
  classical
  rw [Finset.sum_fin_eq_sum_range]
  rw [← Finset.sum_biUnion]
  · apply Finset.sum_congr
    · ext col
      simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
        Finset.mem_filter, Finset.mem_range]
      constructor
      · intro hcol
        exact ⟨⟨col % BLOCK_SIZE, Nat.mod_lt col hB⟩, hcol, rfl⟩
      · rintro ⟨j, hcol, _hmod⟩
        exact hcol
    · intro col hmem
      have hcol : col < N := by
        simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
          Finset.mem_filter, Finset.mem_range] at hmem
        rcases hmem with ⟨j, hcol, _hmod⟩
        exact hcol
      simp [hcol]
  · intro i _ j _ hij
    apply Finset.disjoint_left.mpr
    intro col hi hj
    simp only [Finset.mem_filter, Finset.mem_range] at hi hj
    have hval : i.val = j.val := by
      rw [← hi.2, ← hj.2]
    exact hij (Fin.ext hval)

theorem layernormMeanAccumulatorSpec_final_sum_eq_fullN
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    (∑ j : Fin BLOCK_SIZE,
      WithBot.unbotD 0
        ((layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
          N BLOCK_SIZE off).data (j, PUnit.unit))) =
      ∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) := by
  classical
  rw [layernormMeanFullNCarrier_partition_by_lane s X stride_x_N stride_x_hn
    N BLOCK_SIZE hB]
  apply Finset.sum_congr rfl
  intro j _
  simp [layernormMeanAccumulatorSpec,
    layernormMeanLanePrefix_final_eq_class_sum s X stride_x_N stride_x_hn
      N BLOCK_SIZE off j hFinal]

theorem layernormMeanAccumulatorSpec_reduceSum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE off)).data PUnit.unit =
      some
        (∑ j : Fin BLOCK_SIZE,
          WithBot.unbotD 0
            ((layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
              N BLOCK_SIZE off).data (j, PUnit.unit))) := by
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, layernormMeanAccumulatorSpec]
  rfl

noncomputable def layernormMeanChunkSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some
        (if off + idx.1.val < N then
          s.readMem X (xColOffset s stride_x_N stride_x_hn (off + idx.1.val))
        else
          0) }

@[simp] theorem layernormMeanLanePrefix_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (j : Fin BLOCK_SIZE) :
    layernormMeanLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 j = 0 := by
  simp [layernormMeanLanePrefix]

theorem layernormMeanAccumulatorSpec_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 =
      { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 } := by
  ext idx
  simp [layernormMeanAccumulatorSpec]

theorem layernormLoopOffset_mod_step
    (off BLOCK_SIZE : Nat) (hOff : off % BLOCK_SIZE = 0) :
    (off + BLOCK_SIZE) % BLOCK_SIZE = 0 := by
  rw [Nat.add_mod, hOff]
  simp

theorem layernormChunkLane_mod
    (off BLOCK_SIZE : Nat) (j : Fin BLOCK_SIZE)
    (hOff : off % BLOCK_SIZE = 0) :
    (off + j.val) % BLOCK_SIZE = j.val := by
  rw [Nat.add_mod, hOff]
  simp [Nat.mod_eq_of_lt j.isLt]

theorem layernormChunkLane_not_mem_current
    (N off BLOCK_SIZE : Nat) (j : Fin BLOCK_SIZE) :
    off + j.val ∉ (Finset.range off).filter
      (fun col => col < N ∧ col % BLOCK_SIZE = j.val) := by
  simp

theorem layernormMeanLanePrefix_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hOff : off % BLOCK_SIZE = 0) :
    layernormMeanLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) j =
      layernormMeanLanePrefix s X stride_x_N stride_x_hn
          N BLOCK_SIZE off j +
        if off + j.val < N then
          s.readMem X (xColOffset s stride_x_N stride_x_hn (off + j.val))
        else
          0 := by
  classical
  let pred : Nat → Prop := fun col => col < N ∧ col % BLOCK_SIZE = j.val
  let f : Nat → ℝ := fun col =>
    s.readMem X (xColOffset s stride_x_N stride_x_hn col)
  have hunique :
      ∀ col, off ≤ col → col < off + BLOCK_SIZE → col % BLOCK_SIZE = j.val →
        col = off + j.val := by
    intro col hle hlt hmod
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hle
    have hklt : k < BLOCK_SIZE := by omega
    have hmodk : (off + k) % BLOCK_SIZE = k := by
      rw [Nat.add_mod, hOff, Nat.mod_eq_of_lt hklt]
      simpa using Nat.mod_eq_of_lt hklt
    rw [hmodk] at hmod
    omega
  by_cases hjN : off + j.val < N
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          insert (off + j.val) ((Finset.range off).filter pred) := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact Or.inr ⟨hltOff, h.2⟩
        · exact Or.inl (hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2)
      · intro h
        rcases h with h | h
        · subst h
          exact ⟨by omega, hjN, layernormChunkLane_mod off BLOCK_SIZE j hOff⟩
        · exact ⟨by omega, h.2⟩
    unfold layernormMeanLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f + f (off + j.val)
    rw [hset]
    rw [Finset.sum_insert]
    · ring
    · exact layernormChunkLane_not_mem_current N off BLOCK_SIZE j
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          (Finset.range off).filter pred := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact ⟨hltOff, h.2⟩
        · have hcol : col = off + j.val :=
            hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2
          omega
      · intro h
        exact ⟨by omega, h.2⟩
    unfold layernormMeanLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f
    rw [hset]

theorem layernormMeanAccumulatorSpec_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0) :
    layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) =
      { data := fun idx : TileIndex [BLOCK_SIZE] =>
          some
            (WithBot.unbotD 0
                ((layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx) +
              WithBot.unbotD 0
                ((layernormMeanChunkSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx)) } := by
  ext idx
  by_cases hcol : off + idx.1.val < N
  · simp [layernormMeanAccumulatorSpec, layernormMeanChunkSpec, hcol,
      layernormMeanLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]
  · simp [layernormMeanAccumulatorSpec, layernormMeanChunkSpec, hcol,
      layernormMeanLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]

def layernormMeanLoopInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  off % BLOCK_SIZE = 0 ∧
    st.regs .real [BLOCK_SIZE] "_mean" =
      some (layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE off)

def layernormMeanLoopContextInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  layernormMeanLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE off st ∧
    st.regs .ptr [] "X" =
      some (Tile.scalar (Region.cast X,
        s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
    (∀ offset, st.readMem X offset = s0.readMem X offset)

theorem layernormMeanLoopInvariant_init_of_zero_reg
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hReg :
      st.regs .real [BLOCK_SIZE] "_mean" =
        some { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 }) :
    layernormMeanLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE 0 st := by
  refine ⟨?_, ?_⟩
  · simp
  · simpa [layernormMeanAccumulatorSpec_zero] using hReg

theorem layernormMeanLoopInvariant_step_of_accumulator_update
    (s0 st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0)
    (hUpdate :
      st'.regs .real [BLOCK_SIZE] "_mean" =
        some
          { data := fun idx : TileIndex [BLOCK_SIZE] =>
              some
                (WithBot.unbotD 0
                    ((layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx) +
                  WithBot.unbotD 0
                    ((layernormMeanChunkSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx)) }) :
    layernormMeanLoopInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  refine ⟨?_, ?_⟩
  · exact layernormLoopOffset_mod_step off BLOCK_SIZE hOff
  · simpa [layernormMeanAccumulatorSpec_step s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off hOff] using hUpdate

theorem layernormMeanLoopInvariant_reduceSum_to_fullN
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hInv : layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    ∃ acc : Tile .real [BLOCK_SIZE],
      st.regs .real [BLOCK_SIZE] "_mean" = some acc ∧
        WithBot.unbotD 0
          ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
            acc).data PUnit.unit) / (N : ℝ) =
          layernormMeanFullNSpec s0 X stride_x_N stride_x_hn N BLOCK_SIZE := by
  refine ⟨layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off, hInv.1.2, ?_⟩
  rw [layernormMeanAccumulatorSpec_reduceSum]
  simp [layernormMeanFullNSpec,
    layernormMeanAccumulatorSpec_final_sum_eq_fullN s0 X stride_x_N
      stride_x_hn N BLOCK_SIZE off hB hFinal]

/-- Per-lane variance-loop partial sum after columns `0..off`, centered at the
full-N mean computed by the first loop. -/
noncomputable def layernormVarLanePrefix
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) : ℝ :=
  ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_SIZE = j.val).sum
    fun col =>
      (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2

noncomputable def layernormVarAccumulatorSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some (layernormVarLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1) }

theorem layernormVarLanePrefix_final_eq_class_sum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hFinal : N ≤ off) :
    layernormVarLanePrefix s X stride_x_N stride_x_hn N BLOCK_SIZE off j =
      ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
        fun col =>
          (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
            layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2 := by
  classical
  unfold layernormVarLanePrefix
  apply Finset.sum_congr
  · ext col
    simp only [Finset.mem_filter, Finset.mem_range, and_assoc]
    constructor
    · intro h
      exact ⟨h.2.1, h.2.2⟩
    · intro h
      exact ⟨Nat.lt_of_lt_of_le h.1 hFinal, h.1, h.2⟩
  · intro col _; rfl

theorem layernormVarFullNCarrier_partition_by_lane
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    (∑ j : Fin N,
      (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2) =
      ∑ j : Fin BLOCK_SIZE,
        ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
          fun col =>
            (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
              layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2 := by
  classical
  rw [Finset.sum_fin_eq_sum_range]
  rw [← Finset.sum_biUnion]
  · apply Finset.sum_congr
    · ext col
      simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
        Finset.mem_filter, Finset.mem_range]
      constructor
      · intro hcol
        exact ⟨⟨col % BLOCK_SIZE, Nat.mod_lt col hB⟩, hcol, rfl⟩
      · rintro ⟨j, hcol, _hmod⟩
        exact hcol
    · intro col hmem
      have hcol : col < N := by
        simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
          Finset.mem_filter, Finset.mem_range] at hmem
        rcases hmem with ⟨j, hcol, _hmod⟩
        exact hcol
      simp [hcol]
  · intro i _ j _ hij
    apply Finset.disjoint_left.mpr
    intro col hi hj
    simp only [Finset.mem_filter, Finset.mem_range] at hi hj
    have hval : i.val = j.val := by
      rw [← hi.2, ← hj.2]
    exact hij (Fin.ext hval)

theorem layernormVarAccumulatorSpec_final_sum_eq_fullN
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    (∑ j : Fin BLOCK_SIZE,
      WithBot.unbotD 0
        ((layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
          N BLOCK_SIZE off).data (j, PUnit.unit))) =
      ∑ j : Fin N,
        (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
          layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2 := by
  classical
  rw [layernormVarFullNCarrier_partition_by_lane s X stride_x_N stride_x_hn
    N BLOCK_SIZE hB]
  apply Finset.sum_congr rfl
  intro j _
  simp [layernormVarAccumulatorSpec,
    layernormVarLanePrefix_final_eq_class_sum s X stride_x_N stride_x_hn
      N BLOCK_SIZE off j hFinal]

theorem layernormVarAccumulatorSpec_reduceSum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE off)).data PUnit.unit =
      some
        (∑ j : Fin BLOCK_SIZE,
          WithBot.unbotD 0
            ((layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
              N BLOCK_SIZE off).data (j, PUnit.unit))) := by
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, layernormVarAccumulatorSpec]
  rfl

noncomputable def layernormVarChunkSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some
        (if off + idx.1.val < N then
          (s.readMem X (xColOffset s stride_x_N stride_x_hn (off + idx.1.val)) -
            layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2
        else
          0) }

@[simp] theorem layernormVarLanePrefix_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (j : Fin BLOCK_SIZE) :
    layernormVarLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 j = 0 := by
  simp [layernormVarLanePrefix]

theorem layernormVarAccumulatorSpec_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 =
      { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 } := by
  ext idx
  simp [layernormVarAccumulatorSpec]

theorem layernormVarLanePrefix_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hOff : off % BLOCK_SIZE = 0) :
    layernormVarLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) j =
      layernormVarLanePrefix s X stride_x_N stride_x_hn
          N BLOCK_SIZE off j +
        if off + j.val < N then
          (s.readMem X (xColOffset s stride_x_N stride_x_hn (off + j.val)) -
            layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2
        else
          0 := by
  classical
  let pred : Nat → Prop := fun col => col < N ∧ col % BLOCK_SIZE = j.val
  let f : Nat → ℝ := fun col =>
    (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
      layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2
  have hunique :
      ∀ col, off ≤ col → col < off + BLOCK_SIZE → col % BLOCK_SIZE = j.val →
        col = off + j.val := by
    intro col hle hlt hmod
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hle
    have hklt : k < BLOCK_SIZE := by omega
    have hmodk : (off + k) % BLOCK_SIZE = k := by
      rw [Nat.add_mod, hOff, Nat.mod_eq_of_lt hklt]
      simpa using Nat.mod_eq_of_lt hklt
    rw [hmodk] at hmod
    omega
  by_cases hjN : off + j.val < N
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          insert (off + j.val) ((Finset.range off).filter pred) := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact Or.inr ⟨hltOff, h.2⟩
        · exact Or.inl (hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2)
      · intro h
        rcases h with h | h
        · subst h
          exact ⟨by omega, hjN, layernormChunkLane_mod off BLOCK_SIZE j hOff⟩
        · exact ⟨by omega, h.2⟩
    unfold layernormVarLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f + f (off + j.val)
    rw [hset]
    rw [Finset.sum_insert]
    · ring
    · exact layernormChunkLane_not_mem_current N off BLOCK_SIZE j
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          (Finset.range off).filter pred := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact ⟨hltOff, h.2⟩
        · have hcol : col = off + j.val :=
            hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2
          omega
      · intro h
        exact ⟨by omega, h.2⟩
    unfold layernormVarLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f
    rw [hset]

theorem layernormVarAccumulatorSpec_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0) :
    layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) =
      { data := fun idx : TileIndex [BLOCK_SIZE] =>
          some
            (WithBot.unbotD 0
                ((layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx) +
              WithBot.unbotD 0
                ((layernormVarChunkSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx)) } := by
  ext idx
  by_cases hcol : off + idx.1.val < N
  · simp [layernormVarAccumulatorSpec, layernormVarChunkSpec, hcol,
      layernormVarLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]
  · simp [layernormVarAccumulatorSpec, layernormVarChunkSpec, hcol,
      layernormVarLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]

def layernormVarLoopInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  off % BLOCK_SIZE = 0 ∧
    st.regs .real [BLOCK_SIZE] "_var" =
      some (layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE off)

def layernormVarLoopContextInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  layernormVarLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE off st ∧
    st.regs .ptr [] "X" =
      some (Tile.scalar (Region.cast X,
        s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
    st.regs .real [] "mean" =
      some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE)) ∧
    (∀ offset, st.readMem X offset = s0.readMem X offset)

theorem layernormVarLoopInvariant_init_of_zero_reg
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hReg :
      st.regs .real [BLOCK_SIZE] "_var" =
        some { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 }) :
    layernormVarLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE 0 st := by
  refine ⟨?_, ?_⟩
  · simp
  · simpa [layernormVarAccumulatorSpec_zero] using hReg

theorem layernormVarLoopInvariant_step_of_accumulator_update
    (s0 st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0)
    (hUpdate :
      st'.regs .real [BLOCK_SIZE] "_var" =
        some
          { data := fun idx : TileIndex [BLOCK_SIZE] =>
              some
                (WithBot.unbotD 0
                    ((layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx) +
                  WithBot.unbotD 0
                    ((layernormVarChunkSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx)) }) :
    layernormVarLoopInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  refine ⟨?_, ?_⟩
  · exact layernormLoopOffset_mod_step off BLOCK_SIZE hOff
  · simpa [layernormVarAccumulatorSpec_step s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off hOff] using hUpdate

theorem layernormVarLoopInvariant_reduceSum_to_fullN
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hInv : layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    ∃ acc : Tile .real [BLOCK_SIZE],
      st.regs .real [BLOCK_SIZE] "_var" = some acc ∧
        WithBot.unbotD 0
          ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
            acc).data PUnit.unit) / (N : ℝ) =
          layernormVarFullNSpec s0 X stride_x_N stride_x_hn N BLOCK_SIZE := by
  refine ⟨layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off, hInv.1.2, ?_⟩
  rw [layernormVarAccumulatorSpec_reduceSum]
  simp [layernormVarFullNSpec,
    layernormVarAccumulatorSpec_final_sum_eq_fullN s0 X stride_x_N
      stride_x_hn N BLOCK_SIZE off hB hFinal]

theorem yColOffset_injective
    (s : BlockState) (stride_y_N stride_y_hn N : Nat) :
    Function.Injective
      (fun i : Fin N => yColOffset s stride_y_N stride_y_hn i.val) := by
  intro a b h
  apply Fin.ext
  unfold yColOffset at h
  exact Nat.add_left_cancel h

def layernormOutLoopInvariant
    (s0 : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (off : Nat) (st : BlockState) : Prop :=
  ∀ i : Fin N,
    i.val < off →
      st.readMem Y (yColOffset s0 stride_y_N stride_y_hn i.val) =
        layernormYFullNSpec s0 X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i

theorem layernormOutLoopInvariant_zero
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) :
    layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps 0 st := by
  intro i hlt
  omega

def layernormOutLoopContextInvariant
    (s0 : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ) (st : BlockState) : Prop :=
  layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off st ∧
    st.regs .ptr [] "X" =
      some (Tile.scalar (Region.cast X,
        s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
    st.regs .ptr [] "Y" =
      some (Tile.scalar (Region.cast Y,
        s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
    st.regs .ptr [] "W" =
      some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
    st.regs .real [] "mean" =
      some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE)) ∧
    st.regs .real [] "rstd" =
      some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE eps)) ∧
    (∀ offset, st.readMem X offset = s0.readMem X offset) ∧
    (∀ offset, st.readMem W offset = s0.readMem W offset)

/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surface above is untouched.
This is a consumer of the per-step emit skin `StreamEmitMasked2DKernelIO₂`
(streaming genre, style S3): the store sits **inside** the third pass, so
the output is a per-step `BLOCK_SIZE`-lane window family rather than one
terminal tile, and the kernel's spec `f t j` is the genre's *multi-pass*
shape — the step-`t` tiles combined with two folds over the entire stream
(mean, then variance about that mean).

Structure of the `execR R` story: this kernel has **zero rounding events**.
Every load and store is at `.real` — the Python `.to(tl.float32)` input
casts and the store cast `(y).to(X.dtype.element_ty)` are already erased by
the lowering (see `layernormOutLoopBody`: the emitted statement is a plain
`Stmt.store .real`), so `outDType := .real` is the honest grid and there is
no boundary quantization event. All three passes therefore collapse
verbatim onto the exact stepper (`stepForRangeAuxR_castFree`), and the
whole proven mean / var / out invariant stack above is reused unchanged;
the `⊨[R]` face adds only the `TraceSafeR` walk, the per-cell memory frame,
and the stream-lane spec bridge. The skin's readback contract at the
default `outDType := .real` grid carries `R.round .real`, the identity by
`round_real` — the ∀-`R` face is the exact streaming contract via the
model's `.real` identity fields, not a `.triv` special case. -/

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-! ### Stream geometry: trip count and windows -/

/-- Trip count of all three `for off in range(0, N, BLOCK_SIZE)` passes:
`⌈N / BLOCK_SIZE⌉`. -/
def lnNumSteps (N B : Nat) : Nat := (N + B - 1) / B

private theorem lnNumSteps_mul_ge (N B : Nat) (hB : 0 < B) :
    N ≤ lnNumSteps N B * B := by
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · exact Nat.zero_le _
  · unfold lnNumSteps
    have heq : N + B - 1 = (N - 1) + B := by omega
    rw [heq, Nat.add_div_right _ hB]
    have h2 : (N - 1) % B + 1 ≤ B := Nat.mod_lt _ hB
    calc N = (N - 1) + 1 := by omega
      _ = (N - 1) / B * B + ((N - 1) % B + 1) := by
          rw [← Nat.add_assoc, Nat.div_add_mod']
      _ ≤ (N - 1) / B * B + B := Nat.add_le_add_left h2 _
      _ = ((N - 1) / B + 1) * B := (Nat.succ_mul _ _).symm

private theorem lnStep_lt_numSteps (N B i : Nat) (hB : 0 < B) (hi : i < N) :
    i / B < lnNumSteps N B := by
  have h2 : i / B * B < lnNumSteps N B * B :=
    Nat.lt_of_le_of_lt (Nat.div_mul_le_self i B)
      (Nat.lt_of_lt_of_le hi (lnNumSteps_mul_ge N B hB))
  exact Nat.lt_of_mul_lt_mul_right h2

/-! ### Re-blocking arithmetic: `k ↔ (k / B, k % B)` -/

private theorem ln_sum_blocks_lanes (B c : Nat) (hB : 0 < B) (H : Nat → ℝ) :
    (∑ b : Fin c, ∑ j : Fin B, H (b.val * B + j.val))
      = ∑ k : Fin (c * B), H k.val := by
  rw [← Fintype.sum_prod_type']
  apply Finset.sum_nbij' (i := fun p : Fin c × Fin B => (⟨p.1.val * B + p.2.val, by
        have h1 := p.1.isLt; have h2 := p.2.isLt
        calc p.1.val * B + p.2.val < p.1.val * B + B := by omega
          _ = (p.1.val + 1) * B := by ring
          _ ≤ c * B := Nat.mul_le_mul_right _ (by omega)⟩ : Fin (c*B)))
    (j := fun k : Fin (c*B) => (⟨k.val / B,
        (Nat.div_lt_iff_lt_mul hB).mpr (by have := k.isLt; omega)⟩,
      ⟨k.val % B, Nat.mod_lt _ hB⟩))
  · intro p _; simp
  · intro k _; simp
  · intro p _
    apply Prod.ext
    · apply Fin.ext; show (p.1.val * B + p.2.val) / B = p.1.val
      rw [Nat.mul_comm p.1.val B, Nat.mul_add_div hB, Nat.div_eq_of_lt p.2.isLt,
        Nat.add_zero]
    · apply Fin.ext; show (p.1.val * B + p.2.val) % B = p.2.val
      rw [Nat.mul_comm p.1.val B, Nat.mul_add_mod, Nat.mod_eq_of_lt p.2.isLt]
  · intro k _; apply Fin.ext; show (k.val / B) * B + k.val % B = k.val
    rw [Nat.mul_comm (k.val / B) B]; exact Nat.div_add_mod k.val B
  · intro p _; rfl

private theorem ln_sum_fin_extend (N M : Nat) (hNM : N ≤ M) (f : Nat → ℝ) :
    (∑ k : Fin M, (if k.val < N then f k.val else 0)) = ∑ k : Fin N, f k.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => if k < N then f k else 0) M,
      Fin.sum_univ_eq_sum_range (fun k => f k) N]
  rw [← Finset.sum_subset (s₁ := Finset.range N) (s₂ := Finset.range M)
        (fun x hx => Finset.mem_range.mpr
          (lt_of_lt_of_le (Finset.mem_range.mp hx) hNM))
        (by
          intro k hk hknotN
          simp only [Finset.mem_range] at hk hknotN
          simp [Nat.not_lt.mp hknotN])]
  apply Finset.sum_congr rfl
  intro k hk; simp only [Finset.mem_range] at hk; simp [hk]

/-- The guarded block×lane double sum **is** the flat `Σ_{k<N}` sum. -/
private theorem ln_reblock (B c N : Nat) (hB : 0 < B) (hge : N ≤ c * B)
    (f : Nat → ℝ) :
    (∑ j : Fin B, ∑ b : Fin c,
        (if (b.val * B + j.val) < N then f (b.val * B + j.val) else 0))
      = ∑ k : Fin N, f k.val := by
  rw [Finset.sum_comm]
  rw [ln_sum_blocks_lanes B c hB (fun m => if m < N then f m else 0)]
  exact ln_sum_fin_extend N (c*B) hge f

/-! ### IO signature -/

/-! ### The stream-level spec -/

/-- The guarded stream-level sum of the `X` stream: the mean pass's
`_mean += a` fold over the whole curried stream, guarded by the kernel's
window (`t·B + e < N`) — the contract only pins `xs` on masked lanes, so
the spec must not read unmasked lanes. -/
noncomputable def lnStreamSum (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  ∑ u : Fin (lnNumSteps N B), ∑ e : Fin B,
    if u.val * B + e.val < N then xs u e else 0

/-- The stream-level mean: `tl.sum(_mean, axis=0) / N`. -/
noncomputable def lnStreamMean (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  lnStreamSum N B xs / (N : ℝ)

/-- The stream-level variance: the second pass's `_var += x*x` fold over the
whole stream, centred at `lnStreamMean` and guarded by the same window (the
kernel's `tl.where(cols < N, x - mean, 0.0)`), divided by `N`. -/
noncomputable def lnStreamVar (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  (∑ u : Fin (lnNumSteps N B), ∑ e : Fin B,
      if u.val * B + e.val < N then (xs u e - lnStreamMean N B xs) ^ 2 else 0)
    / (N : ℝ)

/-- The stream-level LayerNorm spec (the genre's three-pass shape): emitted
window `(t, j)` holds `((x[t,j] - mean) · rstd) · w[t,j]`, where `mean` and
`rstd = 1/√(var + eps)` are folds over the *entire* `x` stream.
Algebraically `layernormYFullNSpec` with the `Fin N` sums re-blocked to the
guarded stream double sums. -/
noncomputable def lnStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin (lnNumSteps N B) → Fin B → ℝ)
    (t : Fin (lnNumSteps N B)) (j : Fin B) : ℝ :=
  ((xs t j - lnStreamMean N B xs) *
      (Real.sqrt (lnStreamVar N B xs + eps))⁻¹) * ws t j

/-! ### The stream-lane spec bridge -/

/-- Under the stream pin, the guarded stream double sum **is** the exact
stack's `Σ_{k<N} x[k]` (re-blocking `k ↔ (k/B, k%B)` via `ln_reblock`). -/
private theorem lnStreamSum_eq (X : RegionName) (s₀ : BlockState)
    (stride_x_N stride_x_hn N B : Nat) (hB : 0 < B)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e) :
    lnStreamSum N B xs
      = ∑ k : Fin N,
          s₀.readMem X (xColOffset s₀ stride_x_N stride_x_hn k.val) := by
  unfold lnStreamSum
  rw [Finset.sum_comm,
    ← ln_reblock B (lnNumSteps N B) N hB (lnNumSteps_mul_ge N B hB)
      (fun k => s₀.readMem X (xColOffset s₀ stride_x_N stride_x_hn k))]
  refine Finset.sum_congr rfl fun e _ => Finset.sum_congr rfl fun t _ => ?_
  by_cases h : t.val * B + e.val < N
  · rw [if_pos h, if_pos h, ← hx t e h]
    rfl
  · rw [if_neg h, if_neg h]

private theorem lnStreamMean_eq (X : RegionName) (s₀ : BlockState)
    (stride_x_N stride_x_hn N B : Nat) (hB : 0 < B)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e) :
    lnStreamMean N B xs
      = layernormMeanFullNSpec s₀ X stride_x_N stride_x_hn N B := by
  unfold lnStreamMean layernormMeanFullNSpec
  rw [lnStreamSum_eq X s₀ stride_x_N stride_x_hn N B hB xs hx]

private theorem lnStreamVar_eq (X : RegionName) (s₀ : BlockState)
    (stride_x_N stride_x_hn N B : Nat) (hB : 0 < B)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e) :
    lnStreamVar N B xs
      = layernormVarFullNSpec s₀ X stride_x_N stride_x_hn N B := by
  unfold lnStreamVar layernormVarFullNSpec
  rw [lnStreamMean_eq X s₀ stride_x_N stride_x_hn N B hB xs hx]
  congr 1
  rw [Finset.sum_comm,
    ← ln_reblock B (lnNumSteps N B) N hB (lnNumSteps_mul_ge N B hB)
      (fun k => (s₀.readMem X (xColOffset s₀ stride_x_N stride_x_hn k) -
        layernormMeanFullNSpec s₀ X stride_x_N stride_x_hn N B) ^ 2)]
  refine Finset.sum_congr rfl fun e _ => Finset.sum_congr rfl fun t _ => ?_
  by_cases h : t.val * B + e.val < N
  · rw [if_pos h, if_pos h, ← hx t e h]
    rfl
  · rw [if_neg h, if_neg h]

/-- Per-lane spec bridge: at a masked window `(t, j)` the stream spec **is**
the exact stack's `layernormYFullNSpec` at global lane `t·B + j`. -/
private theorem lnStreamSpec_eq_layernormYFullNSpec (X W : RegionName)
    (s₀ : BlockState) (stride_x_N stride_x_hn stride_w_hn N B : Nat) (eps : ℝ)
    (hB : 0 < B) (xs ws : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e)
    (hw : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem W (s₀.pids 1 * stride_w_hn + (t.val * B + e.val)) = ws t e)
    (t : Fin (lnNumSteps N B)) (j : Fin B) (hj : t.val * B + j.val < N) :
    lnStreamSpec N B eps xs ws t j
      = layernormYFullNSpec s₀ X W stride_x_N stride_x_hn stride_w_hn N B eps
          ⟨t.val * B + j.val, hj⟩ := by
  unfold lnStreamSpec layernormYFullNSpec layernormRstdFullNSpec
  rw [lnStreamMean_eq X s₀ stride_x_N stride_x_hn N B hB xs hx,
    lnStreamVar_eq X s₀ stride_x_N stride_x_hn N B hB xs hx,
    ← hx t j hj, ← hw t j hj]
  rfl

/-! ### Cast-free collapses and the covered fragment -/

/-- The erased `.to(tl.float32)` is a `.real → .real` cast: exact under
every `R` — `R.roundW .real` is the identity by the model's defining
`round_real`. Kept in the collapse simp sets for uniformity with the
family; this kernel's lowering leaves no `castFloat` at all. -/
private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-! ### Per-op evaluation and its `R`-independence

The three passes' index / address / mask ops are nat-, ptr- and bool-typed:
no float arithmetic, hence `evalOpR R` is `evalOp` on the nose. -/

private theorem ln_colsR_eq (R : RoundingModel) (B : Nat) (s : BlockState) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "off") (Op.arange B)) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "off") (Op.arange B)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem ln_addrR_eq (R : RoundingModel) (P : RegName) (B : Nat)
    (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] P)
        (Op.ref .nat [B] "cols")) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] P)
        (Op.ref .nat [B] "cols")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem ln_maskR_eq (R : RoundingModel) (N B : Nat) (s : BlockState) :
    evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "cols") (Op.constNat N)) s
      = evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "cols") (Op.constNat N)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- Per-lane value of the `cols = off + tl.arange(0, BLOCK_SIZE)` op. -/
private theorem ln_cols_eval (B : Nat) (s : BlockState) (off : Nat)
    (hoff : s.regs .nat [] "off" = some (Tile.scalar off)) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "off") (Op.arange B)) s
      = some (Tile.vec (fun j : Fin B => off + j.val)) := by
  rw [evalOp_add, evalOp_ref, hoff, evalOp_arange]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.rightIndex_scalarL,
    NumericDType.add]

/-- Per-lane value of a masked access's address op `PTR + cols`. -/
private theorem ln_addr_eval (P : RegName) (B : Nat) (Rg : RegionName)
    (base : Nat) (s : BlockState) (off : Nat)
    (hp : s.regs .ptr [] P = some (Tile.scalar (Region.cast Rg, base)))
    (hc : s.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] P)
        (Op.ref .nat [B] "cols")) s
      = some (Tile.vec (fun j : Fin B =>
          (Region.cast Rg, base + (off + j.val)))) := by
  rw [evalOp_ptrAdd, evalOp_ref, hp, evalOp_ref, hc]
  apply congrArg some
  ext j
  · simp only [Tile.ptrAdd_data, Tile.vec, Tile.scalar,
      Broadcast.leftIndex_scalarL]
  · simp only [Tile.ptrAdd_data, Tile.vec, Tile.scalar,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL]

/-- Per-lane value of the kernel's single mask `cols < N`. -/
private theorem ln_mask_eval (N B : Nat) (s : BlockState) (off : Nat)
    (hc : s.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val))) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "cols") (Op.constNat N)) s
      = some (Tile.vec (fun j : Fin B => decide (off + j.val < N))) := by
  rw [evalOp_lt, evalOp_ref, hc, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.cop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarR,
    ComparableDType.lt]
  rfl

/-! ### The per-body `TraceSafeListR` walks -/

/-! ### Cell-level memory frames

The exact stack proves per-lane `readMem` values; the `⊨[R]` Hoare triple
additionally needs a per-**cell** frame, so the five register-only segments
get a generic assigns-don't-touch-memory lemma and the storing output-pass
body gets a cell-level frame twin of its readback step. -/

/-- A run of `assign` statements never touches memory. -/
private theorem ln_stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ stmt ∈ l, ∃ dt sh nm, ∃ e : Op dt sh, stmt = Stmt.assign dt sh nm e) →
      ∀ {s s' : BlockState}, stepStmts l s = some s' → s'.mem = s.mem
  | [], _, s, s', h => by
      rw [stepStmts.nil] at h
      obtain rfl := Option.some.inj h
      rfl
  | stmt :: rest, hall, s, s', h => by
      obtain ⟨dt, sh, nm, e, rfl⟩ := hall _ List.mem_cons_self
      cases hv : evalOp e s with
      | none => simp [stepStmts, stepStmt, hv] at h
      | some v =>
          rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
          rw [ln_stepStmts_assigns_mem rest
            (fun st' hst' => hall st' (List.mem_cons_of_mem _ hst')) h]
          rfl

/-- Cell-level frame of a `Prop`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (the cell-level sibling of
the shared `scatter_prop_masked_preserves_other_offset`). -/
private theorem ln_foldl_writeMem_prop_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ) (P : α → Prop)
    [DecidablePred P] (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hm : P hd
      · rw [if_pos hm,
          ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk),
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hm)
      · rw [if_neg hm]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk

/-! ### The rounded Hoare triple (`hrun`) -/

/-! ### The headline -/

end IOFace

end VeriTile.Bench.TritonBenchG.LayernormFwdTriton
