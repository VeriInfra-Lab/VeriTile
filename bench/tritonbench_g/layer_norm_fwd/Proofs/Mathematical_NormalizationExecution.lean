import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Normalization

/-! Success and exact register/memory result of the complete normalization
phase. Its incoming combined tile must still be derived from the intervening
second-stream and residual source statements for a whole-kernel theorem. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Normalization
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def statisticsState (f : Flags) (a : Inputs) (mean : RegionName)
    (M N block : Nat) (p : ℝ) (row : Nat) (s : BlockState) : BlockState :=
  centerVarianceState f a M N block p row
    (if f.rms then s else meanState f a mean M N p row s)

theorem statistics_exec (f : Flags) (a : Inputs) (mean : RegionName)
    (M N block : Nat) (p : ℝ) (row : Nat) (hN : N ≤ block) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Reduction.combinedTile f a M N block p row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) :
    stepStmt (statisticsStmt f.rms mean N block) s =
      some (statisticsState f a mean M N block p row s) := by
  have hcond : evalOp (.boolNot (.constBool f.rms)) s = some (Tile.scalar (!f.rms)) := by
    simp [evalOp, Tile.uop]
  simp only [statisticsStmt, stepStmt, hcond, bind, Option.bind, Tile.scalar]
  cases hr : f.rms
  · simp only [Bool.not_false, ↓reduceIte]
    rw [ProgramRewrite.append, mean_exec f a mean M N block p row hN s hx hrow,
      Option.bind_some]
    have hc := centerVariance_exec f a M N block p row hN
      (meanState f a mean M N p row s)
      (by simpa [meanState, Memory.writeCell] using hx)
      (by simpa [meanState, Memory.writeCell] using hcols)
      (by intro _; simp [meanState, Memory.writeCell])
    simpa only [statisticsState, hr, Bool.false_eq_true, ↓reduceIte] using hc
  · simp only [Bool.not_true, Bool.false_eq_true, ↓reduceIte]
    have hc := centerVariance_exec f a M N block p row hN s hx hcols
      (by intro h; simp [hr] at h)
    simpa only [statisticsState, hr, ↓reduceIte] using hc

theorem statisticsState_var (f : Flags) (a : Inputs) (mean : RegionName)
    (M N block : Nat) (p : ℝ) (row : Nat) (s : BlockState) :
    (statisticsState f a mean M N block p row s).regs .real [] "var" =
      some (Tile.scalar (some (Formula.variance f a M N p row))) := by
  simp [statisticsState, centerVarianceState]

theorem statisticsState_row (f : Flags) (a : Inputs) (mean : RegionName)
    (M N block : Nat) (p : ℝ) (row : Nat) (s : BlockState) :
    (statisticsState f a mean M N block p row s).regs .nat [] "row" =
      s.regs .nat [] "row" := by
  cases hr : f.rms <;> simp [statisticsState, centerVarianceState, meanState, Memory.writeCell, hr]

def state (f : Flags) (a : Inputs) (r : Regions) (M N block : Nat)
    (eps p : ℝ) (row : Nat) (s : BlockState) : BlockState :=
  Memory.writeCell
    ((statisticsState f a r.mean M N block p row s).setReg "rstd" .real []
      (Tile.scalar (some (Formula.rstd f a M N eps p row)))) r.rstd row
    (MemCell.real (Formula.rstd f a M N eps p row))

theorem phase_exec (f : Flags) (a : Inputs) (r : Regions) (M N block : Nat)
    (eps p : ℝ) (row : Nat) (hN : N ≤ block) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Reduction.combinedTile f a M N block p row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) :
    stepStmts (phase f r N block eps) s = some (state f a r M N block eps p row s) := by
  have hs := statistics_exec f a r.mean M N block p row hN s hx hcols hrow
  have hv := stepStmt_assign_eq_some (name := "rstd")
    (Reduction.rstd_op f a M N eps p row (statisticsState f a r.mean M N block p row s)
      (statisticsState_var f a r.mean M N block p row s))
  have hr : (statisticsState f a r.mean M N block p row s).regs .nat [] "row" =
      some (Tile.scalar row) := by rw [statisticsState_row, hrow]
  have hw := Memory.scalar_store r.rstd "rstd" row (Formula.rstd f a M N eps p row)
    ((statisticsState f a r.mean M N block p row s).setReg "rstd" .real []
      (Tile.scalar (some (Formula.rstd f a M N eps p row))))
    (by simpa using hr) (by simp)
  unfold phase rstdStmt scalarStore
  rw [stepStmts.cons_some hs, stepStmts.cons_some hv, stepStmts.cons_some hw, stepStmts.nil]
  rfl

theorem source_phase_exec (f : Flags) (a : Inputs) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (row : Nat) (hN : N ≤ block) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Reduction.combinedTile f a M N block p row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row)) :
    stepStmts (((Execution.source f r stride M N eps p block hBlock).body.drop 15).take 3) s =
      some (state f a r M N block eps p row s) := by
  rw [source_phase]
  exact phase_exec f a r M N block eps p row hN s hx hcols hrow

theorem state_mem (f : Flags) (a : Inputs) (r : Regions) (M N block : Nat)
    (eps p : ℝ) (row : Nat) (s : BlockState) (region : RegionName) (offset : Nat) :
    (state f a r M N block eps p row s).mem region offset =
      if region = r.rstd ∧ offset = row then MemCell.real (Formula.rstd f a M N eps p row)
      else if !f.rms && decide (region = r.mean ∧ offset = row) then
        MemCell.real (Formula.mean f a M N p row) else s.mem region offset := by
  cases hr : f.rms
  all_goals simp [state, statisticsState, centerVarianceState, meanState, Memory.writeCell, hr]

theorem state_rstd (f : Flags) (a : Inputs) (r : Regions) (M N block : Nat)
    (eps p : ℝ) (row : Nat) (s : BlockState) :
    (state f a r M N block eps p row s).regs .real [] "rstd" =
      some (Tile.scalar (some (Formula.rstd f a M N eps p row))) := by
  simp [state, Memory.writeCell]

theorem state_xbar (f : Flags) (a : Inputs) (r : Regions) (M N block : Nat)
    (eps p : ℝ) (row : Nat) (s : BlockState) :
    (state f a r M N block eps p row s).regs .real [block] "xbar" =
      some (Reduction.centeredTile f a M N block p row) := by
  simp [state, Memory.writeCell, statisticsState, centerVarianceState]

theorem state_x (f : Flags) (a : Inputs) (r : Regions) (M N block : Nat)
    (eps p : ℝ) (row : Nat) (s : BlockState) :
    (state f a r M N block eps p row s).regs .real [block] "x" = s.regs .real [block] "x" := by
  cases hr : f.rms
  all_goals simp [state, Memory.writeCell, statisticsState, centerVarianceState, meanState, hr]

theorem state_mean (f : Flags) (a : Inputs) (r : Regions) (M N block : Nat)
    (eps p : ℝ) (row : Nat) (s : BlockState) (hr : f.rms = Bool.false) :
    (state f a r M N block eps p row s).regs .real [] "mean" =
      some (Tile.scalar (some (Formula.mean f a M N p row))) := by
  simp [state, Memory.writeCell, statisticsState, centerVarianceState, meanState, hr]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Normalization
