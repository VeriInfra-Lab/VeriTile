import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_SecondStream

/-! The source residual input and residual output statements, with successful
execution and complete tagged-cell frame. Pointer hypotheses are conditional
on the corresponding source flag; inactive pointers need not be initialized. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def residualState (enabled : Bool) (region : RegionName) (start N block : Nat)
    (x : Fin block → ℝ) (s : BlockState) : BlockState :=
  if enabled then
    (s.setReg "residual" .real [block]
      (Execution.realTile (rowValues region start N block s))).setReg "x" .real [block]
      (Execution.realTile (fun j => x j + rowValues region start N block s j))
  else s

theorem residualStmt_exec (enabled : Bool) (region : RegionName) (start N block : Nat)
    (x : Fin block → ℝ) (s : BlockState)
    (hptr : enabled = Bool.true → s.regs .ptr [] "RESIDUAL" = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x" = some (Execution.realTile x)) :
    stepStmt (residualStmt enabled N block) s =
      some (residualState enabled region start N block x s) := by
  unfold residualStmt
  rw [step_if_const]
  cases enabled
  · rfl
  · simp only [↓reduceIte]
    have hl := stepStmt_assign_eq_some (name := "residual")
      (rowLoadOp_eval "RESIDUAL" region start N block s (hptr rfl) hcols)
    have ha := addStmt_exec "residual" block x (rowValues region start N block s)
      (s.setReg "residual" .real [block] (Execution.realTile (rowValues region start N block s)))
      (by simpa using hx) (by simp)
    rw [stepStmts.cons_some hl, stepStmts.cons_some ha, stepStmts.nil]
    rfl

theorem residualState_x (enabled : Bool) (region : RegionName) (start N block : Nat)
    (x : Fin block → ℝ) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Execution.realTile x)) :
    (residualState enabled region start N block x s).regs .real [block] "x" =
      some (Execution.realTile (fun j => if enabled then
        x j + rowValues region start N block s j else x j)) := by
  cases enabled <;> simp [residualState, hx]

theorem residualState_mem (enabled : Bool) (region : RegionName) (start N block : Nat)
    (x : Fin block → ℝ) (s : BlockState) :
    (residualState enabled region start N block x s).mem = s.mem := by
  cases enabled <;> rfl

theorem residualState_regs (enabled : Bool) (region : RegionName) (start N block : Nat)
    (x : Fin block → ℝ) (s : BlockState) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hx : name ≠ "x") (hr : name ≠ "residual") :
    (residualState enabled region start N block x s).regs dtype shape name = s.regs dtype shape name := by
  cases enabled <;> simp [residualState, hx, hr]

def realStoreStmt (pointer name : RegName) (N block : Nat) : Stmt :=
  .store .real [block]
    (.ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [block] "cols")))
    (.ref .real [block] name)
    (.mask (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N)))

theorem realStoreStmt_exec (pointer name : RegName) (region : RegionName) (start N block : Nat)
    (values : Nat → ℝ) (s : BlockState)
    (hptr : s.regs .ptr [] pointer = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hv : s.regs .real [block] name = some (Execution.realTile (fun j => values j.val))) :
    stepStmt (realStoreStmt pointer name N block) s =
      some (Memory.scatter region start N block (fun col => MemCell.real (values col)) s) := by
  simp [realStoreStmt, stepStmt, evalOp, hptr, hcols, hv, Tile.ptrAdd,
    Tile.cop, ComparableDType.lt, Execution.realTile, Memory.scatter_fold, Memory.real_store]
  rfl

def residualOutState (enabled : Bool) (region : RegionName) (start N block : Nat)
    (values : Nat → ℝ) (s : BlockState) : BlockState :=
  if enabled then Memory.scatter region start N block (fun col => MemCell.real (values col)) s else s

theorem residualOutStmt_exec (enabled : Bool) (region : RegionName) (start N block : Nat)
    (values : Nat → ℝ) (s : BlockState)
    (hptr : enabled = Bool.true → s.regs .ptr [] "RESIDUAL_OUT" = some (Tile.scalar (region, start)))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x" = some (Execution.realTile (fun j => values j.val))) :
    stepStmt (residualOutStmt enabled N block) s =
      some (residualOutState enabled region start N block values s) := by
  change stepStmt (.ifThen (.constBool enabled)
    [realStoreStmt "RESIDUAL_OUT" "x" N block]) s = _
  rw [step_if_const]
  cases enabled
  · rfl
  · simp only [↓reduceIte]
    rw [exec_single]
    exact realStoreStmt_exec "RESIDUAL_OUT" "x" region start N block values s (hptr rfl) hcols hx

theorem residualOutState_mem (enabled : Bool) (region : RegionName) (start N block : Nat)
    (values : Nat → ℝ) (s : BlockState) (hN : N ≤ block) (r : RegionName) (offset : Nat) :
    (residualOutState enabled region start N block values s).mem r offset =
      if enabled && decide (r = region ∧ Contract.inRow start N offset)
      then MemCell.real (values (offset-start)) else s.mem r offset := by
  cases enabled <;> simp [residualOutState, Memory.scatter_cell _ _ _ _ hN]

theorem residualOutState_regs (enabled : Bool) (region : RegionName) (start N block : Nat)
    (values : Nat → ℝ) (s : BlockState) :
    (residualOutState enabled region start N block values s).regs = s.regs := by
  cases enabled <;> simp [residualOutState]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
