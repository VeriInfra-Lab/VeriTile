import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_SecondDropout

/-! Complete second-input body: load, optional rowscale, dropout/mask, and
addition into x. Values here use the body-entry memory; the full prefix must
establish that earlier writes preserve the relevant initial inputs. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

def scaleState (enabled : Bool) (rowscale : RegionName) (M row block : Nat)
    (values : Fin block → ℝ) (s : BlockState) : BlockState :=
  if enabled then
    (s.setReg "rowscale" .real [] (Tile.scalar (some (s.readMem rowscale (M+row))))).setReg
      "x1" .real [block] (Execution.realTile (fun j => values j * s.readMem rowscale (M+row)))
  else s

theorem scalePhase_exec (enabled : Bool) (rowscale : RegionName) (M row block : Nat)
    (values : Fin block → ℝ) (s : BlockState)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hx : s.regs .real [block] "x1" = some (Execution.realTile values)) :
    stepStmts (scalePhase enabled rowscale M block) s =
      some (scaleState enabled rowscale M row block values s) := by
  unfold scalePhase
  rw [exec_single, step_if_const]
  cases enabled
  · rfl
  · simp [scaleState, secondRowOp, stepStmts, stepStmt, evalOp,
      hrow, hx, Execution.realTile, Tile.bop, NumericDType.mul, NumericDType.add]

def loadedState (r : Regions) (start N block : Nat) (s : BlockState) : BlockState :=
  s.setReg "x1" .real [block] (Execution.realTile (rowValues r.x1 start N block s))

def scaledValues (f : Flags) (r : Regions) (M row start N block : Nat)
    (s : BlockState) : Fin block → ℝ :=
  fun j => if f.rowscale then rowValues r.x1 start N block s j * s.readMem r.rowscale (M+row)
    else rowValues r.x1 start N block s j

def scaledState (f : Flags) (r : Regions) (M row start N block : Nat) (s : BlockState) : BlockState :=
  scaleState f.rowscale r.rowscale M row block (rowValues r.x1 start N block s)
    (loadedState r start N block s)

theorem scaledState_x1 (f : Flags) (r : Regions) (M row start N block : Nat) (s : BlockState) :
    (scaledState f r M row start N block s).regs .real [block] "x1" =
      some (Execution.realTile (scaledValues f r M row start N block s)) := by
  cases hs : f.rowscale
  all_goals simp [scaledState, scaleState, loadedState, scaledValues, hs]
  all_goals
    ext j
    simp [Execution.realTile, scaledValues, hs]

theorem scaledState_mem (f : Flags) (r : Regions) (M row start N block : Nat) (s : BlockState) :
    (scaledState f r M row start N block s).mem = s.mem := by
  cases hs : f.rowscale <;> simp [scaledState, scaleState, loadedState, hs] <;> rfl

theorem scaledState_seed (f : Flags) (r : Regions) (M row start N block : Nat) (s : BlockState) :
    (scaledState f r M row start N block s).readMemValue .int r.seeds (M+row) =
      s.readMemValue .int r.seeds (M+row) := by
  cases hs : f.rowscale <;> simp [scaledState, scaleState, loadedState, hs]

def streamValues (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (s : BlockState) : Fin block → ℝ :=
  fun j => if f.dropout then Formula.dropout (s.readMemValue .int r.seeds (M+row)) p
    (scaledValues f r M row start N block s j) j.val
    else scaledValues f r M row start N block s j

def droppedState (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (s : BlockState) : BlockState :=
  dropoutState f.dropout f.dropoutMask r.seeds r.dropoutMask M row N block p
    (scaledValues f r M row start N block s) (scaledState f r M row start N block s)

theorem droppedState_x1 (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (s : BlockState) :
    (droppedState f r M row start N block p s).regs .real [block] "x1" =
      some (Execution.realTile (streamValues f r M row start N block p s)) := by
  rw [droppedState, dropoutState_x1 _ _ _ _ _ _ _ _ _ _ _
    (scaledState_x1 f r M row start N block s), scaledState_seed]
  rfl

theorem droppedState_regs (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (s : BlockState) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hx1 : name ≠ "x1") (hrs : name ≠ "rowscale") (hkeep : name ≠ "keep_mask") :
    (droppedState f r M row start N block p s).regs dtype shape name = s.regs dtype shape name := by
  cases hs : f.rowscale <;> cases hd : f.dropout <;> cases hm : f.dropoutMask
  all_goals simp [droppedState, dropoutState, dropoutBase, scaledState,
    scaleState, loadedState, hs, hd, hm, hx1, hrs, hkeep]

theorem addStmt_exec (name : RegName) (block : Nat) (x values : Fin block → ℝ)
    (s : BlockState)
    (hx : s.regs .real [block] "x" = some (Execution.realTile x))
    (hv : s.regs .real [block] name = some (Execution.realTile values)) :
    stepStmt (addStmt name block) s =
      some (s.setReg "x" .real [block] (Execution.realTile (fun j => x j + values j))) := by
  simp [addStmt, stepStmt, evalOp_add, evalOp_ref, hx, hv,
    Execution.realTile, Tile.bop, NumericDType.add]

def bodyState (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (x : Fin block → ℝ) (s : BlockState) : BlockState :=
  (droppedState f r M row start N block p s).setReg "x" .real [block]
    (Execution.realTile (fun j => x j + streamValues f r M row start N block p s j))

theorem secondBody_exec (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (x : Fin block → ℝ) (s : BlockState)
    (hBlock : Random.ArangeSizeDomain block)
    (hptr : s.regs .ptr [] "X1" = some (Tile.scalar (r.x1, start)))
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x" = some (Execution.realTile x)) :
    stepStmts (secondBody f r M N block p) s =
      some (bodyState f r M row start N block p x s) := by
  have hl : stepStmts [.assign .real [block] "x1" (rowLoadOp "X1" N block)] s =
      some (loadedState r start N block s) := by
    rw [exec_single]
    exact stepStmt_assign_eq_some (rowLoadOp_eval "X1" r.x1 start N block s hptr hcols)
  have hs := scalePhase_exec f.rowscale r.rowscale M row block
    (rowValues r.x1 start N block s) (loadedState r start N block s)
    (by simpa [loadedState] using hrow) (by simp [loadedState])
  have hd := dropoutPhase_exec f.dropout f.dropoutMask r.seeds r.dropoutMask M row N block p
    (scaledValues f r M row start N block s) (scaledState f r M row start N block s) hBlock
    (by cases hscale : f.rowscale <;>
      simpa [scaledState, scaleState, loadedState, hscale] using hrow)
    (by cases hscale : f.rowscale <;>
      simpa [scaledState, scaleState, loadedState, hscale] using hcols)
    (scaledState_x1 f r M row start N block s)
  have ha := addStmt_exec "x1" block x (streamValues f r M row start N block p s)
    (droppedState f r M row start N block p s)
    (by rw [droppedState_regs f r M row start N block p s .real [block] "x"
      (by decide) (by decide) (by decide)]; exact hx)
    (droppedState_x1 f r M row start N block p s)
  simp only [secondBody, ProgramRewrite.append, hl, Option.bind_some]
  rw [hs, Option.bind_some]
  change (stepStmts (dropoutPhase f.dropout f.dropoutMask r.seeds r.dropoutMask M N block p)
    (scaledState f r M row start N block s)).bind
      (stepStmts [addStmt "x1" block]) = _
  rw [hd, Option.bind_some, exec_single]
  exact ha

theorem bodyState_x (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (x : Fin block → ℝ) (s : BlockState) :
    (bodyState f r M row start N block p x s).regs .real [block] "x" =
      some (Execution.realTile (fun j => x j + streamValues f r M row start N block p s j)) := by
  simp [bodyState]

theorem bodyState_mem (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (x : Fin block → ℝ) (s : BlockState) (hN : N ≤ block)
    (region : RegionName) (offset : Nat) :
    (bodyState f r M row start N block p x s).mem region offset =
      if f.dropout && f.dropoutMask &&
          decide (region = r.dropoutMask ∧ Contract.inRow ((M+row)*N) N offset) then
        MemCell.of .bool (Formula.keep (s.readMemValue .int r.seeds (M+row)) p (offset-(M+row)*N))
      else s.mem region offset := by
  cases hd : f.dropout <;> cases hm : f.dropoutMask
  all_goals simp [bodyState, droppedState, dropoutState, dropoutBase, hd, hm,
    Memory.scatter_cell _ _ _ _ hN, scaledState_seed, scaledState_mem, Bool.and_eq_true]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
