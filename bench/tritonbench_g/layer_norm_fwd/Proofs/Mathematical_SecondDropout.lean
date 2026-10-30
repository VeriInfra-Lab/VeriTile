import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_InputSuffix

/-! Actual second-stream dropout, with its distinct seed and mask indices. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem exec_single (st : Stmt) (s : BlockState) :
    stepStmts [st] s = stepStmt st s := by
  simp only [stepStmts]
  cases stepStmt st s <;> rfl

theorem step_if_const (flag : Bool) (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (.constBool flag) body) s =
      if flag then stepStmts body s else some s := by
  have hc : evalOp (.constBool flag) s = some (Tile.scalar flag) := by simp [evalOp]
  simp only [stepStmt, hc, bind, Option.bind, Tile.scalar]

private theorem eval_zero (block : Nat) (s : BlockState) :
    evalOp (.broadcast (.const 0.0) [block]) s =
      some (Execution.realTile (fun _ => 0)) := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  rw [hz]
  simp only [evalOp, Execution.realTile, Tile.scalar, bind, Option.bind, pure]

theorem dropoutValueOp_eval (seed : Int) (block : Nat) (p : ℝ)
    (values : Fin block → ℝ) (s : BlockState)
    (hkeep : s.regs .bool [block] "keep_mask" = some (Execution.keepTile seed p block))
    (hx : s.regs .real [block] "x1" = some (Execution.realTile values)) :
    evalOp (dropoutValueOp block p) s =
      some (Execution.realTile (fun j => Formula.dropout seed p (values j) j.val)) := by
  simp only [dropoutValueOp, evalOp_where, evalOp_div, evalOp_sub,
    evalOp_const, evalOp_ref, hkeep, hx, eval_zero, bind, Option.bind, pure]
  congr 1
  ext j
  simp only [Tile.select, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.div, NumericDType.sub, WithBot.realDiv, WithBot.realSub,
    Option.map₂_some_some, Tile.scalar, Execution.realTile, Execution.keepTile, Formula.dropout]
  split <;> norm_num

theorem maskStmt_exec (mask : RegionName) (M row N block : Nat)
    (seed : Int) (p : ℝ) (s : BlockState)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hkeep : s.regs .bool [block] "keep_mask" = some (Execution.keepTile seed p block)) :
    stepStmt (maskStmt mask M N block) s =
      some (Memory.scatter mask ((M+row)*N) N block
        (fun col => MemCell.of .bool (Formula.keep seed p col)) s) := by
  simp [maskStmt, secondRowOp, stepStmt, evalOp_ref, evalOp_lt, evalOp_add, evalOp_mul,
    evalOp_constNat, hrow, hcols, hkeep, Tile.bop, Tile.cop,
    ComparableDType.lt, NumericDType.add, NumericDType.mul, Execution.keepTile,
    Memory.scatter_fold, Memory.bool_store, Region.cast]

def dropoutBase (seeds : RegionName) (M row block : Nat) (p : ℝ)
    (values : Fin block → ℝ) (s : BlockState) : BlockState :=
  let seed := s.readMemValue .int seeds (M+row)
  let t := s.setReg "keep_mask" .bool [block] (Execution.keepTile seed p block)
  t.setReg "x1" .real [block]
    (Execution.realTile (fun j => Formula.dropout seed p (values j) j.val))

def dropoutState (enabled storeMask : Bool) (seeds mask : RegionName)
    (M row N block : Nat) (p : ℝ) (values : Fin block → ℝ) (s : BlockState) : BlockState :=
  if enabled then
    let t := dropoutBase seeds M row block p values s
    if storeMask then Memory.scatter mask ((M+row)*N) N block
      (fun col => MemCell.of .bool (Formula.keep (s.readMemValue .int seeds (M+row)) p col)) t
    else t
  else s

theorem dropoutPhase_exec (enabled storeMask : Bool) (seeds mask : RegionName)
    (M row N block : Nat) (p : ℝ) (values : Fin block → ℝ) (s : BlockState)
    (hBlock : Random.ArangeSizeDomain block)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x1" = some (Execution.realTile values)) :
    stepStmts (dropoutPhase enabled storeMask seeds mask M N block p) s =
      some (dropoutState enabled storeMask seeds mask M row N block p values s) := by
  unfold dropoutPhase
  rw [exec_single, step_if_const]
  cases enabled
  · rfl
  · simp only [↓reduceIte]
    let seed := s.readMemValue .int seeds (M+row)
    let kept := s.setReg "keep_mask" .bool [block] (Execution.keepTile seed p block)
    have hk := stepStmt_assign_eq_some (name := "keep_mask")
      (secondKeepOp_eval seeds M row block p s hBlock hrow hcols)
    have hxk : kept.regs .real [block] "x1" = some (Execution.realTile values) := by
      simpa [kept] using hx
    have hkk : kept.regs .bool [block] "keep_mask" = some (Execution.keepTile seed p block) := by
      simp [kept]
    have hv := stepStmt_assign_eq_some (name := "x1")
      (dropoutValueOp_eval seed block p values kept hkk hxk)
    rw [stepStmts.cons_some hk]
    change stepStmts [.assign .real [block] "x1" (dropoutValueOp block p),
      .ifThen (.constBool storeMask) [maskStmt mask M N block]] kept = _
    rw [stepStmts.cons_some hv, exec_single, step_if_const]
    cases storeMask
    · rfl
    · simp only [↓reduceIte]
      rw [exec_single]
      change stepStmt (maskStmt mask M N block) (dropoutBase seeds M row block p values s) = _
      apply maskStmt_exec
      · simpa [dropoutBase] using hrow
      · simpa [dropoutBase] using hcols
      · simp [dropoutBase]

theorem dropoutState_x1 (enabled storeMask : Bool) (seeds mask : RegionName)
    (M row N block : Nat) (p : ℝ) (values : Fin block → ℝ) (s : BlockState)
    (hx : s.regs .real [block] "x1" = some (Execution.realTile values)) :
    (dropoutState enabled storeMask seeds mask M row N block p values s).regs .real [block] "x1" =
      some (Execution.realTile (fun j => if enabled then
        Formula.dropout (s.readMemValue .int seeds (M+row)) p (values j) j.val else values j)) := by
  cases enabled <;> cases storeMask
  all_goals simp [dropoutState, dropoutBase, hx]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.InputSuffix
