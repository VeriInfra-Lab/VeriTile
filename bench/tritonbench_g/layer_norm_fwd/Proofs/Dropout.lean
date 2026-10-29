import bench.tritonbench_g.layer_norm_fwd.Proofs.RandomExecution

/-! Complete first-stream dropout phase and optional boolean mask write. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def dropoutBase (seeds : RegionName) (row block : Nat) (p : ℝ)
    (values : Fin block → ℝ) (s : BlockState) : BlockState :=
  let seed := s.readMemValue .int seeds row
  let t := s.setReg "keep_mask" .bool [block] (keepTile seed p block)
  t.setReg "x" .real [block] (realTile (fun j => Formula.dropout seed p (values j) j.val))

def dropoutState (enabled storeMask : Bool) (seeds mask : RegionName)
    (row N block : Nat) (p : ℝ) (values : Fin block → ℝ) (s : BlockState) : BlockState :=
  if enabled then
    let t := dropoutBase seeds row block p values s
    if storeMask then Memory.scatter mask (row*N) N block
      (fun col => MemCell.of .bool (Formula.keep (s.readMemValue .int seeds row) p col)) t
    else t
  else s

private theorem eval_constBool (flag : Bool) (s : BlockState) :
    evalOp (.constBool flag) s = some (Tile.scalar flag) := by
  simp [evalOp]

private theorem exec_single (st : Stmt) (s : BlockState) :
    stepStmts [st] s = stepStmt st s := by
  simp only [stepStmts]
  cases stepStmt st s <;> rfl

private theorem step_if_const (flag : Bool) (body : List Stmt) (s : BlockState) :
    stepStmt (.ifThen (.constBool flag) body) s =
      if flag then stepStmts body s else some s := by
  simp only [stepStmt, eval_constBool, bind, Option.bind, Tile.scalar]

def dropoutValueOp (block : Nat) (p : ℝ) : Op .real [block] :=
  .where (.ref .bool [block] "keep_mask")
    (.div .real .scalarR (.ref .real [block] "x")
      (.sub .real .nil (.const 1.0) (.const p)))
    (.broadcast (.const 0.0) [block])

private theorem eval_zero (block : Nat) (s : BlockState) :
    evalOp (.broadcast (.const 0.0) [block]) s =
      some (realTile (fun _ => 0)) := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  rw [hz]
  simp only [evalOp, realTile, Tile.scalar, bind, Option.bind, pure]

theorem dropoutValueOp_eval (seed : Int) (block : Nat) (p : ℝ)
    (values : Fin block → ℝ) (s : BlockState)
    (hkeep : s.regs .bool [block] "keep_mask" = some (keepTile seed p block))
    (hx : s.regs .real [block] "x" = some (realTile values)) :
    evalOp (dropoutValueOp block p) s =
      some (realTile (fun j => Formula.dropout seed p (values j) j.val)) := by
  simp only [dropoutValueOp, evalOp_where, evalOp_div, evalOp_sub,
    evalOp_const, evalOp_ref, hkeep, hx, eval_zero, bind, Option.bind, pure]
  congr 1
  ext j
  simp only [Tile.select, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.div, NumericDType.sub, WithBot.realDiv, WithBot.realSub,
    Option.map₂_some_some, Tile.scalar, realTile, keepTile, Formula.dropout]
  split <;> norm_num

def dropoutMaskStmt (mask : RegionName) (N block : Nat) : Stmt :=
  .store .bool [block]
    (.region mask.cast (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "row") (.constNat N))
      (.ref .nat [block] "cols")))
    (.ref .bool [block] "keep_mask")
    (.mask (.lt .nat .scalarR (.ref .nat [block] "cols") (.constNat N)))

theorem dropoutMaskStmt_exec (mask : RegionName) (row N block : Nat)
    (seed : Int) (p : ℝ) (s : BlockState)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hkeep : s.regs .bool [block] "keep_mask" = some (keepTile seed p block)) :
    stepStmt (dropoutMaskStmt mask N block) s =
      some (Memory.scatter mask (row*N) N block
        (fun col => MemCell.of .bool (Formula.keep seed p col)) s) := by
  simp [dropoutMaskStmt, stepStmt, evalOp_ref, evalOp_lt, evalOp_add, evalOp_mul,
    evalOp_constNat, hrow, hcols, hkeep, Tile.bop, Tile.cop,
    ComparableDType.lt, NumericDType.add, NumericDType.mul, keepTile,
    Memory.scatter_fold, Memory.bool_store, Region.cast]

theorem dropoutPhase_exec (enabled storeMask : Bool) (seeds mask : RegionName)
    (row N block : Nat) (p : ℝ) (values : Fin block → ℝ) (s : BlockState)
    (hBlock : Random.ArangeSizeDomain block)
    (hrow : s.regs .nat [] "row" = some (Tile.scalar row))
    (hcols : s.regs .nat [block] "cols" = some (Tile.vec (fun j => j.val)))
    (hx : s.regs .real [block] "x" = some (realTile values)) :
    stepStmts (dropoutPhase enabled storeMask seeds mask N block p) s =
      some (dropoutState enabled storeMask seeds mask row N block p values s) := by
  change stepStmts [.ifThen (.constBool enabled)
    [.assign .bool [block] "keep_mask" (keepOp seeds p block),
     .assign .real [block] "x" (dropoutValueOp block p),
     .ifThen (.constBool storeMask) [dropoutMaskStmt mask N block]]] s = _
  rw [exec_single, step_if_const]
  cases enabled
  · rfl
  · simp only [↓reduceIte]
    let seed := s.readMemValue .int seeds row
    let kept := s.setReg "keep_mask" .bool [block] (keepTile seed p block)
    have hk := stepStmt_assign_eq_some (name := "keep_mask")
      (keepOp_eval seeds row block p s hBlock hrow hcols)
    have hxk : kept.regs .real [block] "x" = some (realTile values) := by
      simpa [kept] using hx
    have hkk : kept.regs .bool [block] "keep_mask" = some (keepTile seed p block) := by
      simp [kept]
    have hv := stepStmt_assign_eq_some (name := "x")
      (dropoutValueOp_eval seed block p values kept hkk hxk)
    rw [stepStmts.cons_some hk]
    change stepStmts [.assign .real [block] "x" (dropoutValueOp block p),
      .ifThen (.constBool storeMask) [dropoutMaskStmt mask N block]] kept = _
    rw [stepStmts.cons_some hv, exec_single, step_if_const]
    cases storeMask
    · rfl
    · simp only [↓reduceIte]
      rw [exec_single]
      change stepStmt (dropoutMaskStmt mask N block) (dropoutBase seeds row block p values s) = _
      apply dropoutMaskStmt_exec
      · simpa [dropoutBase] using hrow
      · simpa [dropoutBase] using hcols
      · simp [dropoutBase]

theorem dropoutState_x (enabled storeMask : Bool) (seeds mask : RegionName)
    (row N block : Nat) (p : ℝ) (values : Fin block → ℝ) (s : BlockState)
    (hx : s.regs .real [block] "x" = some (realTile values)) :
    (dropoutState enabled storeMask seeds mask row N block p values s).regs .real [block] "x" =
      some (realTile (fun j => if enabled then
        Formula.dropout (s.readMemValue .int seeds row) p (values j) j.val else values j)) := by
  cases enabled <;> cases storeMask
  all_goals simp [dropoutState, dropoutBase, hx]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Execution
