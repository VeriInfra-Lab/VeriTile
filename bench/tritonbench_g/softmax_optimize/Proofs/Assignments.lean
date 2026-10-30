import bench.tritonbench_g.softmax_optimize.Proofs.Source

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Assignments
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize

inductive Only (names : List RegName) : List Stmt → Prop
  | nil : Only names []
  | cons {dtype : TileDType} {shape : TileShape} {name : RegName} {rest : List Stmt}
      (value : Op dtype shape) (allowed : name ∈ names) (tail : Only names rest) :
      Only names (.assign dtype shape name value :: rest)

theorem preserve {names : List RegName} {statements : List Stmt} (only : Only names statements)
    (P : BlockState → Prop)
    (closed : ∀ (state : BlockState) (name : RegName) (dtype : TileDType) (shape : TileShape)
      (value : Tile dtype shape), name ∈ names → P state → P (state.setReg name dtype shape value))
    {before after : BlockState} (initial : P before)
    (run : stepStmts statements before = some after) : P after := by
  induction only generalizing before after with
  | nil =>
    simp only [stepStmts.nil, Option.some.injEq] at run
    exact run ▸ initial
  | @cons dtype shape name rest value allowed tail ih =>
    cases he : evalOp value before with
    | none =>
      simp only [stepStmts, stepStmt, he, Option.bind_eq_bind, Option.bind_none] at run
      contradiction
    | some tile =>
      rw [stepStmts.cons_some (stepStmt_assign_eq_some he)] at run
      exact ih (closed before name dtype shape tile allowed initial) run

theorem reduction_only (B : Nat) : Only ["final_m", "z", "m"] (reductionStmts B) := by
  unfold reductionStmts
  repeat' apply Only.cons _ (by simp)
  exact Only.nil

theorem reduction_register (B : Nat) {before after : BlockState}
    (run : stepStmts (reductionStmts B) before = some after)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (outside : name ∉ ["final_m", "z", "m"]) :
    after.regs dtype shape name = before.regs dtype shape name := by
  apply preserve (reduction_only B) (fun state => state.regs dtype shape name = before.regs dtype shape name)
    (before := before) (after := after) _ rfl run
  intro state written dt sh value allowed hp
  have hn : name ≠ written := by intro hn; subst written; exact outside allowed
  simpa only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hn] using hp

theorem reduction_numPids (B : Nat) {before after : BlockState}
    (run : stepStmts (reductionStmts B) before = some after) : after.numPids = before.numPids := by
  exact preserve (reduction_only B) (fun state => state.numPids = before.numPids)
    (fun _ _ _ _ _ _ hp => hp) rfl run

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SoftmaxOptimize.Assignments
