import bench.tritonbench_g.isfinite_kernel.Proofs.Candidate
import bench.tritonbench_g.isfinite_kernel.Proofs.Composition
import VeriTile.Triton.Float.Loop

/-! Contextual equality for a discrete index rewrite. The algebra and all
state components remain arbitrary, and failed evaluations remain equal.
No floating arithmetic, cast, or classifier law is assumed. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton FP.Structural
noncomputable section

theorem tileId_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    FP.Structural.step M (tileIdStmt Bool.false) s =
      FP.Structural.step M (tileIdStmt Bool.true) s := by
  cases hp : s.regs .nat [] "pid" <;>
    cases hj : s.regs .nat [] "j" <;>
    cases hn : s.regs .nat [] "num_ctas" <;>
    simp [tileIdStmt, FP.Structural.step, FP.Structural.evalExpr,
      FP.Structural.evalOp_unfold, hp, hj, hn, FP.Structural.numeric,
      FP.Structural.bop, Nat.mul_comm]
  apply congrArg (s.setReg "tile_id" .nat [])
  funext i
  simp [FP.Structural.bop, Nat.mul_comm]

theorem run_cons_congr {α : Type} [Inhabited α] (M : Algebra α)
    (a b : ComputeStmt) (tail : List ComputeStmt)
    (hstep : ∀ s, FP.Structural.step M a s = FP.Structural.step M b s)
    (s : State α) :
    FP.Structural.run M (a :: tail) s = FP.Structural.run M (b :: tail) s := by
  simp only [FP.Structural.run, hstep]

theorem run_prefix_congr {α : Type} [Inhabited α] (M : Algebra α)
    (pre lhs rhs : List ComputeStmt)
    (hbody : ∀ s, FP.Structural.run M lhs s = FP.Structural.run M rhs s)
    (s : State α) :
    FP.Structural.run M (pre ++ lhs) s = FP.Structural.run M (pre ++ rhs) s := by
  rw [FP.Structural.run_append, FP.Structural.run_append]
  apply congrArg ((FP.Structural.run M pre s).bind)
  funext t
  exact hbody t

theorem range_statement_congr {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (cur stop : Nat) (lhs rhs : List ComputeStmt)
    (hbody : ∀ s, FP.Structural.run M lhs s = FP.Structural.run M rhs s)
    (s : State α) :
    FP.Structural.step M
      (.forRangeDyn idx (.constNat cur) (.constNat stop) (.constNat 1) lhs) s =
    FP.Structural.step M
      (.forRangeDyn idx (.constNat cur) (.constNat stop) (.constNat 1) rhs) s := by
  rw [range_statement, range_statement]
  apply FP.Structural.range_congr_on_reachable
  intro j t _ _ _
  exact hbody _

theorem if_statement_congr {α : Type} [Inhabited α] (M : Algebra α)
    (b : Bool) (yes lhs rhs : List ComputeStmt)
    (hno : ∀ s, FP.Structural.run M lhs s = FP.Structural.run M rhs s)
    (s : State α) :
    FP.Structural.step M (.ifThenElse (.constBool b) yes lhs) s =
      FP.Structural.step M (.ifThenElse (.constBool b) yes rhs) s := by
  rw [step_constIf, step_constIf]
  cases b with
  | «false» =>
      simp only [Bool.false_eq_true, if_false]
      exact hno s
  | «true» => rfl

theorem source_context_congr {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (lhs rhs : ComputeKernel) (one bodyA bodyB : List ComputeStmt)
    (hlhs : lhs.surfaceBody = before p ++ [.ifThenElse (.constBool p.oneTile) one
      [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
        (.constNat 1) bodyA]])
    (hrhs : rhs.surfaceBody = before p ++ [.ifThenElse (.constBool p.oneTile) one
      [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
        (.constNat 1) bodyB]])
    (hbody : ∀ s, FP.Structural.run M bodyA s = FP.Structural.run M bodyB s)
    (s : State α) : FP.Structural.exec M lhs s = FP.Structural.exec M rhs s := by
  let loopA := ComputeStmt.forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
    (.constNat 1) bodyA
  let loopB := ComputeStmt.forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
    (.constNat 1) bodyB
  have hloop : ∀ t, FP.Structural.step M loopA t = FP.Structural.step M loopB t :=
    range_statement_congr M "j" 0 p.iterations bodyA bodyB hbody
  have hno : ∀ t, FP.Structural.run M [numCtas, loopA] t =
      FP.Structural.run M [numCtas, loopB] t := by
    intro t
    exact run_prefix_congr M [numCtas] [loopA] [loopB]
      (run_cons_congr M loopA loopB [] hloop) t
  have hif : ∀ t,
      FP.Structural.step M (.ifThenElse (.constBool p.oneTile) one [numCtas, loopA]) t =
      FP.Structural.step M (.ifThenElse (.constBool p.oneTile) one [numCtas, loopB]) t :=
    if_statement_congr M p.oneTile one [numCtas, loopA] [numCtas, loopB] hno
  change FP.Structural.run M lhs.surfaceBody s = FP.Structural.run M rhs.surfaceBody s
  rw [hlhs, hrhs]
  exact run_prefix_congr M (before p) _ _
    (run_cons_congr M _ _ [] hif) s

#axiomsClean tileId_step
#axiomsClean run_cons_congr
#axiomsClean run_prefix_congr
#axiomsClean range_statement_congr
#axiomsClean if_statement_congr
#axiomsClean source_context_congr
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
