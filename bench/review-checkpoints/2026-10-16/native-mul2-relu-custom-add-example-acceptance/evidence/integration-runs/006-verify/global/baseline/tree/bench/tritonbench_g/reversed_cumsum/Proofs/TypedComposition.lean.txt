import bench.tritonbench_g.reversed_cumsum.Proofs.TypedProgram

/-! Small control-flow composition lemmas, with the backend and loop body
kept abstract while the actual signed bounds are evaluated. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedComposition
open VeriTile.Triton FP.Structural RealProgram
noncomputable section
set_option maxHeartbeats 1000000

theorem start_eval {α : Type} [Inhabited α] (M : Algebra α)
    (p : Layout) (u : State α) :
    FP.Structural.evalOp M none (TypedProgram.startOp p) u =
      some (fun _ => 1 - (numChunks p : Int)) := by
  simp only [TypedProgram.startOp, stopOp, evalOp_unfold, bind, Option.bind,
    Option.pure_def, numeric, bop]
  apply congrArg some
  funext i
  change 0 - ((((p.T + p.BT - 1) / p.BT : Nat) : Int) - 1) = 1 - (numChunks p : Int)
  unfold numChunks
  omega

theorem statement_eval {α : Type} [Inhabited α] (M : Algebra α)
    (p : Layout) (body : List ComputeStmt) (u : State α) :
    step M (.forRangeInt "__rev_i_t" (TypedProgram.startOp p) (.constInt 1) (.constNat 1) body) u =
      rangeInt M "__rev_i_t" (1 - (numChunks p : Int)) 1 1 body u := by
  conv_lhs =>
    rw [step, start_eval]
    simp only [evalOp_unfold, bind, Option.bind, Option.pure_def]

theorem exec_of_split {α : Type} [Inhabited α] (M : Algebra α)
    (kernel : ComputeKernel) (before : List ComputeStmt) (last : ComputeStmt)
    (hsplit : kernel.surfaceBody = before ++ [last]) (u v w : State α)
    (hbefore : run M before u = some v) (hlast : step M last v = some w) :
    FP.Structural.exec M kernel u = some w := by
  change run M kernel.surfaceBody u = some w
  rw [hsplit, run_append, hbefore]
  simp only [Option.bind_some, run, hlast, bind, Option.bind]

#axiomsClean start_eval
#axiomsClean statement_eval
#axiomsClean exec_of_split
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedComposition
