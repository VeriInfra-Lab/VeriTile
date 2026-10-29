import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealProgram

/-! Control-flow composition keeps the loop body and arithmetic backend
abstract while checking the actual signed source bounds. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedComposition
open VeriTile.Triton FP.Structural RealProgram
noncomputable section
set_option maxHeartbeats 1000000

theorem start_eval {α : Type} [Inhabited α] (M : Algebra α)
    (T BT : Nat) (u : State α) :
    FP.Structural.evalOp M none (TypedProgram.startOp T BT) u =
      some (fun _ => 1 - (numChunks T BT : Int)) := by
  simp only [TypedProgram.startOp, TypedProgram.countOp, evalOp_unfold, bind, Option.bind,
    Option.pure_def, numeric, bop]
  apply congrArg some
  funext i
  change 0 - ((((T + BT - 1) / BT : Nat) : Int) - 1) = 1 - (numChunks T BT : Int)
  unfold numChunks
  omega

theorem statement_eval {α : Type} [Inhabited α] (M : Algebra α)
    (T BT : Nat) (body : List ComputeStmt) (u : State α) :
    step M (.forRangeInt "__rev_i_t" (TypedProgram.startOp T BT) (.constInt 1) (.constNat 1) body) u =
      rangeInt M "__rev_i_t" (1 - (numChunks T BT : Int)) 1 1 body u := by
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
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedComposition
