import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessHeader

/-! Control composition keeps the algebra, body and states abstract. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural
noncomputable section

theorem grid_step_false {α : Type} [Inhabited α] (M : Algebra α)
    (yes no : List ComputeStmt) (s : State α) :
    FP.Structural.step M (.ifThenElse (.constBool Bool.false) yes no) s =
      FP.Structural.run M no s := by
  conv_lhs =>
    rw [FP.Structural.step]
    simp only [FP.Structural.evalOp_unfold, bind, Option.bind, Option.pure_def,
      Bool.false_eq_true, if_false]

theorem grid_run_false {α : Type} [Inhabited α] (M : Algebra α)
    (before yes no : List ComputeStmt) (s : State α) :
    FP.Structural.run M (before ++ [.ifThenElse (.constBool Bool.false) yes no]) s =
      FP.Structural.run M (before ++ no) s := by
  rw [FP.Structural.run_append, FP.Structural.run_append]
  apply congrArg ((FP.Structural.run M before s).bind)
  funext u
  simp [FP.Structural.run, grid_step_false]

theorem grid_range_statement {α : Type} [Inhabited α] (M : Algebra α)
    (body : List ComputeStmt) (s : State α) :
    FP.Structural.step M
      (.forRangeDyn "j" (.constNat 0) (.constNat 2) (.constNat 1) body) s =
      FP.Structural.range M "j" 0 2 1 body s := by
  conv_lhs =>
    rw [FP.Structural.step]
    simp only [FP.Structural.evalOp_unfold, bind, Option.bind, Option.pure_def]

theorem grid_range_two {α : Type} [Inhabited α] (M : Algebra α)
    (body : List ComputeStmt) (s v w : State α)
    (hfirst : FP.Structural.run M body (s.setReg "j" .nat [] (fun _ => 0)) = some v)
    (hsecond : FP.Structural.run M body (v.setReg "j" .nat [] (fun _ => 1)) = some w) :
    FP.Structural.range M "j" 0 2 1 body s = some w := by
  rw [FP.Structural.range]
  simp only [Nat.one_ne_zero, ↓reduceIte, hfirst, bind, Option.bind_some]
  rw [FP.Structural.range]
  simp only [Nat.one_ne_zero, ↓reduceIte, hsecond, bind, Option.bind_some]
  rw [FP.Structural.range]
  simp

theorem grid_exec_of_steps {α : Type} [Inhabited α] (M : Algebra α)
    (k : ComputeKernel) (before yes : List ComputeStmt) (init : ComputeStmt)
    (body : List ComputeStmt) (s entry first last : State α)
    (hsplit : k.surfaceBody = before ++
      [.ifThenElse (.constBool Bool.false) yes
        [init, .forRangeDyn "j" (.constNat 0) (.constNat 2) (.constNat 1) body]])
    (hheader : FP.Structural.run M (before ++ [init]) s = some entry)
    (hfirst : FP.Structural.run M body (entry.setReg "j" .nat [] (fun _ => 0)) = some first)
    (hsecond : FP.Structural.run M body (first.setReg "j" .nat [] (fun _ => 1)) = some last) :
    FP.Structural.exec M k s = some last := by
  change FP.Structural.run M k.surfaceBody s = some last
  rw [hsplit, grid_run_false]
  change FP.Structural.run M (before ++ ([init] ++
    [.forRangeDyn "j" (.constNat 0) (.constNat 2) (.constNat 1) body])) s = some last
  rw [← List.append_assoc, FP.Structural.run_append, hheader]
  simp only [Option.bind_some, FP.Structural.run, grid_range_statement,
    grid_range_two M body entry first last hfirst hsecond, bind, Option.bind_some]

#axiomsClean grid_step_false
#axiomsClean grid_run_false
#axiomsClean grid_range_statement
#axiomsClean grid_range_two
#axiomsClean grid_exec_of_steps
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
