import bench.tritonbench_g.matmul_persistent_triton.Proofs.Numeric3Store

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem loop_partition (storage : TypedSource.Storage) :
    loopCode storage = ((loopCode storage).drop 0 |>.take 2) ++
      ((loopCode storage).drop 2 |>.take 5) ++
      ((loopCode storage).drop 7 |>.take 1) ++
      ((loopCode storage).drop 8 |>.take 1) := by
  cases storage <;> rfl

theorem iteration0_run (storage : TypedSource.Storage) :
    run (engine model) (loopCode storage)
      ((prefix_run storage).val.setReg "_i" .nat [] (fun _ => 0)) =
        some (iteration0_store_run storage).val := by
  conv_lhs => arg 2; rw [loop_partition]
  simp only [run_append, (iteration0_schedule_run storage).property, Option.bind_some,
    (iteration0_loads_run storage).property, (iteration0_dot_run storage).property,
    (iteration0_store_run storage).property]

theorem iteration1_run (storage : TypedSource.Storage) :
    run (engine model) (loopCode storage)
      ((iteration0_store_run storage).val.setReg "_i" .nat [] (fun _ => 1)) =
        some (iteration1_store_run storage).val := by
  conv_lhs => arg 2; rw [loop_partition]
  simp only [run_append, (iteration1_schedule_run storage).property, Option.bind_some,
    (iteration1_loads_run storage).property, (iteration1_dot_run storage).property,
    (iteration1_store_run storage).property]

theorem iteration2_run (storage : TypedSource.Storage) :
    run (engine model) (loopCode storage)
      ((iteration1_store_run storage).val.setReg "_i" .nat [] (fun _ => 2)) =
        some (iteration2_store_run storage).val := by
  conv_lhs => arg 2; rw [loop_partition]
  simp only [run_append, (iteration2_schedule_run storage).property, Option.bind_some,
    (iteration2_loads_run storage).property, (iteration2_dot_run storage).property,
    (iteration2_store_run storage).property]

theorem iteration3_run (storage : TypedSource.Storage) :
    run (engine model) (loopCode storage)
      ((iteration2_store_run storage).val.setReg "_i" .nat [] (fun _ => 3)) =
        some (iteration3_store_run storage).val := by
  conv_lhs => arg 2; rw [loop_partition]
  simp only [run_append, (iteration3_schedule_run storage).property, Option.bind_some,
    (iteration3_loads_run storage).property, (iteration3_dot_run storage).property,
    (iteration3_store_run storage).property]

theorem loop_control (storage : TypedSource.Storage) :
    run (engine model) ((sample.source storage Bool.false).surfaceBody.drop 16)
      (prefix_run storage).val =
    FP.Structural.range (engine model) "_i" 0 4 1 (loopCode storage) (prefix_run storage).val := by
  cases storage
  all_goals simp (maxSteps := 300000) [sample, Parameters.source, TypedSource.kernel,
    ComputeKernel.surfaceBody, loopCode, FPProof.loop, SourceChecks.loopBody,
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, bind, Option.bind_some, Option.bind_none, Option.bind_fun_some,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, numeric,
    FP.Structural.bop, natLt, integralFloorDiv, integralMod, prefix_run, initial]

private theorem range_next (body : List ComputeStmt) (s : State (WithBot ℝ))
    (i stop : Nat) (hi : i < stop) :
    FP.Structural.range (engine model) "_i" i stop 1 body s =
      (run (engine model) body (s.setReg "_i" .nat [] (fun _ => i)) >>= fun t =>
        FP.Structural.range (engine model) "_i" (i + 1) stop 1 body t) := by
  rw [FP.Structural.range]
  simp only [Nat.one_ne_zero, if_false, hi, if_true]

private theorem range_done (body : List ComputeStmt) (s : State (WithBot ℝ)) (i : Nat) :
    FP.Structural.range (engine model) "_i" i i 1 body s = some s := by
  rw [FP.Structural.range]
  simp only [Nat.one_ne_zero, if_false, Nat.lt_irrefl]

private theorem range_four (body : List ComputeStmt) (s : State (WithBot ℝ)) :
    FP.Structural.range (engine model) "_i" 0 4 1 body s =
    (run (engine model) body (s.setReg "_i" .nat [] (fun _ => 0)) >>= fun s1 =>
      run (engine model) body (s1.setReg "_i" .nat [] (fun _ => 1)) >>= fun s2 =>
      run (engine model) body (s2.setReg "_i" .nat [] (fun _ => 2)) >>= fun s3 =>
      run (engine model) body (s3.setReg "_i" .nat [] (fun _ => 3))) := by
  simp only [range_next, range_done, Nat.reduceLT, Nat.reduceAdd, bind, Option.bind_fun_some]

/-- Four dynamic persistent iterations execute the complete original source.
Every staged state is connected to the preceding actual source fragment. -/
def numeric_run (storage : TypedSource.Storage) :
    {t : State (WithBot ℝ) // run (engine model)
      (sample.source storage Bool.false).surfaceBody (initial storage) = some t} := by
  refine ⟨(iteration3_store_run storage).val, ?_⟩
  conv_lhs => arg 2; rw [← List.take_append_drop 16 (sample.source storage Bool.false).surfaceBody]
  simp only [run_append, (prefix_run storage).property, Option.bind_some,
    loop_control, range_four, bind, iteration0_run, iteration1_run, iteration2_run, iteration3_run]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
