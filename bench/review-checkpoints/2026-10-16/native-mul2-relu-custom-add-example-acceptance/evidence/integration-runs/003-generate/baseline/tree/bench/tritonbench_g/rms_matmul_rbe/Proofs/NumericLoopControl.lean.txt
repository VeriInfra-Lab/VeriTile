import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericLoopCode

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem loopCode_length (fp8 : Bool) (p : Pass) : (loopCode fp8 p).length = if fp8 then 14 else 9 := by
  cases fp8 <;> cases p <;> rfl

private theorem range_one (body : List ComputeStmt) (s : State (WithBot ℝ)) :
    FP.Structural.range (engine model) "_i" 0 1 1 body s =
      run (engine model) body (s.setReg "_i" .nat [] (fun _ => 0)) := by
  simp [FP.Structural.range]

/-- Evaluate only the actual source loop's bounds and control. Its complete
body is retained verbatim and executed once with the actual bound register. -/
theorem loop_control (fp8 : Bool) (p : Pass) (s : State (WithBot ℝ)) :
    run (engine model) ((pass fp8 Bool.false p).surfaceBody.drop 12 |>.take 1) s =
      run (engine model) (loopCode fp8 p) (s.setReg "_i" .nat [] (fun _ => 0)) := by
  have hcode : ((pass fp8 Bool.false p).surfaceBody.drop 12 |>.take 1) =
      [ComputeStmt.forRangeDyn "_i" (.constNat 0)
        (.div .nat .nil (.sub .nat .nil (.add .nat .nil (.constNat 1) (.constNat 2)) (.constNat 1)) (.constNat 2))
        (.constNat 1) (loopCode fp8 p)] := by cases fp8 <;> cases p <;> rfl
  rw [hcode, run, step]
  simp [evalOp_unfold, bind, pure, Option.bind, numeric, integralFloorDiv, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, range_one, run]
  cases run (engine model) (loopCode fp8 p) (s.setReg "_i" .nat [] (fun _ => 0)) <;> rfl

theorem iteration_partition (fp8 : Bool) (p : Pass) : loopCode fp8 p =
    ((loopCode fp8 p).drop 0 |>.take 4) ++ ((loopCode fp8 p).drop 4 |>.take 4) ++
    ((loopCode fp8 p).drop 8 |>.take 4) ++ ((loopCode fp8 p).drop 12 |>.take 4) := by
  cases fp8 <;> cases p <;> rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
