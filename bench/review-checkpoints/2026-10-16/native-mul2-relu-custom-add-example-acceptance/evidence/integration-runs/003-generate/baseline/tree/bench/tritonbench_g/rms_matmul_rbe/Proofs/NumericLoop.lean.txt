import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericIteration3
import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericLoopControl

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
def loop_run (fp8 : Bool) (p : Pass) (s : State (WithBot ℝ))
    :
    {t : State (WithBot ℝ) // run (engine model) ((pass fp8 Bool.false p).surfaceBody.drop 12 |>.take 1)
      (prefix_run fp8 p s).val = some t} := by
  refine ⟨(iteration3_run fp8 p s).val, ?_⟩
  conv_lhs =>
    rw [loop_control]
    arg 2
    rw [iteration_partition]
  simp only [run_append, (iteration0_run fp8 p s).property, Option.bind_some,
    (iteration1_run fp8 p s).property, (iteration2_run fp8 p s).property,
    (iteration3_run fp8 p s).property]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
