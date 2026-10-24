import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericIteration3
import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericLoopControl

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- Successful execution of the original whole cdiv loop. Every assignment
and conversion is connected in source order to its actual predecessor state. -/
def loop_run (fp8 : Bool) :
    {t : State (WithBot ℝ) // run (engine model) ((kernelExample fp8 Bool.false).surfaceBody.drop 12 |>.take 1)
      (prefix_run fp8).val = some t} := by
  refine ⟨(iteration3_run fp8).val, ?_⟩
  rw [loop_control]
  conv_lhs => rw [iteration_partition]
  simp only [run_append, (iteration0_run fp8).property, Option.bind_some,
    (iteration1_run fp8).property, (iteration2_run fp8).property,
    (iteration3_run fp8).property]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
