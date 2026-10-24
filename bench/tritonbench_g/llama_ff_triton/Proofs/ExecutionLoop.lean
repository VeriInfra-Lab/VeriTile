import bench.tritonbench_g.llama_ff_triton.Proofs.ExecutionIteration4
import bench.tritonbench_g.llama_ff_triton.Proofs.ExecutionLoopControl

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000

/-- Successful execution of the original whole cdiv loop. Every assignment
and conversion is connected in source order to its actual predecessor state. -/
def loop_run (fp8 : Bool) :
    {t : State (WithBot ℝ) // run (engine model) ((kernelExample fp8 Bool.false).surfaceBody.drop 13 |>.take 1)
      (prefix_run fp8).val = some t} := by
  refine ⟨(iteration4_run fp8).val, ?_⟩
  rw [loop_control]
  conv_lhs => rw [iteration_partition]
  simp only [run_append, (iteration0_run fp8).property, Option.bind_some,
    (iteration1_run fp8).property, (iteration2_run fp8).property,
    (iteration3_run fp8).property, (iteration4_run fp8).property]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
