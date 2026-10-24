import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericStore

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- Consecutive pieces equal the entire original source. Neither the loop
nor the output conversion is replaced by a numerical summary. -/
theorem source_partition (fp8 variant : Bool) :
    (kernelExample fp8 variant).surfaceBody =
      ((kernelExample fp8 Bool.false).surfaceBody.take 12) ++
      ((kernelExample fp8 Bool.false).surfaceBody.drop 12 |>.take 1) ++
      ((kernelExample fp8 Bool.false).surfaceBody.drop 13 |>.take 3) ++
      ((kernelExample fp8 variant).surfaceBody.drop 16 |>.take 4) ++
      ((kernelExample fp8 variant).surfaceBody.drop 20) := by
  cases fp8 <;> cases variant <;> rfl

/-- Complete typed backend execution of both arms and both variants. This
abstract arithmetic model establishes nonvacuity, not physical IEEE rounding. -/
def numeric_run (fp8 variant : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (kernelExample fp8 variant).surfaceBody
      (initial fp8) = some t} := by
  refine ⟨(store_run fp8 variant).val, ?_⟩
  conv_lhs => rw [source_partition]
  simp only [run_append, (prefix_run fp8).property, Option.bind_some,
    (loop_run fp8).property, (math_run fp8).property,
    (address_run fp8 variant).property, (store_run fp8 variant).property]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
