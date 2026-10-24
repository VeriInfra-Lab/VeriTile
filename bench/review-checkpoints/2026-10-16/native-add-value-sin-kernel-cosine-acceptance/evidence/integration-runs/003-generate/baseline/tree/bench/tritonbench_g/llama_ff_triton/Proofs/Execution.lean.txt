import bench.tritonbench_g.llama_ff_triton.Proofs.ExecutionStore

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
/-- These are consecutive pieces of the original full source; no replacement
program or precomputed output is introduced by the execution proof. -/
theorem source_partition (fp8 variant : Bool) :
    (kernelExample fp8 variant).surfaceBody =
      ((kernelExample fp8 Bool.false).surfaceBody.take 13) ++
      ((kernelExample fp8 Bool.false).surfaceBody.drop 13 |>.take 1) ++
      ((kernelExample fp8 Bool.false).surfaceBody.drop 14 |>.take 5) ++
      ((kernelExample fp8 variant).surfaceBody.drop 19 |>.take 4) ++
      ((kernelExample fp8 variant).surfaceBody.drop 23) := by
  cases fp8 <;> cases variant <;> rfl

/-- Successful execution of the complete original typed source, for both
storage profiles and both variants, composed from consecutive actual stages. -/
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
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
