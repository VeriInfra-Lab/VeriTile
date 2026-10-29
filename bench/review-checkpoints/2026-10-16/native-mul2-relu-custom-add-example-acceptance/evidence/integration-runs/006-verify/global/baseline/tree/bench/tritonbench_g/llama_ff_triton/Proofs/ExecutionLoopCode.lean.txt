import bench.tritonbench_g.llama_ff_triton.Proofs.ExecutionPrefix

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false
/-- Extract only the actual typed loop body; its surrounding loop is
reconnected after checking all consecutive assignments. -/
def loopCode (fp8 : Bool) : List ComputeStmt :=
  match ((kernelExample fp8 Bool.false).surfaceBody[13]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []


end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
