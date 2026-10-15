import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericPrefix

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- The body is extracted from the actual typed source loop. -/
def loopCode (fp8 : Bool) : List ComputeStmt :=
  match ((kernelExample fp8 Bool.false).surfaceBody[12]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
