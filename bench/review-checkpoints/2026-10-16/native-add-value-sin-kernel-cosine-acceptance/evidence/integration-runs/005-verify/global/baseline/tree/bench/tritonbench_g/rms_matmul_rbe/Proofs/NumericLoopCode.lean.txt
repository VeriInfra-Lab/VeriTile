import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericPrefix

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
/-- Extract the actual source loop body, preserving every assignment. -/
def loopCode (fp8 : Bool) (p : Pass) : List ComputeStmt :=
  match ((pass fp8 Bool.false p).surfaceBody[12]? : Option ComputeStmt) with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
