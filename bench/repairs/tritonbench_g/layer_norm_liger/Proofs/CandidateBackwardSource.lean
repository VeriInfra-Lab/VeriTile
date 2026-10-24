import bench.repairs.tritonbench_g.layer_norm_liger.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
open VeriTile.Triton
open LayerNormLiger.Backward
def prelude (p : Parameters) := (candidateKernel p).surfaceBody.take 13
def body (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (candidateKernel p).surfaceBody
  match code[13]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ b) => b
  | _ => []
def postlude (p : Parameters) := (candidateKernel p).surfaceBody.drop 14


theorem source_split (p : Parameters) : (candidateKernel p).surfaceBody = prelude p ++
    [.forRangeDyn "_row" (Op.ref .nat [] "row_start") (Op.ref .nat [] "row_end")
      (Op.constNat 1) (body p)] ++ postlude p := by rfl

end VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
