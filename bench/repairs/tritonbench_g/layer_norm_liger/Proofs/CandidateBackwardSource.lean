import bench.repairs.tritonbench_g.layer_norm_liger.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
open VeriTile.Triton
open LayerNormLiger.Backward
def prelude (p : Parameters) := (candidateKernel p).surfaceBody.take 22
def body (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (candidateKernel p).surfaceBody
  match code[22]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ b) => b
  | _ => []
def postlude (p : Parameters) := (candidateKernel p).surfaceBody.drop 23


theorem source_split (p : Parameters) : (candidateKernel p).surfaceBody = prelude p ++
    [.forRangeInt "_" (Op.ref .int [] "row_start") (Op.ref .int [] "row_end")
      (Op.constNat 1) (body p)] ++ postlude p := by rfl

end VeriTile.Bench.Repairs.LayerNormLiger.Backward.Candidate
