import bench.repairs.tritonbench_g.layer_norm_liger.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.LayerNormLiger.Backward
open VeriTile.Triton
def prelude (p : Parameters) := (kernel p).surfaceBody.take 21
def body (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel p).surfaceBody
  match code[21]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ b) => b
  | _ => []
def postlude (p : Parameters) := (kernel p).surfaceBody.drop 22


theorem source_split (p : Parameters) : (kernel p).surfaceBody = prelude p ++
    [.forRangeInt "_" (Op.ref .int [] "row_start") (Op.ref .int [] "row_end")
      (Op.constNat 1) (body p)] ++ postlude p := by rfl

end VeriTile.Bench.Repairs.LayerNormLiger.Backward
