import bench.tritonbench_g.token_attn_reduceV.Kernels

/-! Proof-only decomposition of the actual source in Kernels.lean. -/
namespace VeriTile.Bench.Repairs.TokenAttnReduceV
open VeriTile.Triton

def prelude (p : Parameters) := (kernel p).surfaceBody.take 13
def body (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel p).surfaceBody
  match code[13]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ xs) => xs
  | some (ComputeStmt.forRangeInt _ _ _ _ xs) => xs
  | _ => []
def postlude (p : Parameters) := (kernel p).surfaceBody.drop 14

end VeriTile.Bench.Repairs.TokenAttnReduceV
