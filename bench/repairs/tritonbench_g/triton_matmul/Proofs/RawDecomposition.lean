import bench.repairs.tritonbench_g.triton_matmul.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

def prelude (p : Parameters) (st : Storage) := (Source.typed p st).surfaceBody.take 24
def body (p : Parameters) (st : Storage) : List ComputeStmt :=
  match (Source.typed p st).surfaceBody[24]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) (st : Storage) := (Source.typed p st).surfaceBody.drop 25

/-- The exact copied source retains its dynamic signed loop and both dtype
branches. The branch-local continuation contains the original final store. -/
theorem decomposition (p : Parameters) (st : Storage) :
    (Source.typed p st).surfaceBody = prelude p st ++
      [.forRangeInt "k" (.constInt 0) (.ref .int [] "blocks_k") (.constNat 1) (body p st)] ++
      epilogue p st := by
  cases st <;> rfl

#axiomsClean decomposition
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
