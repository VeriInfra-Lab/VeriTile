import bench.tritonbench_g.triton_matmul.Sources.Kernels
import bench.tritonbench_g.triton_matmul.Sources.Parameters
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1200000

def prelude (p : Parameters) (st : Storage) := (typed p st).surfaceBody.take 33

def iteration (p : Parameters) (st : Storage) : List ComputeStmt :=
  match (typed p st).surfaceBody[33]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ body)=>body
  | _=>[]

def epilogue (p : Parameters) (st : Storage) := (typed p st).surfaceBody.drop 34

theorem decomposition (p : Parameters) (st : Storage) :
    (typed p st).surfaceBody=prelude p st++
      [.forRangeInt "k" (.constInt 0) (.ref .int [] "blocks_k") (.constNat 1)
        (iteration p st)]++epilogue p st := by
  cases st <;> rfl

#axiomsClean decomposition
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
