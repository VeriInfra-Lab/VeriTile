import bench.repairs.tritonbench_g.triton_matmul.Kernels

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
noncomputable section

namespace Source
def prelude (p : Parameters) (st : Storage) := (typed p st).surfaceBody.take 19
def body (p : Parameters) (st : Storage) : List ComputeStmt :=
  match (typed p st).surfaceBody[19]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) (st : Storage) := (typed p st).surfaceBody.drop 20
theorem decomposition (p : Parameters) (st : Storage) :
    (typed p st).surfaceBody=prelude p st ++
      [.forRange "kk" 0 p.numKBlocks 1 (body p st)] ++ epilogue p st := by
  cases st <;> rfl
end Source

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
