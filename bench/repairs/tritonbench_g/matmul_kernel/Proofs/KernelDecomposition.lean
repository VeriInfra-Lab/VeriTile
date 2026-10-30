import bench.repairs.tritonbench_g.matmul_kernel.Proofs.SourcePreservation

namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource
open VeriTile.Triton

def prelude (cfg : Config) := (source cfg).surfaceBody.take 8
def body (cfg : Config) : List ComputeStmt :=
  match (source cfg).surfaceBody[8]? with
  | some (ComputeStmt.forRange _ _ _ _ code)=>code
  | _=>[]
def epilogue (cfg : Config) := (source cfg).surfaceBody.drop 9

theorem source_decomposition (cfg : Config) :
    (source cfg).surfaceBody=prelude cfg ++ [.forRange "k" 0 cfg.steps 1 (body cfg)] ++ epilogue cfg := rfl

#axiomsClean source_decomposition
end VeriTile.Bench.Repairs.MatmulKernel.KernelSource
