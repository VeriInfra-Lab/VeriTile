import bench.repairs.tritonbench_g.context_attn_bloom.Parameters
import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.RealStream
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.KernelProfile

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Raw
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
noncomputable section


def call (g : Geometry) (r : Regions) (u : Inputs) : FPSuccess.Call :=
  ⟨g.arguments,r,u⟩
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Raw
