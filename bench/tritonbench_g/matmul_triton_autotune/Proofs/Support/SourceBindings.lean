import bench.tritonbench_g.matmul_triton_autotune.Proofs.Sources.Mathematical_Host
import bench.tritonbench_g.matmul_triton_autotune.Kernels

/-! Pure call/schedule bindings for the copied repaired-source candidate. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
open VeriTile.Triton Whole
noncomputable section
def sourceKernel (r : Request) : ComputeKernel := Source.kernel r.parameters
abbrev sourceConfigurations := Whole.configurations
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair
