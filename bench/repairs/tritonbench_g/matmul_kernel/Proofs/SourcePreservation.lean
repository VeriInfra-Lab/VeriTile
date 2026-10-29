import bench.repairs.tritonbench_g.matmul_kernel.Kernels
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.LegacyPastedSurface
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource.SourcePreservation
open VeriTile.Triton
set_option maxRecDepth 65536
set_option maxHeartbeats 2000000
/-- Every source statement, including the live dot and both conversions. -/
theorem original (C A B : RegionName) (cfg : Config) :
    (typed C A B cfg).surfaceBody = (LegacySource.typed C A B cfg).surfaceBody := by
  rfl
theorem candidate (C A B : RegionName) (cfg : Config) :
    (typedRemaining C A B cfg).surfaceBody = (LegacySource.typedRemaining C A B cfg).surfaceBody := by
  rfl
/-- Exact complete kernels, so execution failure, successful final state and
all memory/trace observations transfer for arbitrary primitive models. -/
theorem original_kernel (C A B : RegionName) (cfg : Config) :
    typed C A B cfg = LegacySource.typed C A B cfg := rfl
theorem candidate_kernel (C A B : RegionName) (cfg : Config) :
    typedRemaining C A B cfg = LegacySource.typedRemaining C A B cfg := rfl
#axiomsClean original_kernel
#axiomsClean candidate_kernel
#axiomsClean original
#axiomsClean candidate
end VeriTile.Bench.Repairs.MatmulKernel.KernelSource.SourcePreservation
