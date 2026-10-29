import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Kernels
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.LegacySurface
namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.SourcePreservation
/-- Equality of the complete literal-epsilon original source function. -/
theorem original : kernel=LegacySurface.kernel := rfl
/-- Equality of the complete runtime-epsilon original source function. -/
theorem runtime : runtimeKernel=LegacySurface.runtimeKernel := rfl
/-- Equality of the complete literal-epsilon existing candidate function. -/
theorem candidate : indexCommutedKernel=LegacySurface.indexCommutedKernel := rfl
/-- Equality of the complete runtime-epsilon existing candidate function. -/
theorem runtime_candidate : runtimeCandidate=LegacySurface.runtimeCandidate := rfl
#axiomsClean original
#axiomsClean runtime
#axiomsClean candidate
#axiomsClean runtime_candidate
end VeriTile.Bench.Repairs.RmsnormFusedLlama.SourcePreservation
