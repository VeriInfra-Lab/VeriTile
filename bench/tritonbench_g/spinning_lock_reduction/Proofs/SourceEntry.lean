import bench.tritonbench_g.spinning_lock_reduction.Kernels
import bench.tritonbench_g.spinning_lock_reduction.Proofs.LegacySource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SpinningLock
/-- Equality of the entire source-compiled program to the retained typed
program, including every address, CAS, barrier and control-flow statement. -/
theorem kernel_source_preserved : kernel = SourceLegacy.kernel := rfl
theorem area_kernel_source_preserved : areaKernel = SourceLegacy.areaKernel := rfl
#axiomsClean kernel_source_preserved
#axiomsClean area_kernel_source_preserved
end VeriTile.Bench.Repairs.SpinningLock
