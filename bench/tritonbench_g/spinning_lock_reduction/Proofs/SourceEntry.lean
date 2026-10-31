import bench.tritonbench_g.spinning_lock_reduction.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SpinningLock
/-- The entire repaired source lowers successfully to the program used by
execution, including every address, CAS, barrier and control-flow statement. -/
theorem kernel_source_preserved : SourceAdapter.decode source = .ok kernel :=
  kernelDecoded.property
theorem area_kernel_source_preserved : SourceAdapter.decode areaSource = .ok areaKernel :=
  areaKernelDecoded.property
#axiomsClean kernel_source_preserved
#axiomsClean area_kernel_source_preserved
end VeriTile.Bench.Repairs.SpinningLock
