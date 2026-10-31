import bench.tritonbench_g.int_scaled_matmul.Sources.Kernels
import VeriTile.Meta.StatementAudit
import VeriTile.Meta.Specification
namespace VeriTile.Bench.Repairs.IntScaledMatmul.SourcePreservation
open Revision001
noncomputable section
/-- The proof decomposition names bind directly to the actual repaired programs. -/
theorem unscaled (p : BlockWord.Parameters) : Kernels.unscaled p=BlockWord.source p := by rfl
theorem scaled (p : ScaledWord.Parameters) : Kernels.scaled p=ScaledWord.source p := by rfl
/-- Both optimization targets are actual distinct complete programs. -/
theorem unscaled_changed (p : BlockWord.Parameters) : Kernels.unscaled p≠Kernels.unscaledCopyFree p := by
  intro same
  have lengths := congrArg (fun k : VeriTile.Triton.ComputeKernel=>k.surfaceBody.length) same
  change 16=15 at lengths
  omega
theorem scaled_changed (p : ScaledWord.Parameters) : Kernels.scaled p≠Kernels.scaledSharedIndices p := by
  intro same
  have lengths := congrArg (fun k : VeriTile.Triton.ComputeKernel=>k.surfaceBody.length) same
  change 25=23 at lengths
  omega
#axiomsClean unscaled
#axiomsClean scaled
end
end VeriTile.Bench.Repairs.IntScaledMatmul.SourcePreservation
