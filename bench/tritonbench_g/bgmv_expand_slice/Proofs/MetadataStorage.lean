import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Pasted
noncomputable section
/-- Complete source equality for both native signed metadata widths, after
its explicit int64 widening and before all execution or numerical premises. -/
theorem original_metadata_widths : originalKernel .i32 = originalKernel .i64 := by rfl

theorem candidate_metadata_widths : candidateKernel .i32 = candidateKernel .i64 := by rfl

#axiomsClean original_metadata_widths
#axiomsClean candidate_metadata_widths
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Pasted
