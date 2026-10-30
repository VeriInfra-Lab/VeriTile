import bench.tritonbench_g.streamk_matmul.Kernels
import bench.tritonbench_g.streamk_matmul.Proofs.LegacySource
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.StreamK.TileChecked
set_option maxRecDepth 50000
/-- Preservation covers all helper and stage bodies in the original set. -/
theorem kernels_source_preserved : kernels = SourceLegacy.kernels := rfl
/-- Preservation covers the complete candidate set, including shared bodies. -/
theorem area_kernels_source_preserved : areaKernels = SourceLegacy.areaKernels := rfl
#axiomsClean kernels_source_preserved
#axiomsClean area_kernels_source_preserved
end VeriTile.Bench.Repairs.StreamK.TileChecked
