import bench.tritonbench_g.bgmv_expand_slice.SignedSetup
import bench.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPValues

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPReadback
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem active_lanes : (TileShape.allIndices [256]).filter (fun i => decide (i.1.val<1))=
    [(⟨0,by decide⟩,PUnit.unit)] := by decide

theorem active_lanes_zero : (TileShape.allIndices [256]).filter (fun i => decide (i.1=0))=
    [(⟨0,by decide⟩,PUnit.unit)] := by decide

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Direct.FPReadback
