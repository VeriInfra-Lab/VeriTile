import bench.tritonbench_g.layer_norm_fwd.Proofs.RealIOOutputs
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Safety

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

theorem implements (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (width : widthCheck N=Bool.true)
    (sep : Domain.ReadWriteSeparated f r) (geometry : NativeGeometry.Geometry f r stride M N) :
    (io f r stride M N eps p).Implements (formula f M N eps p) := by
  intro a _ s hin
  obtain ⟨t,executed,post⟩:=RealRevision.ActiveGrid.source_correct f r stride M N eps p
    (tile N) (tile_arange N) ((widthCheck_iff N).mp width) sep geometry s
  refine ⟨t,.cons executed (.nil t),?_,output_binding a s t hin geometry post,frame a s t post⟩
  intro bounds backed
  refine ⟨?_,trivial⟩
  intro idx
  exact Safety.source_safe f r stride M N eps p (tile N) (tile_arange N)
    (s.withGridIndex idx) ((widthCheck_iff N).mp width) sep bounds
    (row_bounds a s bounds backed ⟨WholeGrid.row idx,WholeGrid.row_lt idx⟩)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.RealInterface
