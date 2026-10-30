import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativeOrdinary
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativeIOOutputs
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.IOSafetyComplete

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000

/-- The existing ordinary source grid supplies success and its disjoint merge.
Finite channel binding supplies the independent formula; capacities establish
actual source trace safety without restricting the output or frame theorem. -/
theorem implements {α : Type} [Inhabited α] (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (width : widthCheck N=Bool.true) (sep : Domain.ReadWriteSeparated f r)
    (geometry : NativeGeometry.Geometry f r stride M N) :
    (io A f r stride M N eps p).Implements
      (target A cmp f r stride M N eps p ((widthCheck_iff N).mp width)) := by
  intro a _ s hin
  obtain ⟨t,⟨executed⟩,post⟩:=TypedSource.source_grid_post A cmp f r stride M N eps p
    (tile N) (tile_arange N) s sep geometry
  refine ⟨t,⟨ordinary_of_original A f r stride M N eps p a s t executed⟩,?_,
    output_binding A cmp ((widthCheck_iff N).mp width) a s t hin post,frame A cmp a s t post⟩
  intro bounds backed i
  change (List.ofFn (fun row : Fin M => invocation A f r stride M N eps p row.val))[i.val].Safe bounds s
  simp only [List.getElem_ofFn]
  change TraceSafe (engine A) bounds (quoted f r stride M N eps p) (TypedSource.atRow s M i.val)
  rw [source_exact]
  exact IOSafety.complete A cmp f r stride M N eps p (tile N) (tile_arange N)
    (TypedSource.atRow s M i.val) bounds
    (row_bounds A a bounds backed ⟨i.val,by simpa only [io,List.length_ofFn] using i.isLt⟩)

#axiomsClean implements
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.NativeInterface
