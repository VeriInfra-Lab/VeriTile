import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Host
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Grid
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

def Post {α : Type} (g : Geometry.Layout) (out : RegionName) (initial final : State α)
    (target : Geometry.Cell g → Cell α) : Prop :=
  (∀ x, final.mem out (Geometry.address g x)=target x) ∧
  (∀ r o, (r≠out ∨ ∀ x, o≠Geometry.address g x) → final.mem r o=initial.mem r o)

theorem post_of_lanes {α : Type} (g : Geometry.Layout) (valid : Geometry.Valid g)
    (out : RegionName) (initial final : State α) (target : Geometry.Cell g → Cell α)
    (readback : ∀ pid, pid<Geometry.count g → ∀ i, ∀ h : Geometry.Active g pid i,
      final.mem out (Geometry.offset g pid i)=target (Geometry.index g pid i h))
    (frame : ∀ r o, (r≠out ∨ ∀ pid, pid<Geometry.count g → ∀ i,
      Geometry.Active g pid i → o≠Geometry.offset g pid i) → final.mem r o=initial.mem r o) :
    Post g out initial final target := by
  constructor
  · intro x
    obtain ⟨pid,hp,i,hr,hc⟩ := Geometry.covers g valid x
    have hi := Geometry.owns_active g pid x i hr hc
    have hx : Geometry.index g pid i hi=x := Prod.ext (Fin.ext hr) (Fin.ext hc)
    have ha : Geometry.offset g pid i=Geometry.address g x := by
      simp only [Geometry.offset,Geometry.address,hr,hc]
    simpa only [ha,hx] using readback pid hp i hi
  · intro r o h
    apply frame r o
    exact h.imp id (fun h pid hp i hi => h (Geometry.index g pid i hi))

#axiomsClean post_of_lanes
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001.Grid
