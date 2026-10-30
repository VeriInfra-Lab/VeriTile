import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelPhysical

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract

/-- The selected backward branch differs even on small exact inputs. -/
theorem backward_not_forward : Spec.pair 1 2 3 4=(11,2) ∧ Spec.forward 1 2 3 4=(-5,10) := by
  norm_num [Spec.pair,Spec.forward]

/-- The fixed-coefficient adjoint is not an inverse without normalization. -/
theorem nonunit_not_inverse :
    let f:=Spec.forward 1 2 2 0
    Spec.pair f.1 f.2 2 0=(4,8) ∧ Spec.pair f.1 f.2 2 0≠(1,2) := by
  norm_num [Spec.pair,Spec.forward,Prod.mk.injEq]

theorem odd_tail (c : Call) (key : Bool) (row head : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key)
    (odd : c.geometry.dimension%2=1) :
    ¬Written c.geometry key (c.geometry.offset key row head (c.geometry.dimension-1)) := by
  rw [written_coordinate c key row head (c.geometry.dimension-1) hr hh (by omega)]
  unfold Request.half
  omega

/-- Individually valid small metadata can still form an invalid i32 offset. -/
theorem source_index_counterexample :
    let r : Request:=⟨1,2^30+1,1,1,2,2^30+1,2^30+1,1,1,1,1⟩
    r.rows<2^31 ∧ r.qHeads*r.dimension<2^31 ∧
      outputLast r Bool.false=2^31+1 ∧ ¬outputLast r Bool.false<2^31 := by decide

#axiomsClean backward_not_forward
#axiomsClean nonunit_not_inverse
#axiomsClean odd_tail
#axiomsClean source_index_counterexample
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
