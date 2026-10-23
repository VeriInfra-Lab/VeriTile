import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.InterGrid
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.InterCells
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry RealInterMath
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem preserves_not_lower (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) (hn : ¬j.val/16<t.val%c.BT/16) :
    (InterGrid.result c scale s hu).mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      s.mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val) := by
  apply InterGrid.unwritten
  rintro q ⟨_,hen,i,hi,he⟩
  have heo : outputAddress (layout c) (launchIndex c (owner c h t j)) (lane c t j).1.val (lane c t j).2.1.val=
      outputAddress (layout c) (launchIndex c q) i.1.val i.2.1.val := (owner_address c h t j).trans he
  have hqo := (address_injective c _ _ _ _ (owner_active c h t j) hi heo).1
  have hen' : enabled (layout c) (launchIndex c (owner c h t j)) := by simpa only [hqo] using hen
  have hj : j.val/16<c.NC := quotient_bound (by simpa [c.bt_eq] using j.isLt) (by decide)
  have hp : (t.val%c.BT/16*c.NC+j.val/16)/c.NC=t.val%c.BT/16 := by
    rw [Nat.mul_comm _ c.NC,Nat.mul_add_div c.nc_pos,Nat.div_eq_of_lt hj,Nat.add_zero]
  apply hn
  simpa only [owner_launch,rowBlock,columnBlock,layout,Nat.add_mod,Nat.mul_mod_left,Nat.zero_add,
    Nat.mod_eq_of_lt hj,hp] using hen'.2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.InterCells
