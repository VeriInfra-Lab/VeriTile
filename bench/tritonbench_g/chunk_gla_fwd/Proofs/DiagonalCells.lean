import bench.tritonbench_g.chunk_gla_fwd.Proofs.IntraGrid
import bench.tritonbench_g.chunk_gla_fwd.Proofs.MergeGrid
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.DiagonalCells
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos)
open DiagonalGeometry
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem cell_same (c : Shape) (q : GridIndex (grid c)) (i : TileIndex [16,16])
    (hi : active c (launchIndex c q) i) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (he : address c (launchIndex c q) i=h.val*c.T*c.BT+t.val*c.BT+j.val) :
    j.val/16=t.val%c.BT/16 := by
  have he' : Geometry.attentionAddress c.T c.BT (launchIndex c q).head (row c (launchIndex c q) i.1.val)
      (column (launchIndex c q) i.2.1.val)=Geometry.attentionAddress c.T c.BT h.val t.val j.val := by
    simpa only [address,Geometry.attentionAddress,Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨_,hr,hc⟩ := Geometry.attention_injective hi.1 t.isLt hi.2 j.isLt he'
  have hr' : (launchIndex c q).chunk*c.BT+((launchIndex c q).block*16+i.1.val)=t.val := by
    simpa only [row,Nat.add_assoc] using hr
  have hm := congrArg (fun n => n%c.BT) hr'
  dsimp only at hm
  simp only [Nat.add_mod,Nat.mul_mod_left,Nat.zero_add,Nat.mod_mod,Nat.mod_eq_of_lt (local_bound c q i.1)] at hm
  have hri := congrArg (fun n => n/16) hm
  have hcj := congrArg (fun n => n/16) hc
  dsimp only at hri
  dsimp only [column] at hcj
  have hi0 := Nat.div_eq_of_lt i.1.isLt
  have hj0 := Nat.div_eq_of_lt i.2.1.isLt
  rw [Nat.mul_comm _ 16,Nat.mul_add_div (by decide),hi0,Nat.add_zero] at hri
  rw [Nat.mul_comm _ 16,Nat.mul_add_div (by decide),hj0,Nat.add_zero] at hcj
  exact hcj.symm.trans hri

theorem intra_preserves_other_block (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ)
    (s : BlockState) (hu : s.undef=fun _ _ => 0) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hne : j.val/16≠t.val%c.BT/16) :
    (IntraGrid.result c BK hK scale s hu).mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      s.mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val) := by
  apply Kernel.mergeFrames_mem_eq_of_not_written
  rintro ⟨q,hq⟩
  obtain ⟨_,i,hi,he⟩ := IntraGrid.footprint_tile c BK q _ hq
  exact hne (cell_same c q i hi h t j he.symm)

theorem merge_preserves_other_block (c : Shape) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) (hne : j.val/16≠t.val%c.BT/16) :
    (MergeGrid.result c s hu).mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      s.mem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val) := by
  apply MergeGrid.unwritten
  rintro q ⟨_,_,i,hi,he⟩
  exact hne (cell_same c q i hi h t j he.symm)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.DiagonalCells
