import bench.tritonbench_g.chunk_gla_fwd.Proofs.SplitGrid
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.SplitTail
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos)
open SplitGeometry SplitGrid
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

theorem tail_unwritten (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16)
    (hc : ¬t.val/c.BT*c.BT+t.val%c.BT/16*16+j.val<c.T) :
    (SplitGrid.result c scale s hu).mem "scratch" (((f.val*c.heads+h.val)*c.T+t.val)*16+j.val)=
      s.mem "scratch" (((f.val*c.heads+h.val)*c.T+t.val)*16+j.val) := by
  apply SplitGrid.unwritten
  intro q hw
  obtain ⟨_,i,hi,he⟩ := footprint_tile c q _ hw
  have ha : active c (launchIndex c (owner c f h t)) (ownerLane c t j) := by
    unfold active
    rw [owner_row]
    exact t.isLt
  have hei : address c (launchIndex c (owner c f h t)) (ownerLane c t j)=address c (launchIndex c q) i :=
    (owner_address c f h t j).trans he
  obtain ⟨hqo,hio⟩ := address_injective c _ _ _ _ ha hi hei
  rcases hw with ⟨hen,hr,z,hz,il,hil,he'⟩
  have hz16 : z<16 := Nat.lt_of_lt_of_le hz (Nat.min_le_left _ _)
  let iz : TileIndex [16,16] := (il.1,⟨z,hz16⟩,⟨⟩)
  have hae : address c (launchIndex c (owner c f h t)) (ownerLane c t j)=address c (launchIndex c q) iz := by
    rw [owner_address]
    change _=RealSplitMath.outputAddress (layout c) (launchIndex c q) iz.1.val iz.2.1.val at he'
    rw [source_address c _ rfl] at he'
    exact he'
  have hizi : active c (launchIndex c q) iz := hil
  have hii := (address_injective c _ _ _ _ ha hizi hae).2
  have hzj : z=j.val := by
    have h := congrArg (fun x : TileIndex [16,16] => x.2.1.val) hii
    exact h.symm
  have hcols : z<RealSplitMath.columns (layout c) (launchIndex c (owner c f h t)) := by
    simpa only [hqo] using hz
  rw [owner_launch] at hcols
  dsimp [RealSplitMath.columns,layout] at hcols
  exact hc (by omega)

theorem tail_value_zero (c : Shape) (a : Logical.Inputs) (scale : ℝ)
    (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16)
    (hc : ¬t.val/c.BT*c.BT+t.val%c.BT/16*16+j.val<c.T) :
    tileValue c a scale f.val h.val t.val j.val=0 := by
  have hr := owner_row c f h t j
  simp only [owner_launch,row,ownerLane] at hr
  have ht := t.isLt
  have hij : ¬j.val≤t.val%c.BT%16 := by omega
  simp only [tileValue,RealSplitMath.expected,hij,ite_false]

/-- Includes every live row and all sixteen scratch columns. Unstreamed
columns of a final time block retain their allocated zero value. -/
theorem readback (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (hzero : ∀ (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16),
      s.readMem "scratch" (((f.val*c.heads+h.val)*c.T+t.val)*16+j.val)=0)
    (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16) :
    (SplitGrid.result c scale s hu).readMem "scratch" (((f.val*c.heads+h.val)*c.T+t.val)*16+j.val)=
      tileValue c (InterGrid.inputs s) scale f.val h.val t.val j.val := by
  by_cases hc : t.val/c.BT*c.BT+t.val%c.BT/16*16+j.val<c.T
  · exact readback_streamed c scale s hu f h t j hc
  · rw [tail_value_zero c _ scale f h t j hc]
    unfold BlockState.readMem
    rw [tail_unwritten c scale s hu f h t j hc]
    exact hzero f h t j

#axiomsClean readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.SplitTail
