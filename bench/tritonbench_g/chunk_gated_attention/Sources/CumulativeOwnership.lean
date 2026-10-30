import bench.tritonbench_g.chunk_gated_attention.Sources.CumulativeHost
import bench.tritonbench_g.chunk_gated_attention.Sources.Batch

/-! An active cumulative output cell uniquely determines its linear program;
guarded dummy programs own no cells. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeOwnership
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram RebasedCumExecution RebasedCumAccess
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem coordinates (p : Layout) (heads : Nat) (l m : Launch) (c d : Nat)
    (hp : CumulativeHost.Canonical p) (hl : CumulativeHost.Valid p heads l c) (hm : CumulativeHost.Valid p heads m d)
    (i j : TileIndex [p.BT,p.BS])
    (he : RealCumExecution.offset (tileLayout p l c) localLaunch 0 i =
      RealCumExecution.offset (tileLayout p m d) localLaunch 0 j) :
    l.feature=m.feature ∧ c=d ∧ l.head=m.head := by
  rw [CumulativeHost.local_offset,CumulativeHost.local_offset,hp.head,hp.time,hp.channel] at he
  have he' : (l.head*p.T+(c*p.BT+i.1.val))*p.S+(l.feature*p.BS+i.2.1.val) =
      (m.head*p.T+(d*p.BT+j.1.val))*p.S+(m.feature*p.BS+j.2.1.val) := by
    simpa [Nat.mul_one,Nat.add_mul,Nat.mul_assoc] using he
  have h1 := StagedLayout.split_address (CumulativeHost.feature_bound p heads l c hp hl i.2.1)
    (CumulativeHost.feature_bound p heads m d hp hm j.2.1) he'
  have h2 := StagedLayout.split_address (CumulativeHost.time_bound p heads l c hp hl i.1)
    (CumulativeHost.time_bound p heads m d hp hm j.1) h1.1
  exact ⟨(StagedLayout.split_address i.2.1.isLt j.2.1.isLt h1.2).1,
    (StagedLayout.split_address i.1.isLt j.1.isLt h2.2).1,h2.1⟩

theorem owner (p : Layout) (heads base otherBase : Nat) (o : RegionName) (u v : BlockState)
    (hp : CumulativeHost.Canonical p) (cell : MemCellAddr)
    (hu : RebasedCumExecution.footprint p heads base o u cell)
    (hv : RebasedCumExecution.footprint p heads otherBase o v cell) : linear base u=linear otherBase v := by
  obtain ⟨hu,_,i,hi,hei⟩ := hu
  obtain ⟨hv,_,j,hj,hej⟩ := hv
  have hl := CumulativeHost.decoded_valid p heads base u hu
  have hm := CumulativeHost.decoded_valid p heads otherBase v hv
  have hc := coordinates p heads (decoded p base u) (decoded p otherBase v) (chunk p base u) (chunk p otherBase v)
    hp hl hm i j (hei.symm.trans hej)
  have hx : 0 < p.S/p.BS := Nat.zero_lt_of_lt hl.feature
  have hy : 0 < p.T/p.BT := Nat.zero_lt_of_lt hl.chunk
  have he : Grid.encode (p.S/p.BS) (p.T/p.BT) (decoded p base u).feature (chunk p base u) (decoded p base u).head =
      Grid.encode (p.S/p.BS) (p.T/p.BT) (decoded p otherBase v).feature (chunk p otherBase v) (decoded p otherBase v).head := by
    rw [hc.1,hc.2.1,hc.2.2]
  change Grid.encode _ _ (Grid.decodeX _ _) (Grid.decodeY _ _ _) (Grid.decodeZ _ _ _) =
    Grid.encode _ _ (Grid.decodeX _ _) (Grid.decodeY _ _ _) (Grid.decodeZ _ _ _) at he
  rwa [Grid.decode_encode_identity hx hy,Grid.decode_encode_identity hx hy] at he

theorem bounded (p : Layout) (heads base : Nat) (o : RegionName) (u : BlockState)
    (hp : CumulativeHost.Canonical p) (bounds : RegionBounds) (hb : heads*p.T*p.S ≤ bounds o)
    (rg : RegionName) (off : Nat) (hw : RebasedCumExecution.footprint p heads base o u (rg,off)) :
    off < bounds rg := by
  rcases hw with ⟨ha,rfl,i,hi,rfl⟩
  have hl := CumulativeHost.decoded_valid p heads base u ha
  have h := Address.flatten3_bounded hl.head
    (CumulativeHost.time_bound p heads _ _ hp hl i.1) (CumulativeHost.feature_bound p heads _ _ hp hl i.2.1)
  apply lt_of_lt_of_le _ hb
  simpa [q,CumulativeHost.local_offset,hp.head,hp.time,hp.channel] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeOwnership
