import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.RebasedCumFlat
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.Materialize
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.CumulativeMath

/-! Complete cumulative program, ordinary dense input materialization and
independent logical prefix. Active bounds/collisions follow from padded sizes;
exhausted autotune batches remain successful and write no cells. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeHost
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
open ChunkGatedAttention RealCumProgram RebasedCumExecution RebasedCumAccess
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

structure Canonical (p : Layout) : Prop where
  head : p.headStride = p.T*p.S
  time : p.timeStride = p.S
  channel : p.channelStride = 1
  time_full : p.T/p.BT*p.BT = p.T
  feature_full : p.S/p.BS*p.BS = p.S
  bt : 0 < p.BT
  bs : 0 < p.BS

structure Valid (p : Layout) (heads : Nat) (l : Launch) (c : Nat) : Prop where
  head : l.head < heads
  feature : l.feature < p.S/p.BS
  chunk : c < p.T/p.BT

theorem decoded_valid (p : Layout) (heads base : Nat) (u : BlockState)
    (h : enabled p heads base u) : Valid p heads (decoded p base u) (chunk p base u) := by
  have hx : 0 < p.S/p.BS := by
    by_contra hn
    have hz := Nat.eq_zero_of_not_pos hn
    simp [enabled,hz] at h
  have hy : 0 < p.T/p.BT := by
    by_contra hn
    have hz := Nat.eq_zero_of_not_pos hn
    simp [enabled,hz] at h
  have hd := Grid.decode_bounded hx hy h
  exact ⟨hd.2.2,hd.1,hd.2.1⟩

theorem feature_bound (p : Layout) (heads : Nat) (l : Launch) (c : Nat)
    (hp : Canonical p) (hl : Valid p heads l c) (f : Fin p.BS) : l.feature*p.BS+f.val < p.S := by
  rw [← hp.feature_full]
  exact Address.flatten2_bounded hl.feature f.isLt

theorem time_bound (p : Layout) (heads : Nat) (l : Launch) (c : Nat)
    (hp : Canonical p) (hl : Valid p heads l c) (t : Fin p.BT) : c*p.BT+t.val < p.T := by
  rw [← hp.time_full]
  exact Address.flatten2_bounded hl.chunk t.isLt

theorem local_offset (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) :
    RealCumExecution.offset (tileLayout p l c) localLaunch 0 i =
      l.head*p.headStride+(c*p.BT+i.1.val)*p.timeStride+(l.feature*p.BS+i.2.1.val)*p.channelStride := by
  simp only [RealCumExecution.offset,ChunkCumsumVector.RealStep.address,tileLayout,localLaunch,tileBase]
  ring

theorem noCollision (p : Layout) (heads : Nat) (l : Launch) (c : Nat)
    (hp : Canonical p) (hl : Valid p heads l c) : RealCumExecution.NoCollision (tileLayout p l c) localLaunch 0 := by
  intro i j _ _ heq
  rw [local_offset,local_offset,hp.head,hp.time,hp.channel] at heq
  simp only [Nat.mul_one] at heq
  have heq' : (l.head*p.T+(c*p.BT+i.1.val))*p.S+(l.feature*p.BS+i.2.1.val) =
      (l.head*p.T+(c*p.BT+j.1.val))*p.S+(l.feature*p.BS+j.2.1.val) := by
    simpa [Nat.add_mul,Nat.mul_assoc] using heq
  have h := StagedLayout.split_address (feature_bound p heads l c hp hl i.2.1)
    (feature_bound p heads l c hp hl j.2.1) heq'
  have ht : i.1=j.1 := Fin.ext (by omega)
  have hf : i.2.1=j.2.1 := Fin.ext (by omega)
  rcases i with ⟨ti,fi,⟨⟩⟩
  rcases j with ⟨tj,fj,⟨⟩⟩
  cases ht
  cases hf
  rfl

theorem bounds (p : Layout) (heads : Nat) (l : Launch) (c : Nat) (s o : RegionName)
    (hp : Canonical p) (hl : Valid p heads l c) (b : RegionBounds)
    (hs : heads*p.T*p.S ≤ b s) (ho : heads*p.T*p.S ≤ b o) :
    RebasedCumSafety.Bounds p l c s o b := by
  have hb (r : RegionName) (hr : heads*p.T*p.S ≤ b r) :
      RealHAccessSafety.MatrixBounds r (tileBase p l c) p.BT p.BS p.BT p.BS p.timeStride p.channelStride 0 0 b := by
    intro t f _
    have h := lt_of_lt_of_le (Address.flatten3_bounded hl.head (time_bound p heads l c hp hl t)
      (feature_bound p heads l c hp hl f)) hr
    convert h using 1 <;> simp only [tileBase,hp.head,hp.time,hp.channel,Nat.zero_add,Nat.mul_one] <;> ring
  exact ⟨hb s hs,hb o ho⟩

def materialized (p : Layout) (T S : Nat) (x : Nat → Nat → Nat → ℝ) : Nat → ℝ :=
  Materialize.dense p.T p.S (fun h => Host.zeroPad (x h) T S)

theorem expected_eq (p : Layout) (heads T S : Nat) (l : Launch) (c : Nat)
    (hp : Canonical p) (hl : Valid p heads l c) (x : Nat → Nat → Nat → ℝ)
    (i : TileIndex [p.BT,p.BS]) (ht : c*p.BT+i.1.val < T) (hf : l.feature*p.BS+i.2.1.val < S) :
    RealCumExecution.expected (tileLayout p l c) localLaunch 0 (materialized p T S x) i =
      Host.chunkPrefix p.BT (x l.head) (c*p.BT+i.1.val) (l.feature*p.BS+i.2.1.val) := by
  rw [← CumulativeMath.prefix_tile p.BT c (l.feature*p.BS+i.2.1.val) i.1 (x l.head)]
  apply Finset.sum_congr rfl
  intro u hu
  have hle : u.val ≤ i.1.val := (Finset.mem_filter.mp hu).2
  have hlog : c*p.BT+u.val < T := by omega
  have htp := time_bound p heads l c hp hl u
  have hfp := feature_bound p heads l c hp hl i.2.1
  have hoff : ChunkCumsumVector.RealStep.address (tileLayout p l c) localLaunch (0*(tileLayout p l c).BT+u.val) i.2.1.val =
      l.head*(p.T*p.S)+(c*p.BT+u.val)*p.S+(l.feature*p.BS+i.2.1.val) := by
    simp only [ChunkCumsumVector.RealStep.address,tileLayout,localLaunch,tileBase,hp.head,hp.time,hp.channel]
    ring
  rw [hoff]
  exact (Materialize.dense_read p.T p.S l.head _ _ _ htp hfp).trans
    (Host.zeroPad_valid _ T S _ _ hlog hf)

theorem executes (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (p : Layout) (heads T S : Nat) (s o : RegionName) (base : Nat) (swapped : Bool)
    (dtype out_dtype : FloatDType) (x : Nat → Nat → Nat → ℝ) (u : BlockState)
    (hp : Canonical p) (hs : heads*p.T*p.S ≤ A.extent s) (ho : heads*p.T*p.S ≤ A.extent o)
    (hx : ∀ off, u.readMem s off = materialized p T S x off) :
    ∃ v, exec (A.flattenKernel (RebasedCumProgram.source p s o heads base swapped dtype out_dtype).eraseDType.toAlgKernel)
        (A.flattenState u) = some v ∧
      (enabled p heads base u → ∀ i : TileIndex [p.BT,p.BS],
        chunk p base u*p.BT+i.1.val < T → (decoded p base u).feature*p.BS+i.2.1.val < S →
        v.readMem A.flat (A.addr o (RealCumExecution.offset (q p base u) localLaunch 0 i)) =
          Host.chunkPrefix p.BT (x (decoded p base u).head)
            (chunk p base u*p.BT+i.1.val) ((decoded p base u).feature*p.BS+i.2.1.val)) ∧
      BlockState.WriteWithin (RebasedCumFlat.footprint A p heads base o u) (A.flattenState u) v := by
  obtain ⟨v,he,hout,hframe⟩ := RebasedCumFlat.source_executes A hd hcov p s o heads base swapped dtype out_dtype u
    (fun he => noCollision p heads (decoded p base u) (chunk p base u) hp (decoded_valid p heads base u he))
    (fun he => bounds p heads (decoded p base u) (chunk p base u) s o hp (decoded_valid p heads base u he) A.extent hs ho)
  refine ⟨v,he,?_,hframe⟩
  intro ha i ht hf
  rw [hout ha i]
  have hm : u.readMem s = materialized p T S x := funext hx
  rw [hm]
  exact expected_eq p heads T S (decoded p base u) (chunk p base u) hp (decoded_valid p heads base u ha) x i ht hf

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeHost
