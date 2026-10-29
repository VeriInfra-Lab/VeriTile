import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.SplitTail
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.MergeGrid
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.SplitMerge
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape bt_pos)
open scoped BigOperators
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def result (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) : BlockState :=
  MergeGrid.result c (SplitGrid.result c scale s hu) hu

theorem tile_sum (c : Shape) (a : Logical.Inputs) (scale : ℝ)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) (hsame : j.val/16=t.val%c.BT/16) :
    (∑ f ∈ Finset.range (Geometry.ceilDiv c.K c.BK),
      SplitGrid.tileValue c a scale f h.val t.val (j.val%16))=
      Logical.attention c.T c.K c.BT a scale h.val t.val j.val := by
  let l : RealSplitMath.Launch := ⟨t.val/c.BT,t.val%c.BT/16,h.val,0,c.heads⟩
  have hs := Logical.split_expected_sum (SplitGrid.layout c) l a scale (t.val%c.BT%16) (j.val%16)
    rfl rfl c.bk_pos
  have hdt := Nat.div_add_mod' t.val c.BT
  have hdti := Nat.div_add_mod' (t.val%c.BT) 16
  have hdj := Nat.div_add_mod' j.val 16
  have hr : RealSplitMath.rowStart (SplitGrid.layout c) l+t.val%c.BT%16=t.val := by
    dsimp [RealSplitMath.rowStart,SplitGrid.layout,l]
    omega
  have hc : RealSplitMath.rowStart (SplitGrid.layout c) l+j.val%16=t.val/c.BT*c.BT+j.val := by
    dsimp [RealSplitMath.rowStart,SplitGrid.layout,l]
    omega
  have hij : j.val%16≤t.val%c.BT%16 ↔ j.val≤t.val%c.BT := by omega
  rw [hr,hc] at hs
  simpa only [hij,Logical.attention] using hs

theorem readback (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (hzero : ∀ (f : Fin (Geometry.ceilDiv c.K c.BK)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16),
      s.readMem "scratch" (((f.val*c.heads+h.val)*c.T+t.val)*16+j.val)=0)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) (hsame : j.val/16=t.val%c.BT/16) :
    (result c scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val := by
  rw [result,MergeGrid.readback c (SplitGrid.result c scale s hu) hu h t j hsame]
  rw [←tile_sum c (InterGrid.inputs s) scale h t j hsame]
  apply Finset.sum_congr rfl
  intro f hf
  exact SplitTail.readback c scale s hu hzero ⟨f,Finset.mem_range.mp hf⟩ h t ⟨j.val%16,Nat.mod_lt _ (by decide)⟩

#axiomsClean readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.SplitMerge
