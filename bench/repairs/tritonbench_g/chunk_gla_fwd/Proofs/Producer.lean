import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.InterCells
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.StageFrames
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Producer
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open ProducerGeometry (Shape)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def ZeroAttention (c : Shape) (s : BlockState) : Prop :=
  ∀ (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT),
    s.readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=0

/-- The split kernel always uses BK=128; the inter kernel keeps its own BK. -/
def splitShape (c : Shape) : Shape := {c with BK:=128,bk_pos:=by decide}
def ZeroScratch (c : Shape) (s : BlockState) : Prop :=
  ∀ (f : Fin (Geometry.ceilDiv c.K 128)) (h : Fin c.heads) (t : Fin c.T) (j : Fin 16),
    s.readMem "scratch" (((f.val*c.heads+h.val)*c.T+t.val)*16+j.val)=0

def direct (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :=
  IntraGrid.result c BK hK scale (InterGrid.result c scale s hu) hu
def large (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :=
  SplitMerge.result (splitShape c) scale (InterGrid.result c scale s hu) hu

structure DirectRun (c : Shape) (BK : Nat) (scale : ℝ) (s sf : BlockState) where
  inter : BlockState
  interLaunch : Kernel.GridLaunchedOrdinary (InterGrid.kernel c scale) (ProducerGeometry.grid c) s inter
  intraLaunch : Kernel.GridLaunchedOrdinary (IntraGrid.kernel c BK scale) (DiagonalGeometry.grid c) inter sf

def direct_runs (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    DirectRun c BK scale s (direct c BK hK scale s hu) :=
  ⟨InterGrid.result c scale s hu,InterGrid.launched c scale s hu,
    IntraGrid.launched c BK hK scale (InterGrid.result c scale s hu) hu⟩

structure LargeRun (c : Shape) (scale : ℝ) (s sf : BlockState) where
  inter : BlockState
  split : BlockState
  interLaunch : Kernel.GridLaunchedOrdinary (InterGrid.kernel c scale) (ProducerGeometry.grid c) s inter
  splitLaunch : Kernel.GridLaunchedOrdinary (SplitGrid.kernel (splitShape c) scale) (SplitGeometry.grid (splitShape c)) inter split
  mergeLaunch : Kernel.GridLaunchedOrdinary (MergeGrid.kernel (splitShape c)) (DiagonalGeometry.grid (splitShape c)) split sf

def large_runs (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    LargeRun c scale s (large c scale s hu) :=
  ⟨InterGrid.result c scale s hu,SplitGrid.result (splitShape c) scale (InterGrid.result c scale s hu) hu,
    InterGrid.launched c scale s hu,SplitGrid.launched (splitShape c) scale (InterGrid.result c scale s hu) hu,
    MergeGrid.launched (splitShape c) (SplitGrid.result (splitShape c) scale (InterGrid.result c scale s hu) hu) hu⟩

theorem inter_non_diagonal (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (hz : ZeroAttention c s) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hne : j.val/16≠t.val%c.BT/16) :
    (InterGrid.result c scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val := by
  by_cases hl : j.val/16<t.val%c.BT/16
  · have hc : j.val≤t.val%c.BT := by omega
    simpa only [Logical.attention,hc,ite_true] using InterGrid.readback c scale s hu h t j hl
  · have hc : ¬j.val≤t.val%c.BT := by omega
    simp only [Logical.attention,hc,ite_false]
    unfold BlockState.readMem
    rw [InterCells.preserves_not_lower c scale s hu h t j hl]
    exact hz h t j

theorem inter_diagonal_zero (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (hz : ZeroAttention c s) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT)
    (hsame : j.val/16=t.val%c.BT/16) :
    (InterGrid.result c scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=0 := by
  unfold BlockState.readMem
  rw [InterCells.preserves_not_lower c scale s hu h t j (by omega)]
  exact hz h t j

theorem direct_readback (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ)
    (s : BlockState) (hu : s.undef=fun _ _ => 0) (hz : ZeroAttention c s)
    (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    (direct c BK hK scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val := by
  by_cases he : j.val/16=t.val%c.BT/16
  · have ho := IntraTail.readback c BK hK scale (InterGrid.result c scale s hu) hu
      (inter_diagonal_zero c scale s hu hz) h t j he
    rw [StageFrames.inter_inputs] at ho
    exact ho
  · unfold direct BlockState.readMem
    rw [DiagonalCells.intra_preserves_other_block c BK hK scale (InterGrid.result c scale s hu) hu h t j he]
    exact inter_non_diagonal c scale s hu hz h t j he

theorem large_readback (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0)
    (hz : ZeroAttention c s) (hs : ZeroScratch c s) (h : Fin c.heads) (t : Fin c.T) (j : Fin c.BT) :
    (large c scale s hu).readMem "attention" (h.val*c.T*c.BT+t.val*c.BT+j.val)=
      Logical.attention c.T c.K c.BT (InterGrid.inputs s) scale h.val t.val j.val := by
  by_cases he : j.val/16=t.val%c.BT/16
  · have hs' : ZeroScratch c (InterGrid.result c scale s hu) := by
      intro f hh tt jj
      unfold BlockState.readMem
      rw [StageFrames.inter_region c scale s hu "scratch" (by decide)]
      exact hs f hh tt jj
    have ho := SplitMerge.readback (splitShape c) scale (InterGrid.result c scale s hu) hu hs' h t j he
    rw [StageFrames.inter_inputs] at ho
    exact ho
  · unfold large SplitMerge.result BlockState.readMem
    have hm := DiagonalCells.merge_preserves_other_block (splitShape c)
      (SplitGrid.result (splitShape c) scale (InterGrid.result c scale s hu) hu) hu h t j he
    dsimp only [splitShape] at hm ⊢
    rw [hm]
    have hp := StageFrames.split_region (splitShape c) scale (InterGrid.result c scale s hu) hu "attention" (by decide)
      (h.val*c.T*c.BT+t.val*c.BT+j.val)
    dsimp only [splitShape] at hp
    rw [hp]
    exact inter_non_diagonal c scale s hu hz h t j he

theorem direct_inputs (c : Shape) (BK : Nat) (hK : c.K≤BK) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    InterGrid.inputs (direct c BK hK scale s hu)=InterGrid.inputs s := by
  rw [direct,StageFrames.intra_inputs,StageFrames.inter_inputs]

theorem large_inputs (c : Shape) (scale : ℝ) (s : BlockState) (hu : s.undef=fun _ _ => 0) :
    InterGrid.inputs (large c scale s hu)=InterGrid.inputs s := by
  rw [large,SplitMerge.result,StageFrames.merge_inputs,StageFrames.split_inputs,StageFrames.inter_inputs]

#axiomsClean direct_readback
#axiomsClean large_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Producer
