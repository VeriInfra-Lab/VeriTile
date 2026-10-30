import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Coordinates
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.OutputLayout
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Relocation
import VeriTile.Triton.Launch.Composition

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.GridCoordinates
open VeriTile.Triton Memory Relocation
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The host omits every empty launch, including D=0. -/
def grid (p : Parameters) : Grid :=
  ⟨[if p.headDim=0 then 0 else (p.context+127)/128,p.batches*p.heads,1]⟩

def cfg (A : FlatAlloc) (s : BlockState) (p : Parameters) (idx : GridIndex (grid p)) : Config :=
  config A (Coordinates.configuration (s.withGridIndex idx) p)

theorem row (A : FlatAlloc) (s : BlockState) (p : Parameters) (idx : GridIndex (grid p)) :
    (cfg A s p idx).startM = (idx ⟨0,by simp [grid,Grid.rank]⟩).val := by
  simp [cfg,config,Coordinates.configuration,BlockState.withGridIndex,BlockState.withPids,
    BlockState.withNumPids,GridIndex.toPids,grid,Grid.rank]

theorem batch (A : FlatAlloc) (s : BlockState) (p : Parameters) (idx : GridIndex (grid p)) :
    (cfg A s p idx).z = (idx ⟨1,by simp [grid,Grid.rank]⟩).val/p.heads := by
  simp [cfg,config,Coordinates.configuration,BlockState.withGridIndex,BlockState.withPids,
    BlockState.withNumPids,GridIndex.toPids,grid,Grid.rank]

theorem head (A : FlatAlloc) (s : BlockState) (p : Parameters) (idx : GridIndex (grid p)) :
    (cfg A s p idx).h = (idx ⟨1,by simp [grid,Grid.rank]⟩).val%p.heads := by
  simp [cfg,config,Coordinates.configuration,BlockState.withGridIndex,BlockState.withPids,
    BlockState.withNumPids,GridIndex.toPids,grid,Grid.rank]

theorem program (A : FlatAlloc) (s : BlockState) (p : Parameters) (idx : GridIndex (grid p)) :
    ViewBounds.Program (cfg A s p idx) := by
  have h0 := (idx ⟨0,by simp [grid,Grid.rank]⟩).isLt
  have h1 := (idx ⟨1,by simp [grid,Grid.rank]⟩).isLt
  change _ < (if p.headDim=0 then 0 else (p.context+127)/128) at h0
  change _ < p.batches*p.heads at h1
  have hH : 0 < p.heads := by nlinarith
  constructor
  · change (cfg A s p idx).z < p.batches
    rw [batch]
    exact (Nat.div_lt_iff_lt_mul hH).mpr h1
  · change (cfg A s p idx).h < p.heads
    rw [head]
    exact Nat.mod_lt _ hH
  · change (cfg A s p idx).startM*128 < p.context
    rw [row]
    split_ifs at h0 <;> omega

theorem index_eq (A : FlatAlloc) (s : BlockState) (p : Parameters) (a b : GridIndex (grid p))
    (hr : (cfg A s p a).startM = (cfg A s p b).startM)
    (hz : (cfg A s p a).z = (cfg A s p b).z)
    (hh : (cfg A s p a).h = (cfg A s p b).h) : a=b := by
  rw [row,row] at hr
  rw [batch,batch] at hz
  rw [head,head] at hh
  have h1 : (a ⟨1,by simp [grid,Grid.rank]⟩).val = (b ⟨1,by simp [grid,Grid.rank]⟩).val := by
    have ha := Nat.mod_add_div (a ⟨1,by simp [grid,Grid.rank]⟩).val p.heads
    have hb := Nat.mod_add_div (b ⟨1,by simp [grid,Grid.rank]⟩).val p.heads
    rw [hz,hh] at ha
    omega
  funext ax
  obtain ⟨ax,hax⟩ := ax
  have ha : ax<3 := hax
  interval_cases ax
  · exact Fin.ext hr
  · exact Fin.ext h1
  · apply Fin.ext
    have ha := (a ⟨2,hax⟩).isLt
    have hb := (b ⟨2,hax⟩).isLt
    change _<1 at ha hb
    omega

def index (p : Parameters) (i : OutputLayout.Coordinate p) : GridIndex (grid p) := fun ax => by
  obtain ⟨ax,hax⟩ := ax
  have ha : ax<3 := hax
  interval_cases ax
  · refine ⟨i.2.2.1.val/128,?_⟩
    have hD : p.headDim ≠ 0 := Nat.ne_zero_of_lt i.2.2.2.isLt
    simp only [grid,Grid.dim,List.getD_cons_zero,hD,↓reduceIte]
    exact (RealProofs.query_grid_coverage p.context i.2.2.1.val i.2.2.1.isLt).1
  · refine ⟨i.1.val*p.heads+i.2.1.val,?_⟩
    exact RealProofs.head_grid_coverage _ _ _ _ i.1.isLt i.2.1.isLt
  · exact ⟨0,by change 0<1; omega⟩

theorem index_row (A : FlatAlloc) (s : BlockState) (p : Parameters) (i : OutputLayout.Coordinate p) :
    (cfg A s p (index p i)).startM = i.2.2.1.val/128 := by
  rw [row]
  rfl

theorem index_batch (A : FlatAlloc) (s : BlockState) (p : Parameters) (i : OutputLayout.Coordinate p) :
    (cfg A s p (index p i)).z = i.1.val := by
  rw [batch]
  exact (RealProofs.head_grid_decode p.heads i.1.val i.2.1.val i.2.1.isLt).1

theorem index_head (A : FlatAlloc) (s : BlockState) (p : Parameters) (i : OutputLayout.Coordinate p) :
    (cfg A s p (index p i)).h = i.2.1.val := by
  rw [head]
  exact (RealProofs.head_grid_decode p.heads i.1.val i.2.1.val i.2.1.isLt).2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.GridCoordinates
