import bench.tritonbench_g.decay_cumsum.Proofs.Grid

namespace VeriTile.Bench.Repairs.DecayCumsum.Logical
open Geometry Grid
noncomputable section
set_option maxHeartbeats 5000000

def cells (h : Call) : Nat := h.rows*(h.T*h.DK)
abbrev Coordinate (h : Call) := Fin h.rows × Fin h.T × Fin h.DK

def coordinates (h : Call) (i : Fin (cells h)) : Coordinate h :=
  let outer : Fin h.rows × Fin (h.T*h.DK) := finProdFinEquiv.symm i
  (outer.1,finProdFinEquiv.symm outer.2)

def index (h : Call) (c : Coordinate h) : Fin (cells h) :=
  finProdFinEquiv (c.1,finProdFinEquiv c.2)

theorem coordinates_index (h : Call) (c : Coordinate h) : coordinates h (index h c)=c := by
  simp [coordinates,index]

theorem index_coordinates (h : Call) (i : Fin (cells h)) : index h (coordinates h i)=i := by
  unfold coordinates index
  rw [finProdFinEquiv.apply_symm_apply]
  exact finProdFinEquiv.apply_symm_apply i

def chunkStart (h : Call) (t : Nat) : Nat := t/h.BT*h.BT
def chunkSpan (h : Call) (t : Nat) : Nat := min h.BT (h.T-chunkStart h t)
def start (p : Phase) (prog : Program h p) : Nat := p.start+prog.chunk*p.span

structure Owner (h : Call) (c : Coordinate h) where
  work : Work h
  member : work∈schedule h
  step : Fin work.1.span
  lane : Fin h.BK
  rowEq : work.2.row=c.1.val
  timeEq : Geometry.time work.1 work.2.chunk step.val=c.2.1.val
  columnEq : Geometry.column h work.2.feature lane.val=c.2.2.val

theorem has_owner (h : Call) (c : Coordinate h) : Nonempty (Owner h c) := by
  let k : Fin h.width := ⟨c.2.2.val,c.2.2.isLt.trans_le (width_covers h)⟩
  obtain ⟨p,hp,prog,t,j,hr,ht,hk⟩ := Coverage.coordinate_owner h c.1 c.2.1 k
  exact ⟨⟨⟨p,prog⟩,schedule_member h p hp prog,t,j,hr,ht,hk⟩⟩

def owner (h : Call) (c : Coordinate h) : Owner h c := Classical.choice (has_owner h c)

theorem chunk_start (h : Call) (p : Phase) (hp : p∈phases h) (prog : Program h p) (t : Fin p.span) :
    chunkStart h (Geometry.time p prog.chunk t.val)=start p prog := by
  rcases Coverage.phase_cases h p hp with rfl|rfl
  · have remainder := t.isLt
    dsimp [mainPhase] at remainder
    simp only [chunkStart,start,Geometry.time,mainPhase,Nat.zero_add]
    rw [Nat.add_comm (prog.chunk*h.BT),Nat.add_mul_div_right _ _ h.timeTilePositive,
      Nat.div_eq_of_lt remainder,Nat.zero_add]
  · have chunk : prog.chunk=0 := by have hc := prog.chunkInside; dsimp [tailPhase] at hc; omega
    have remainder : t.val<h.BT := t.isLt.trans (Nat.mod_lt _ h.timeTilePositive)
    simp only [chunkStart,start,Geometry.time,tailPhase,chunk,Nat.zero_mul,Nat.add_zero]
    rw [Nat.add_comm (h.T/h.BT*h.BT),Nat.add_mul_div_right _ _ h.timeTilePositive,
      Nat.div_eq_of_lt remainder,Nat.zero_add]

theorem chunk_span (h : Call) (p : Phase) (hp : p∈phases h) (prog : Program h p) (t : Fin p.span) :
    chunkSpan h (Geometry.time p prog.chunk t.val)=p.span := by
  unfold chunkSpan
  rw [chunk_start h p hp prog t]
  rcases Coverage.phase_cases h p hp with rfl|rfl
  · have chunks := Nat.mul_le_mul_right h.BT (Nat.succ_le_of_lt prog.chunkInside)
    have total := Nat.div_mul_le_self h.T h.BT
    dsimp [start,mainPhase] at *
    apply Nat.min_eq_left
    rw [Nat.add_mul] at chunks
    simp only [Nat.one_mul] at chunks
    omega
  · have chunk : prog.chunk=0 := by have hc := prog.chunkInside; dsimp [tailPhase] at hc; omega
    have decomposition := Nat.mod_add_div h.T h.BT
    have remainder := Nat.mod_lt h.T h.timeTilePositive
    dsimp [start,tailPhase]
    rw [chunk]
    simp only [Nat.zero_mul,Nat.add_zero]
    have rest : h.T-h.T/h.BT*h.BT=h.T%h.BT := by
      rw [Nat.mul_comm h.BT] at decomposition
      omega
    rw [rest,Nat.min_eq_right (Nat.le_of_lt remainder)]

theorem owner_chunk (h : Call) (c : Coordinate h) (o : Owner h c) :
    chunkStart h c.2.1.val=start o.work.1 o.work.2 ∧ chunkSpan h c.2.1.val=o.work.1.span ∧
    c.2.1.val-chunkStart h c.2.1.val=o.step.val := by
  have hp := schedule_valid h o.work o.member
  have hs := chunk_start h o.work.1 hp o.work.2 o.step
  have hn := chunk_span h o.work.1 hp o.work.2 o.step
  rw [o.timeEq] at hs hn
  refine ⟨hs,hn,?_⟩
  rw [hs]
  have ht := o.timeEq
  dsimp [Geometry.time,start] at *
  omega

#axiomsClean owner_chunk
end
end VeriTile.Bench.Repairs.DecayCumsum.Logical
