import bench.repairs.tritonbench_g.chunk_delta_fwd.Geometry
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Ownership
open VeriTile Triton
open Geometry
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (owner_lt)
set_option maxHeartbeats 6000000

def pid (p : Parameters) (head value : Nat) : PID := (value/p.BV,head)
def hLane (p : Parameters) (h : ShapeValid p) (key value : Nat) (hk : key < p.K) : HIndex p :=
  ⟨⟨key,lt_of_lt_of_le hk h.k_cover⟩,⟨value%p.BV,Nat.mod_lt _ h.bv_pos⟩,()⟩
def cLane (p : Parameters) (h : ShapeValid p) (t value : Nat) : CIndex p :=
  ⟨⟨(t%p.BT)%p.BC,Nat.mod_lt _ h.bc_pos⟩,⟨value%p.BV,Nat.mod_lt _ h.bv_pos⟩,()⟩
def block (p : Parameters) (t : Nat) := (t%p.BT)/p.BC

theorem valid (p : Parameters) (h : ShapeValid p) (head value : Nat)
    (hh : head < p.B*p.H) (hv : value < p.V) : ValidPID p (pid p head value) :=
  ⟨owner_lt h.bv_pos hv,hh⟩

theorem column_h (p : Parameters) (h : ShapeValid p) (head key value : Nat) (hk : key < p.K) :
    column p (pid p head value) (hLane p h key value hk).2.1.val=value := by
  simp only [column,pid,hLane]
  exact Nat.div_add_mod' value p.BV
theorem column_c (p : Parameters) (h : ShapeValid p) (head t value : Nat) :
    column p (pid p head value) (cLane p h t value).2.1.val=value := by
  simp only [column,pid,cLane]
  exact Nat.div_add_mod' value p.BV
theorem time_c (p : Parameters) (h : ShapeValid p) (t value : Nat) :
    time p (t/p.BT) (block p t) (cLane p h t value).1.val=t := by
  have h1 := Nat.div_add_mod' (t%p.BT) p.BC
  have h2 := Nat.div_add_mod' t p.BT
  simp only [time,block,cLane]
  omega

theorem h_active (p : Parameters) (h : ShapeValid p) (head key value : Nat)
    (hk : key < p.K) (hv : value < p.V) : hActive p (pid p head value) (hLane p h key value hk) := by
  exact ⟨hk,by rw [column_h]; exact hv⟩
theorem c_active (p : Parameters) (h : ShapeValid p) (head t value : Nat)
    (ht : t < p.T) (hv : value < p.V) :
    cActive p (pid p head value) (t/p.BT) (block p t) (cLane p h t value) := by
  exact ⟨by rw [time_c]; exact ht,by rw [column_c]; exact hv⟩

theorem block_lt (p : Parameters) (h : ShapeValid p) (t : Nat) : block p t < innerCount p :=
  owner_lt h.bc_pos (Nat.mod_lt t h.bt_pos)
theorem chunk_lt (p : Parameters) (h : ShapeValid p) (t : Nat) (ht : t < p.T) : t/p.BT < p.NT :=
  (last_chunk p h t ht).1
theorem base_span (p : Parameters) (h : ShapeValid p) (t value : Nat) :
    time p (t/p.BT) (block p t) (cLane p h t value).1.val < (t/p.BT+1)*p.BT := by
  rw [time_c]
  have hm := Nat.mod_lt t h.bt_pos
  have hd := Nat.div_add_mod' t p.BT
  simp only [Nat.add_mul,Nat.one_mul]
  omega

theorem h_offset (p : Parameters) (h : ShapeValid p) (head c key value : Nat) (hk : key < p.K) :
    hOffset p (pid p head value) c (hLane p h key value hk)=
      ((head*p.NT+c)*p.K+key)*p.V+value := by
  rw [history_address,column_h]
  rfl
theorem c_offset (p : Parameters) (h : ShapeValid p) (head t value : Nat) :
    cOffset p (pid p head value) (t/p.BT) (block p t) (cLane p h t value)=
      (head*p.T+t)*p.V+value := by
  rw [corrected_address,time_c,column_c]
  rfl
theorem f_offset (p : Parameters) (h : ShapeValid p) (head key value : Nat) (hk : key < p.K) :
    fOffset p (pid p head value) (hLane p h key value hk)=(head*p.K+key)*p.V+value := by
  rw [final_address,column_h]
  rfl

#axiomsClean valid
#axiomsClean h_offset
#axiomsClean c_offset
#axiomsClean f_offset
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Ownership
