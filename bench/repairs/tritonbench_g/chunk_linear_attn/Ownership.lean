import bench.repairs.tritonbench_g.chunk_linear_attn.Geometry
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Ownership
open VeriTile Triton Geometry
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (owner_lt)
set_option maxHeartbeats 6000000

def statePID (p : Parameters) (head key value : Nat) : PID := ⟨key/p.BK,value/p.BV,head⟩
def outputPID (p : Parameters) (head t value : Nat) : PID := ⟨value/p.BV,t/p.BT,head⟩
def gradientPID (p : Parameters) (head t key : Nat) : PID := ⟨key/p.BK,t/p.BT,head⟩
def partialPID (p : Parameters) (kb head t : Nat) : PID := ⟨kb,t/p.BT,head⟩
def stateLane (p : Parameters) (h : ShapeValid p) (key value : Nat) : TileIndex [p.BK,p.BV] :=
  ⟨⟨key%p.BK,Nat.mod_lt _ h.bk_pos⟩,⟨value%p.BV,Nat.mod_lt _ h.bv_pos⟩,()⟩
def outputLane (p : Parameters) (h : ShapeValid p) (t value : Nat) : TileIndex [p.BT,p.BV] :=
  ⟨⟨t%p.BT,Nat.mod_lt _ h.bt_pos⟩,⟨value%p.BV,Nat.mod_lt _ h.bv_pos⟩,()⟩
def gradientLane (p : Parameters) (h : ShapeValid p) (t key : Nat) : TileIndex [p.BT,p.BK] :=
  ⟨⟨t%p.BT,Nat.mod_lt _ h.bt_pos⟩,⟨key%p.BK,Nat.mod_lt _ h.bk_pos⟩,()⟩

theorem state_valid (p : Parameters) (h : ShapeValid p) (head key value : Nat)
    (hh : head<p.B*p.H) (hk : key<p.K) (hv : value<p.V) : StatePID p (statePID p head key value) :=
  ⟨owner_lt h.bk_pos hk,owner_lt h.bv_pos hv,hh⟩
theorem output_valid (p : Parameters) (h : ShapeValid p) (head t value : Nat)
    (hh : head<p.B*p.H) (ht : t<p.T) (hv : value<p.V) : OutputPID p (outputPID p head t value) := by
  exact ⟨owner_lt h.bv_pos hv,by rw [h.chunks]; exact owner_lt h.bt_pos ht,hh⟩
theorem gradient_valid (p : Parameters) (h : ShapeValid p) (head t key : Nat)
    (hh : head<p.B*p.H) (ht : t<p.T) (hk : key<p.K) : GradientPID p (gradientPID p head t key) := by
  exact ⟨owner_lt h.bk_pos hk,by rw [h.chunks]; exact owner_lt h.bt_pos ht,hh⟩
theorem partial_valid (p : Parameters) (h : ShapeValid p) (kb head t : Nat)
    (hkb : kb<keyBlocks p) (hh : head<p.B*p.H) (ht : t<p.T) : GradientPID p (partialPID p kb head t) := by
  exact ⟨hkb,by rw [h.chunks]; exact owner_lt h.bt_pos ht,hh⟩

theorem state_active (p : Parameters) (h : ShapeValid p) (head key value : Nat)
    (hk : key<p.K) (hv : value<p.V) : hActive p (statePID p head key value) (stateLane p h key value) := by
  simpa [hActive,statePID,stateLane,Nat.div_add_mod'] using And.intro hk hv
theorem output_active (p : Parameters) (h : ShapeValid p) (head t value : Nat)
    (ht : t<p.T) (hv : value<p.V) : outputActive p (outputPID p head t value) (outputLane p h t value) := by
  simpa [outputActive,outputPID,outputLane,Nat.div_add_mod'] using And.intro ht hv
theorem gradient_active (p : Parameters) (h : ShapeValid p) (head t key : Nat)
    (ht : t<p.T) (hk : key<p.K) : gradKeyActive p (gradientPID p head t key) (gradientLane p h t key) := by
  simpa [gradKeyActive,gradientPID,gradientLane,Nat.div_add_mod'] using And.intro ht hk
theorem partial_active (p : Parameters) (h : ShapeValid p) (kb head t value : Nat)
    (ht : t<p.T) (hv : value<p.V) : gradValueActive p (partialPID p kb head t) (value/p.BV) (outputLane p h t value) := by
  simpa [gradValueActive,partialPID,outputLane,Nat.div_add_mod'] using And.intro ht hv

theorem history_offset (p : Parameters) (h : ShapeValid p) (head c key value : Nat) :
    hOffset p (statePID p head key value) c (stateLane p h key value)=((head*p.NT+c)*p.K+key)*p.V+value := by
  rw [history_address]; simp [statePID,stateLane,Nat.div_add_mod']
theorem final_offset (p : Parameters) (h : ShapeValid p) (head key value : Nat) :
    fOffset p (statePID p head key value) (stateLane p h key value)=(head*p.K+key)*p.V+value := by
  rw [final_address]; simp [statePID,stateLane,Nat.div_add_mod']
theorem output_offset (p : Parameters) (h : ShapeValid p) (head t value : Nat) :
    outputOffset p (outputPID p head t value) (outputLane p h t value)=(head*p.T+t)*p.V+value := by
  rw [output_address]; simp [outputPID,outputLane,Nat.div_add_mod']
theorem gradient_offset (p : Parameters) (h : ShapeValid p) (head t key : Nat) :
    gradKeyOffset p (gradientPID p head t key) (gradientLane p h t key)=(head*p.T+t)*p.K+key := by
  rw [gradient_address]; simp [gradientPID,gradientLane,Nat.div_add_mod']
theorem partial_offset (p : Parameters) (h : ShapeValid p) (kb head t value : Nat) :
    gradValueOffset p (partialPID p kb head t) (p.B*p.H) (value/p.BV) (outputLane p h t value)=
      ((kb*(p.B*p.H)+head)*p.T+t)*p.V+value := by
  rw [partial_address]; simp [partialPID,outputLane,Nat.div_add_mod']

theorem chunk_owner (p : Parameters) (rev : Bool) (c : Nat) (hc : c<p.NT) :
    chunk p rev (chunk p rev c)=c := by cases rev <;> simp [chunk] <;> omega

#axiomsClean state_valid
#axiomsClean output_valid
#axiomsClean gradient_valid
#axiomsClean partial_valid
#axiomsClean history_offset
#axiomsClean partial_offset
end VeriTile.Bench.Repairs.ChunkLinearAttn.Ownership
