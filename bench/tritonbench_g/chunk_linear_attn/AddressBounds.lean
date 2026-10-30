import bench.tritonbench_g.chunk_linear_attn.HostMetadata
import bench.tritonbench_g.chunk_linear_attn.Physical
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Host
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv packed_lt)
set_option maxHeartbeats 12000000

def queryWords (c : Dimensions) := c.B*c.H*c.T*c.K
def valueWords (c : Dimensions) := c.B*c.H*c.T*c.V
def historyWords (c : Dimensions) := c.B*c.H*c.NT*c.K*c.V
def partialWords (c : Dimensions) := keyBlocks (c.parameters Bool.true)*c.B*c.H*c.T*c.V
def largest (c : Dimensions) := max (max (queryWords c) (valueWords c)) (max (historyWords c) (partialWords c))
def formedLimit (c : Dimensions) := largest c+256*max c.K c.V
def MachineGuard (c : Dimensions) := formedLimit c<2^31

theorem block_start (N W block : Nat) (hw : 0<W) (hb : block<ceilDiv N W) : block*W<N := by
  have ha := Nat.mul_le_mul_right W (Nat.succ_le_of_lt hb)
  have hd := Nat.div_mul_le_self (N+W-1) W
  unfold ceilDiv at ha
  simp only [Nat.succ_mul,Nat.add_mul,Nat.one_mul] at ha
  omega

theorem padded_coordinate (N W block lane : Nat) (hw : 0<W) (hcap : W≤64)
    (hb : block<ceilDiv N W) (hl : lane<W) : block*W+lane<N+64 := by
  have hs := block_start N W block hw hb
  omega

theorem padded_address (heads rows cols head row col : Nat)
    (hh : head<heads) (hr : row<rows+64) (hc : col<cols+64) :
    head*(rows*cols)+row*cols+col<heads*rows*cols+64*cols+64 := by
  have ha := Nat.mul_le_mul_right (rows*cols) (Nat.succ_le_of_lt hh)
  have hb := Nat.mul_le_mul_right cols (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul,Nat.add_mul,Nat.one_mul] at ha hb
  nlinarith

theorem block_caps (c : Dimensions) (valid : Valid c) (backward : Bool) :
    (c.parameters backward).BK≤64 ∧ (c.parameters backward).BV≤64 := by
  have hk := bk_legal c valid
  have hv := bv_legal c valid backward
  change c.BK≤64 ∧ c.BV backward≤64
  constructor
  · rcases hk with hk|hk|hk <;> omega
  · rcases hv with hv|hv|hv <;> omega

theorem margin (c : Dimensions) (valid : Valid c) (size d o : Nat)
    (hs : size≤largest c) (hd : d≤max c.K c.V) (ho : o<size+64*d+64) :
    o<formedLimit c := by
  have hm : 16≤max c.K c.V := le_trans valid.k_min (le_max_left _ _)
  unfold formedLimit
  omega

theorem query_formed (c : Dimensions) (valid : Valid c) (backward : Bool)
    (head timeBlock keyBlock timeLane keyLane : Nat)
    (hh : head<c.B*c.H) (ht : timeBlock<c.NT)
    (hk : keyBlock<keyBlocks (c.parameters backward))
    (hl : timeLane<64) (he : keyLane<c.BK) :
    head*(c.T*c.K)+(timeBlock*64+timeLane)*c.K+(keyBlock*c.BK+keyLane)<formedLimit c := by
  have hs := shape_valid c valid backward
  have cap := block_caps c valid backward
  have htime := padded_coordinate c.T 64 timeBlock timeLane (by decide) le_rfl ht hl
  have hkey := padded_coordinate c.K c.BK keyBlock keyLane hs.bk_pos cap.1 hk he
  have ho := padded_address (c.B*c.H) c.T c.K head _ _ hh htime hkey
  apply margin c valid (queryWords c) c.K _ (le_trans (le_max_left _ _) (le_max_left _ _)) (le_max_left _ _) ho

theorem value_formed (c : Dimensions) (valid : Valid c) (backward : Bool)
    (head timeBlock valueBlock timeLane valueLane : Nat)
    (hh : head<c.B*c.H) (ht : timeBlock<c.NT)
    (hv : valueBlock<valueBlocks (c.parameters backward))
    (hl : timeLane<64) (hj : valueLane<c.BV backward) :
    head*(c.T*c.V)+(timeBlock*64+timeLane)*c.V+(valueBlock*c.BV backward+valueLane)<formedLimit c := by
  have hs := shape_valid c valid backward
  have cap := block_caps c valid backward
  have htime := padded_coordinate c.T 64 timeBlock timeLane (by decide) le_rfl ht hl
  have hvalue := padded_coordinate c.V (c.BV backward) valueBlock valueLane hs.bv_pos cap.2 hv hj
  have ho := padded_address (c.B*c.H) c.T c.V head _ _ hh htime hvalue
  apply margin c valid (valueWords c) c.V _ (le_trans (le_max_right _ _) (le_max_left _ _)) (le_max_right _ _) ho

theorem history_formed (c : Dimensions) (valid : Valid c) (backward : Bool)
    (head timeBlock keyBlock valueBlock keyLane valueLane : Nat)
    (hh : head<c.B*c.H) (ht : timeBlock<c.NT)
    (hk : keyBlock<keyBlocks (c.parameters backward)) (hv : valueBlock<valueBlocks (c.parameters backward))
    (he : keyLane<c.BK) (hj : valueLane<c.BV backward) :
    head*(c.NT*c.K*c.V)+timeBlock*c.K*c.V+
      (keyBlock*c.BK+keyLane)*c.V+(valueBlock*c.BV backward+valueLane)<formedLimit c := by
  have hs := shape_valid c valid backward
  have cap := block_caps c valid backward
  have hkey := padded_coordinate c.K c.BK keyBlock keyLane hs.bk_pos cap.1 hk he
  have hvalue := padded_coordinate c.V (c.BV backward) valueBlock valueLane hs.bv_pos cap.2 hv hj
  have ho := padded_address (c.B*c.H*c.NT) c.K c.V (head*c.NT+timeBlock) _ _ (packed_lt hh ht) hkey hvalue
  have ha : (head*c.NT+timeBlock)*(c.K*c.V)+(keyBlock*c.BK+keyLane)*c.V+
      (valueBlock*c.BV backward+valueLane)=head*(c.NT*c.K*c.V)+timeBlock*c.K*c.V+
      (keyBlock*c.BK+keyLane)*c.V+(valueBlock*c.BV backward+valueLane) := by ring
  rw [ha] at ho
  apply margin c valid (historyWords c) c.V _ (le_trans (le_max_left _ _) (le_max_right _ _)) (le_max_right _ _) ho

theorem state_formed (c : Dimensions) (valid : Valid c) (backward : Bool)
    (head keyBlock valueBlock keyLane valueLane : Nat)
    (hh : head<c.B*c.H)
    (hk : keyBlock<keyBlocks (c.parameters backward)) (hv : valueBlock<valueBlocks (c.parameters backward))
    (he : keyLane<c.BK) (hj : valueLane<c.BV backward) :
    head*c.K*c.V+(keyBlock*c.BK+keyLane)*c.V+(valueBlock*c.BV backward+valueLane)<formedLimit c := by
  have hcover := (covers c valid backward).1
  have hp : 0<c.NT := by
    change c.T≤c.NT*64 at hcover
    have := valid.t_pos
    omega
  have ho := history_formed c valid backward head 0 keyBlock valueBlock keyLane valueLane hh hp hk hv he hj
  have hh' : head*c.K*c.V≤head*(c.NT*c.K*c.V) := by
    have hm := Nat.mul_le_mul_left (head*c.K*c.V) (Nat.succ_le_of_lt hp)
    nlinarith
  omega

theorem partial_formed (c : Dimensions) (valid : Valid c)
    (kb head timeBlock valueBlock timeLane valueLane : Nat)
    (hkb : kb<keyBlocks (c.parameters Bool.true)) (hh : head<c.B*c.H) (ht : timeBlock<c.NT)
    (hv : valueBlock<valueBlocks (c.parameters Bool.true)) (hl : timeLane<64) (hj : valueLane<c.BV Bool.true) :
    (kb*(c.B*c.H)+head)*(c.T*c.V)+(timeBlock*64+timeLane)*c.V+
      (valueBlock*c.BV Bool.true+valueLane)<formedLimit c := by
  have hs := shape_valid c valid Bool.true
  have cap := block_caps c valid Bool.true
  have htime := padded_coordinate c.T 64 timeBlock timeLane (by decide) le_rfl ht hl
  have hvalue := padded_coordinate c.V (c.BV Bool.true) valueBlock valueLane hs.bv_pos cap.2 hv hj
  have ho := padded_address (keyBlocks (c.parameters Bool.true)*(c.B*c.H)) c.T c.V
    (kb*(c.B*c.H)+head) _ _ (packed_lt hkb hh) htime hvalue
  have hsize : keyBlocks (c.parameters Bool.true)*(c.B*c.H)*c.T*c.V=partialWords c := by unfold partialWords; ring
  rw [hsize] at ho
  apply margin c valid (partialWords c) c.V _ (le_trans (le_max_right _ _) (le_max_right _ _)) (le_max_right _ _) ho

theorem machine_safe (c : Dimensions) (guard : MachineGuard c) (offset : Nat) (h : offset<formedLimit c) :
    offset<2^31 := lt_trans h guard

structure PointerABI (c : Dimensions) (buffers : List Physical.Buffer) : Prop where
  room : ∀ b∈buffers, b.base+formedLimit c<2^61

theorem pointer_safe (c : Dimensions) (buffers : List Physical.Buffer) (abi : PointerABI c buffers)
    (b : Physical.Buffer) (hb : b∈buffers) (offset : Nat) (ho : offset<formedLimit c) :
    4*(b.base+offset)<2^63 := by
  have h := abi.room b hb
  norm_num at h ⊢
  omega

#axiomsClean padded_coordinate
#axiomsClean query_formed
#axiomsClean value_formed
#axiomsClean history_formed
#axiomsClean state_formed
#axiomsClean partial_formed
#axiomsClean machine_safe
#axiomsClean pointer_safe
#specNonCircular PointerABI avoiding [gridMemory,executeFour]
end VeriTile.Bench.Repairs.ChunkLinearAttn.Host
