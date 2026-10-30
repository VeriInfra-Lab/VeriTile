import bench.repairs.tritonbench_g.chunk_linear_attn.HostPhysical
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Host
open VeriTile Triton
set_option maxHeartbeats 12000000

/-- Actual make_block_ptr affine address, before its boundary mask. -/
def formed (base sr sc ro co row col : Nat) := base+(ro+row)*sr+(co+col)*sc
 theorem chunk_in_range (p : Parameters) (rev : Bool) (n : Nat) (hn : n<p.NT) : chunk p rev n<p.NT := by
  cases rev <;> simp only [chunk,Bool.false_eq_true,if_false,if_true] <;> omega

 theorem state_history_address (c : Dimensions) (valid : Valid c) (backward : Bool)
    (pid : PID) (hp : ValidPID .forwardState (c.parameters backward) pid)
    (n : Nat) (hn : n<c.NT) (i : TileIndex [c.BK,c.BV backward]) :
    hOffset (c.parameters backward) pid n i<formedLimit c := by
  exact history_formed c valid backward pid.z n pid.x pid.y i.1.val i.2.1.val hp.2.2 hn hp.1 hp.2.1 i.1.isLt i.2.1.isLt
 theorem state_final_address (c : Dimensions) (valid : Valid c) (backward : Bool)
    (pid : PID) (hp : ValidPID .forwardState (c.parameters backward) pid)
    (i : TileIndex [c.BK,c.BV backward]) :
    fOffset (c.parameters backward) pid i<formedLimit c := by
  exact state_formed c valid backward pid.z pid.x pid.y i.1.val i.2.1.val hp.2.2 hp.1 hp.2.1 i.1.isLt i.2.1.isLt
 theorem state_left_address (c : Dimensions) (valid : Valid c) (backward rev : Bool)
    (pid : PID) (hp : ValidPID .forwardState (c.parameters backward) pid)
    (n : Nat) (hn : n<c.NT) (i : TileIndex [c.BK,64]) :
    formed (pid.z*(c.T*c.K)) 1 c.K (pid.x*c.BK) (chunk (c.parameters backward) rev n*64) i.1.val i.2.1.val<formedLimit c := by
  have h := query_formed c valid backward pid.z (chunk (c.parameters backward) rev n) pid.x i.2.1.val i.1.val
    hp.2.2 (chunk_in_range _ rev n hn) hp.1 i.2.1.isLt i.1.isLt
  simpa only [formed,Nat.mul_one,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using h
 theorem state_right_address (c : Dimensions) (valid : Valid c) (backward rev : Bool)
    (pid : PID) (hp : ValidPID .forwardState (c.parameters backward) pid)
    (n : Nat) (hn : n<c.NT) (i : TileIndex [64,c.BV backward]) :
    formed (pid.z*(c.T*c.V)) c.V 1 (chunk (c.parameters backward) rev n*64) (pid.y*c.BV backward) i.1.val i.2.1.val<formedLimit c := by
  simpa only [formed,Nat.mul_one] using value_formed c valid backward pid.z (chunk (c.parameters backward) rev n) pid.y i.1.val i.2.1.val
    hp.2.2 (chunk_in_range _ rev n hn) hp.2.1 i.1.isLt i.2.1.isLt
 theorem output_store_address (c : Dimensions) (valid : Valid c) (pid : PID)
    (hp : ValidPID .forwardOutput (c.parameters Bool.false) pid) (i : TileIndex [64,c.BV Bool.false]) :
    outputOffset (c.parameters Bool.false) pid i<formedLimit c := by
  exact value_formed c valid Bool.false pid.z pid.y pid.x i.1.val i.2.1.val hp.2.2 hp.2.1 hp.1 i.1.isLt i.2.1.isLt
 theorem output_query_address (c : Dimensions) (valid : Valid c) (pid : PID)
    (hp : ValidPID .forwardOutput (c.parameters Bool.false) pid) (kb : Nat)
    (hk : kb<keyBlocks (c.parameters Bool.false)) (i : TileIndex [64,c.BK]) :
    formed (pid.z*(c.T*c.K)) c.K 1 (pid.y*64) (kb*c.BK) i.1.val i.2.1.val<formedLimit c := by
  simpa only [formed,Nat.mul_one] using query_formed c valid Bool.false pid.z pid.y kb i.1.val i.2.1.val hp.2.2 hp.2.1 hk i.1.isLt i.2.1.isLt
 theorem output_history_address (c : Dimensions) (valid : Valid c) (pid : PID)
    (hp : ValidPID .forwardOutput (c.parameters Bool.false) pid) (kb : Nat)
    (hk : kb<keyBlocks (c.parameters Bool.false)) (i : TileIndex [c.BK,c.BV Bool.false]) :
    formed (pid.z*(c.NT*c.K*c.V)+pid.y*c.K*c.V) c.V 1 (kb*c.BK) (pid.x*c.BV Bool.false) i.1.val i.2.1.val<formedLimit c := by
  simpa only [formed,Nat.mul_one] using history_formed c valid Bool.false pid.z pid.y kb pid.x i.1.val i.2.1.val hp.2.2 hp.2.1 hk hp.1 i.1.isLt i.2.1.isLt
 theorem gradient_key_address (c : Dimensions) (valid : Valid c) (pid : PID)
    (hp : ValidPID .backwardGradient (c.parameters Bool.true) pid) (i : TileIndex [64,c.BK]) :
    gradKeyOffset (c.parameters Bool.true) pid i<formedLimit c := by
  exact query_formed c valid Bool.true pid.z pid.y pid.x i.1.val i.2.1.val hp.2.2 hp.2.1 hp.1 i.1.isLt i.2.1.isLt
 theorem gradient_value_address (c : Dimensions) (valid : Valid c) (pid : PID)
    (hp : ValidPID .backwardGradient (c.parameters Bool.true) pid) (vb : Nat)
    (hv : vb<valueBlocks (c.parameters Bool.true)) (i : TileIndex [64,c.BV Bool.true]) :
    gradValueOffset (c.parameters Bool.true) pid (c.B*c.H) vb i<formedLimit c := by
  exact partial_formed c valid pid.x pid.z pid.y vb i.1.val i.2.1.val hp.1 hp.2.2 hp.2.1 hv i.1.isLt i.2.1.isLt
 theorem gradient_history_address (c : Dimensions) (valid : Valid c) (pid : PID)
    (hp : ValidPID .backwardGradient (c.parameters Bool.true) pid) (vb : Nat)
    (hv : vb<valueBlocks (c.parameters Bool.true)) (i : TileIndex [c.BK,c.BV Bool.true]) :
    formed (pid.z*(c.NT*c.K*c.V)) c.V 1 (pid.y*c.K+pid.x*c.BK) (vb*c.BV Bool.true) i.1.val i.2.1.val<formedLimit c := by
  have h := history_formed c valid Bool.true pid.z pid.y pid.x vb i.1.val i.2.1.val hp.2.2 hp.2.1 hp.1 hv i.1.isLt i.2.1.isLt
  convert h using 1 <;> unfold formed <;> ring

/-- Both orientations of a transposed block pointer form exactly the same
address. This covers key/query and history/history-gradient transposes. -/
theorem formed_transpose (base sr sc ro co row col : Nat) :
    formed base sr sc ro co row col=formed base sc sr co ro col row := by unfold formed; ring

/-- Applying the source guard and actual allocation bases to any of the proven
formed offsets yields both signed32 arithmetic and the four-byte pointer bound. -/
theorem actual_pointer_safe (c : Invocation) (cc : Contract c) (a : Allocations) (pa : PhysicalAllocation c a)
    (b : Physical.Buffer) (hb : b∈outputBuffers a++inputBuffers c a) (o : Nat) (ho : o<formedLimit c.dims) :
    o<2^31 ∧ 4*(b.base+o)<2^63 :=
  ⟨machine_safe c.dims cc.machine o ho,pointer_safe c.dims _ pa.pointerRoom b hb o ho⟩
 theorem affine_component_safe (c : Dimensions) (guard : MachineGuard c)
    (base row column total : Nat) (he : total=base+row+column) (ht : total<formedLimit c) :
    base<2^31 ∧ row<2^31 ∧ column<2^31 := by
  have := machine_safe c guard total ht
  omega

#axiomsClean state_history_address
#axiomsClean state_final_address
#axiomsClean state_left_address
#axiomsClean state_right_address
#axiomsClean output_store_address
#axiomsClean output_query_address
#axiomsClean output_history_address
#axiomsClean gradient_key_address
#axiomsClean gradient_value_address
#axiomsClean gradient_history_address
#axiomsClean actual_pointer_safe
end VeriTile.Bench.Repairs.ChunkLinearAttn.Host
