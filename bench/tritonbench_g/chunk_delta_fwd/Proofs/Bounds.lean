import bench.tritonbench_g.chunk_delta_fwd.Proofs.HostMetadata
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Host
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
set_option maxHeartbeats 6000000

def weightSpan (c : Dimensions) :=
  (c.B-1)*c.wBatch+(c.H-1)*c.wHead+(c.T-1)*c.wTime+(c.K-1)*c.wDim+1
/-- Concrete descriptor/index inequalities. They contain no execution or
numerical result premise and are witnessed below. Block-pointer products are
signed 64-bit; source base and loop arithmetic use signed 32-bit. -/
structure AddressContract (c : Dimensions) : Prop where
  time : c.T+c.BT+256 < 2^31
  key : c.B*c.H*c.T*c.K < 2^31
  value : c.B*c.H*c.T*c.V < 2^31
  state : c.B*c.H*c.K*c.V < 2^31
  history : c.B*c.H*c.NT*c.K*c.V+256*c.V < 2^31
  weight : weightSpan c < 2^31
  wBatch : c.wBatch < 2^31
  wHead : c.wHead < 2^31
  wTime : c.wTime < 2^31
  wDim : c.wDim < 2^31

theorem ceil_start (n width i : Nat) (hw : 0 < width) (hi : i < ceilDiv n width) : i*width < n := by
  have h1 := Nat.mul_le_mul_right width hi
  have h2 := Nat.div_mul_le_self (n+width-1) width
  change (i+1)*width ≤ (ceilDiv n width)*width at h1
  simp only [Nat.add_mul,Nat.one_mul,ceilDiv] at h1
  omega

theorem time_bound (c : Dimensions) (h : Valid c) (bounds : AddressContract c)
    (outer inner lane : Nat) (ho : outer < c.NT)
    (hi : inner < innerCount c.parameters) (hl : lane < c.BC) :
    time c.parameters outer inner lane < 2^31 := by
  have hs := shape_valid c h
  have houter := ceil_start c.T c.BT outer hs.bt_pos ho
  have hinner := ceil_start c.BT c.BC inner hs.bc_pos hi
  have hbv := bv_legal c
  have hbc : c.BC ≤ 64 := by
    have hh := Nat.min_le_right c.BT c.BV
    change c.BC ≤ c.BV at hh
    rcases hbv with hbv|hbv|hbv <;> omega
  have hlim := bounds.time
  change outer*c.BT+inner*c.BC+lane < 2^31
  omega

theorem value_lane_bound (c : Dimensions) (h : Valid c) (bounds : AddressContract c)
    (pid : PID) (hp : Geometry.ValidPID c.parameters pid) (lane : Nat) (hl : lane < c.BV) :
    column c.parameters pid lane < 2^31 := by
  have hs := shape_valid c h
  have hv := ceil_start c.V c.BV pid.1 hs.bv_pos hp.1
  have hsz := bounds.history
  have hB := h.b_pos
  have hH := h.h_pos
  have hK := h.k_min
  have hn : 0 < c.NT := by
    have ho := VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry.owner_lt hs.bt_pos h.t_pos
    simpa only [Nat.zero_div] using ho
  have hprod : c.V ≤ c.B*c.H*c.NT*c.K*c.V := by
    have hp' : 1 ≤ c.B*c.H*c.NT*c.K := Nat.succ_le_of_lt
      (Nat.mul_pos (Nat.mul_pos (Nat.mul_pos hB hH) hn) (by omega))
    simpa only [Nat.one_mul] using Nat.mul_le_mul_right c.V hp'
  have hbv := bv_legal c
  change pid.1*c.BV+lane < 2^31
  rcases hbv with hbv|hbv|hbv <;> omega

theorem weight_address_bound (c : Dimensions) (h : Valid c) (head t e : Nat)
    (hh : head < c.B*c.H) (ht : t < c.T) (he : e < c.K) :
    head/c.H*c.wBatch+head%c.H*c.wHead+t*c.wTime+e*c.wDim < weightSpan c := by
  have hb : head/c.H < c.B := (Nat.div_lt_iff_lt_mul h.h_pos).mpr hh
  have hd : head%c.H < c.H := Nat.mod_lt _ h.h_pos
  have hb' := Nat.mul_le_mul_right c.wBatch (show head/c.H ≤ c.B-1 by omega)
  have hd' := Nat.mul_le_mul_right c.wHead (show head%c.H ≤ c.H-1 by omega)
  have ht' := Nat.mul_le_mul_right c.wTime (show t ≤ c.T-1 by omega)
  have he' := Nat.mul_le_mul_right c.wDim (show e ≤ c.K-1 by omega)
  unfold weightSpan
  omega

theorem block_pointer_i64 (base row col rowStride colStride : Nat)
    (hb : base < 2^31) (hr : row < 2^31) (hc : col < 2^31)
    (hrs : rowStride < 2^31) (hcs : colStride < 2^31) :
    base+row*rowStride+col*colStride < 2^63 := by
  have hp := Nat.mul_le_mul (show row ≤ 2^31-1 by omega) (show rowStride ≤ 2^31-1 by omega)
  have hq := Nat.mul_le_mul (show col ≤ 2^31-1 by omega) (show colStride ≤ 2^31-1 by omega)
  norm_num at hp hq ⊢
  omega

def witness : Dimensions := ⟨1,1,17,16,5,16,272,272,16,1,Bool.false,Bool.false⟩
theorem witness_valid : Valid witness := by
  refine ⟨by decide,by decide,by decide,by decide,by decide,by decide,by decide,?_,by decide,by decide,by decide,by decide⟩
  exact ⟨⟨4,by decide⟩,by decide⟩
theorem witness_bounds : AddressContract witness := by constructor <;> decide

#axiomsClean time_bound
#axiomsClean value_lane_bound
#axiomsClean weight_address_bound
#axiomsClean block_pointer_i64
#axiomsClean witness_bounds
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Host
