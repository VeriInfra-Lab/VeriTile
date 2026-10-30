import bench.repairs.tritonbench_g.chunk_delta_fwd.GridOutputs
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.HostMetadata
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Host
open VeriTile Triton
open VeriTile.Bench.Repairs.ChunkGlaFwd.Host (nextPowerOfTwo nextPowerOfTwo_ge)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (ceilDiv)
set_option maxHeartbeats 12000000

structure Dimensions where
  (B H T K V BT : Nat)
  (wBatch wHead wTime wDim : Nat)
  (seed final : Bool)
def Dimensions.BK (c : Dimensions) := nextPowerOfTwo c.K
def Dimensions.BV (c : Dimensions) := if c.BK > 128 then 16 else if c.BK > 64 then 32 else 64
def Dimensions.BC (c : Dimensions) := min c.BT c.BV
def Dimensions.NT (c : Dimensions) := ceilDiv c.T c.BT
def Dimensions.parameters (c : Dimensions) : Parameters :=
  ⟨c.B,c.H,c.T,c.K,c.V,c.BT,c.BC,c.BK,c.BV,c.NT,c.wBatch,c.wHead,c.wTime,c.wDim,c.seed,c.final⟩

structure Valid (c : Dimensions) : Prop where
  b_pos : 0 < c.B
  h_pos : 0 < c.H
  t_pos : 0 < c.T
  k_min : 16 ≤ c.K
  k_max : c.K ≤ 256
  v_pos : 0 < c.V
  bt_min : 16 ≤ c.BT
  bc_power : ∃ e : Fin 7, c.BC=2^e.val
  wBatch_pos : 0 < c.wBatch
  wHead_pos : 0 < c.wHead
  wTime_pos : 0 < c.wTime
  wDim_pos : 0 < c.wDim

/-- Exact installed next-power helper on the selected K domain. -/
theorem bk_table (K : Nat) (hlo : 16 ≤ K) (hhi : K ≤ 256) :
    nextPowerOfTwo K=if K≤16 then 16 else if K≤32 then 32 else if K≤64 then 64 else
      if K≤128 then 128 else 256 := by
  interval_cases K <;> decide

theorem bk_legal (c : Dimensions) (h : Valid c) :
    c.BK∈([16,32,64,128,256] : List Nat) ∧ c.K ≤ c.BK ∧ c.BK ≤ 256 := by
  have ht := bk_table c.K h.k_min h.k_max
  have hk := nextPowerOfTwo_ge c.K
  simp only [Dimensions.BK] at *
  rw [ht] at *
  split_ifs at * <;> simp_all

theorem bv_legal (c : Dimensions) : c.BV=16 ∨ c.BV=32 ∨ c.BV=64 := by
  unfold Dimensions.BV
  split_ifs <;> simp
theorem shape_valid (c : Dimensions) (h : Valid c) : Geometry.ShapeValid c.parameters := by
  have hv := bv_legal c
  have hb := h.bt_min
  refine ⟨by change 0 < c.BT; omega,?_,?_,nextPowerOfTwo_ge c.K,rfl⟩
  · change 0 < min c.BT c.BV
    exact lt_min (by omega) (by rcases hv with hv|hv|hv <;> omega)
  · change 0 < c.BV
    rcases hv with hv|hv|hv <;> omega

theorem nk_one (c : Dimensions) (h : Valid c) : ceilDiv c.K c.BK=1 := by
  have hmin := h.k_min
  have hk := (bk_legal c h).2.1
  have hp : 0 < c.BK := lt_of_lt_of_le (by omega : 0 < c.K) hk
  unfold ceilDiv
  apply Nat.div_eq_of_lt_le
  · omega
  · omega

def launchGrid (c : Dimensions) : Nat × Nat × Nat :=
  (ceilDiv c.K c.BK,ceilDiv c.V c.BV,c.B*c.H)
theorem launch_grid (c : Dimensions) (h : Valid c) :
    launchGrid c=(1,ceilDiv c.V c.BV,c.B*c.H) := by simp [launchGrid,nk_one c h]
def historyShape (c : Dimensions) := [c.B,c.H,c.NT*c.K,c.V]
def correctedShape (c : Dimensions) := [c.B,c.H,c.T,c.V]
def stateShape (c : Dimensions) := [c.B,c.H,c.K,c.V]
theorem history_numel (c : Dimensions) : (historyShape c).prod=c.B*c.H*c.NT*c.K*c.V := by
  simp [historyShape]
  ring

#axiomsClean bk_table
#axiomsClean shape_valid
#axiomsClean launch_grid
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Host
