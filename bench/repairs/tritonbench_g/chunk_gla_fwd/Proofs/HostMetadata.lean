import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPStages
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Host
open VeriTile Triton
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

def smear (n shift : Nat) : Nat := n ||| (n >>> shift)
/-- The installed Triton helper uses exactly six shifts and maps zero to
zero. This is not an idealized unbounded next-power function. -/
def nextPowerOfTwo (n : Nat) : Nat :=
  if n=0 then 0 else smear (smear (smear (smear (smear (smear (n-1) 1) 2) 4) 8) 16) 32+1
theorem smear_ge (n shift : Nat) : n ≤ smear n shift := Nat.left_le_or
theorem nextPowerOfTwo_ge (n : Nat) : n ≤ nextPowerOfTwo n := by
  by_cases hn : n=0
  · simp [nextPowerOfTwo,hn]
  · have h := (smear_ge (n-1) 1).trans ((smear_ge _ 2).trans ((smear_ge _ 4).trans
      ((smear_ge _ 8).trans ((smear_ge _ 16).trans (smear_ge _ 32)))))
    simp only [nextPowerOfTwo,hn,if_false]
    omega
theorem nextPowerOfTwo_pos (n : Nat) (hn : 0<n) : 0<nextPowerOfTwo n :=
  lt_of_lt_of_le hn (nextPowerOfTwo_ge n)

structure Dimensions where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  k_pos : 0<K
  bt_power : ∃ e, BT=2^(e+4)

def Dimensions.heads (c : Dimensions) := c.B*c.H
def Dimensions.NC (c : Dimensions) := Geometry.ceilDiv c.BT 16
def Dimensions.NT (c : Dimensions) := Geometry.ceilDiv c.T c.BT
def Dimensions.NK (c : Dimensions) := Geometry.ceilDiv c.K 128
def Dimensions.interBK (c : Dimensions) := min 64 (nextPowerOfTwo (max 16 c.K))
def Dimensions.intraBK (c : Dimensions) := nextPowerOfTwo c.K
def Dimensions.outputBK (c : Dimensions) := min 32 (nextPowerOfTwo (max 16 c.K))
def Dimensions.BV (c : Dimensions) := min 32 (nextPowerOfTwo (max 1 c.V))

theorem Dimensions.bt_divisible (c : Dimensions) : 16∣c.BT := by
  obtain ⟨e,he⟩ := c.bt_power
  rw [he]
  refine ⟨2^e,?_⟩
  simp [pow_add,Nat.mul_comm]
theorem Dimensions.bt_ge (c : Dimensions) : 16 ≤ c.BT := by
  obtain ⟨e,he⟩ := c.bt_power
  rw [he,pow_add]
  have hp : 0<2^e := Nat.pow_pos (by decide)
  norm_num
  omega
theorem Dimensions.bt_pos (c : Dimensions) : 0<c.BT := lt_of_lt_of_le (by decide) c.bt_ge
theorem Dimensions.nc_eq (c : Dimensions) : c.NC=c.BT/16 := by
  obtain ⟨n,hn⟩ := c.bt_divisible
  simp only [Dimensions.NC,Geometry.ceilDiv,hn]
  omega
theorem Dimensions.bt_eq (c : Dimensions) : c.BT=c.NC*16 := by
  rw [c.nc_eq,Nat.div_mul_cancel c.bt_divisible]
theorem Dimensions.nc_pos (c : Dimensions) : 0<c.NC := by
  have hb := c.bt_pos
  rw [c.bt_eq] at hb
  omega
theorem Dimensions.interBK_pos (c : Dimensions) : 0<c.interBK := by
  apply lt_min (by decide)
  exact nextPowerOfTwo_pos _ (lt_of_lt_of_le (by decide : 0<16) (Nat.le_max_left _ _))
theorem Dimensions.bv_pos (c : Dimensions) : 0<c.BV := by
  apply lt_min (by decide)
  exact nextPowerOfTwo_pos _ (lt_of_lt_of_le (by decide : 0<1) (Nat.le_max_left _ _))

def Dimensions.producer (c : Dimensions) : ProducerGeometry.Shape :=
  ⟨c.heads,c.T,c.K,c.BT,c.NC,c.interBK,c.nc_pos,c.bt_eq,c.interBK_pos⟩
def Dimensions.split (c : Dimensions) : ProducerGeometry.Shape :=
  {c.producer with BK:=128,bk_pos:=by decide}
def Dimensions.consumer (c : Dimensions) : OutputGrid.Shape :=
  ⟨c.heads,c.T,c.K,c.V,c.BT,c.outputBK,c.BV,c.bt_pos,c.bv_pos⟩

theorem source_inter_grid (c : Dimensions) : (FPStages.Stage.inter c.producer 0).grid=
    ⟨[c.NT,c.NC*c.NC,c.B*c.H]⟩ := rfl
theorem source_intra_grid (c : Dimensions) : (FPStages.Stage.intra c.producer c.intraBK 0).grid=
    ⟨[c.NT,c.NC,c.B*c.H]⟩ := rfl
theorem source_split_grid (c : Dimensions) : (FPStages.Stage.split c.split 0).grid=
    ⟨[c.NK,c.NT*c.NC,c.B*c.H]⟩ := rfl
theorem source_merge_grid (c : Dimensions) : (FPStages.Stage.merge c.split).grid=
    ⟨[c.NT,c.NC,c.B*c.H]⟩ := rfl
theorem source_output_grid (c : Dimensions) : (FPStages.Stage.output c.consumer 0).grid=
    ⟨[Geometry.ceilDiv c.V c.BV,c.NT,c.B*c.H]⟩ := rfl

theorem intra_covers_K (c : Dimensions) : c.K ≤ c.intraBK := nextPowerOfTwo_ge _
theorem allocation_attention (c : Dimensions) : c.B*c.H*c.T*c.BT=c.heads*c.T*c.BT := rfl
theorem allocation_scratch (c : Dimensions) : c.NK*c.B*c.H*c.T*16=c.NK*c.heads*c.T*16 := by
  simp [Dimensions.heads,Nat.mul_assoc]
theorem allocation_history (c : Dimensions) : c.B*c.H*(c.NT*c.K)*c.V=c.heads*(c.NT*c.K*c.V) := by
  simp [Dimensions.heads,Nat.mul_assoc]

#axiomsClean nextPowerOfTwo_ge
#axiomsClean source_split_grid
#axiomsClean intra_covers_K
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Host
