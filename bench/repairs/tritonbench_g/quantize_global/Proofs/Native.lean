import bench.repairs.tritonbench_g.quantize_global.Proofs.Indices
namespace QuantizeGlobalRepair
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000
def i64 (n : Nat) : BitVec 64 := BitVec.ofInt 64 (n : Int)
def nativeOffset (base pid block lane : Nat) : BitVec 64 :=
  (i64 pid * i64 block + i64 lane) + i64 base

theorem native_offset_word (base pid block lane : Nat) :
    nativeOffset base pid block lane = i64 (base + pid*block + lane) := by
  simp only [nativeOffset,i64,← BitVec.ofInt_mul,← BitVec.ofInt_add]
  congr 1
  push_cast
  omega

theorem i64_exact (n : Nat) (h : n < 2^63) : (i64 n).toInt = (n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · omega
  · exact_mod_cast h

theorem native_lane_exact (n k pid lane : Nat) (c : Config) (hn : n < 2^63)
    (hk : k < segmentCount n) (hp : pid < programs n k c) (hl : lane < c.block) :
    (nativeOffset (k*span) pid c.block lane).toInt = (k*span+pid*c.block+lane : Nat) := by
  rw [native_offset_word]
  exact i64_exact _ (offsets_i64 n k pid lane c hn hk hp hl)

/-- Native PID is i32 before its explicit widening. The segmented grid keeps
that producer well inside signed32, including both autotune choices. -/
theorem pid_widening (n k pid : Nat) (c : Config) (hp : pid < programs n k c) :
    BitVec.ofInt 64 (BitVec.ofInt 32 (pid : Int)).toInt = i64 pid := by
  have hbound := grid_bound n k c
  have hdecode : (BitVec.ofInt 32 (pid : Int)).toInt = (pid : Int) := by
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega
  rw [hdecode]
  rfl

theorem lane_widening (c : Config) (lane : Nat) (hl : lane < c.block) :
    BitVec.ofInt 64 (BitVec.ofInt 32 (lane : Int)).toInt = i64 lane := by
  have hdecode : (BitVec.ofInt 32 (lane : Int)).toInt = (lane : Int) := by
    apply BitVec.toInt_ofInt_eq_self (by decide)
    · omega
    · cases c <;> simp only [QuantizeGlobalFinite.Config.block] at hl <;> norm_num <;> omega
  rw [hdecode]
  rfl

theorem intermediate_bounds (n k pid lane : Nat) (c : Config) (hn : n < 2^63)
    (hk : k < segmentCount n) (hp : pid < programs n k c) (hl : lane < c.block) :
    k*span < 2^63 ∧ pid*c.block < 2^63 ∧ pid*c.block+lane < 2^63 ∧ lane < 2^63 := by
  have h := offsets_i64 n k pid lane c hn hk hp hl
  omega

/-- The ordinary successful allocation supplies a byte window, not a custom
native-safe input restriction. A safe in-window logical offset has an exact
unsigned64 pointer, for both FP32 loads (width4) and int8 stores (width1). -/
theorem pointer_exact (base width n index : Nat) (hw : 0 < width)
    (window : base + width*n ≤ 2^64) (hi : index < n) :
    (BitVec.ofNat 64 base + BitVec.ofNat 64 width * BitVec.ofNat 64 index).toNat =
      base + width*index := by
  have hlt : base + width*index < 2^64 := by
    have hmul := Nat.mul_lt_mul_of_pos_left hi hw
    omega
  rw [← BitVec.ofNat_mul,← BitVec.ofNat_add]
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt hlt]

def NativeLaunchSafe (n : Nat) (trials : Nat → List Config) : Prop :=
  ∀ k, k < segmentCount n → ∀ c ∈ trials k, programs n k c ≤ 1048576 ∧
    ∀ pid, pid < programs n k c → ∀ lane, lane < c.block →
      (nativeOffset (k*span) pid c.block lane).toInt = (k*span+pid*c.block+lane : Nat) ∧
      k*span+pid*c.block+lane < 2^63

theorem native_launch_safe (n : Nat) (trials : Nat → List Config) (hn : n < 2^63) :
    NativeLaunchSafe n trials := by
  intro k hk c _
  refine ⟨grid_bound n k c,?_⟩
  intro pid hp lane hl
  exact ⟨native_lane_exact n k pid lane c hn hk hp hl, offsets_i64 n k pid lane c hn hk hp hl⟩

#axiomsClean native_lane_exact
#axiomsClean pid_widening
#axiomsClean lane_widening
#axiomsClean intermediate_bounds
#axiomsClean pointer_exact
end QuantizeGlobalRepair
