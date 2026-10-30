import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.Contract

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Native
set_option maxHeartbeats 3000000

abbrev blocks := launchGrid

theorem lane_bounds (n b width col pid lane : Nat)
    (valid : LaunchContract n width b col) (hb : 0<b ∧ b≤1024)
    (hp : pid<blocks n b) (hl : lane<b) :
    pid<2^31 ∧ lane<2^31 ∧ pid*b+lane≤blocks n b*b-1 := by
  have nonempty : 0<n := by
    by_contra empty
    have hn:n=0 := by omega
    simp only [blocks,launchGrid,hn,Nat.zero_add] at hp
    have divzero:(b-1)/b=0 := Nat.div_eq_of_lt (by omega)
    omega
  have grid : blocks n b<2^31 := (valid nonempty).1
  have before:=Nat.mul_le_mul_right b (Nat.succ_le_of_lt hp)
  rw [Nat.succ_mul] at before
  norm_num at grid ⊢
  exact ⟨by omega,by omega,by omega⟩

theorem signed64 (n : Nat) (bound : n<2^63) : (BitVec.ofNat 64 n).toInt=(n : Int) :=
  BitVec.toInt_ofInt_eq_self (w:=64) (n:=(n : Int)) (by decide) (by omega) (by norm_num;omega)

theorem widened (n : Nat) (bound : n<2^31) :
    (BitVec.ofNat 32 n).signExtend 64=BitVec.ofNat 64 n := by
  apply BitVec.eq_of_toInt_eq
  rw [BitVec.toInt_signExtend_of_le (by decide)]
  have h32 : (BitVec.ofNat 32 n).toInt=(n : Int) :=
    BitVec.toInt_ofInt_eq_self (w:=32) (n:=(n : Int)) (by decide) (by omega) (by norm_num;omega)
  rw [h32]
  symm
  exact signed64 n (by norm_num at bound ⊢;omega)

/-- The actual pid and arange int32 values are widened before multiplication. -/
def offset (b width col pid lane channel : Nat) : BitVec 64 :=
  (BitVec.ofNat 32 lane).signExtend 64 * BitVec.ofNat 64 width +
  BitVec.ofNat 64 (b*width) * (BitVec.ofNat 32 pid).signExtend 64 +
  BitVec.ofNat 64 col + BitVec.ofNat 64 channel

def mask (n b width col pid lane channel : Nat) : Bool :=
  (offset b width col pid lane channel).slt (BitVec.ofNat 64 (n*width))

theorem offset_word (b width col pid lane channel : Nat) (hp : pid<2^31) (hl : lane<2^31) :
    offset b width col pid lane channel=BitVec.ofNat 64 ((pid*b+lane)*width+col+channel) := by
  simp only [offset,widened pid hp,widened lane hl,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  congr 1
  ring

theorem mask_exact (n b width col pid lane channel : Nat)
    (valid : LaunchContract n width b col) (hb : 0<b ∧ b≤1024)
    (hc : col+channel<width) (hchannel : channel≤10)
    (hp : pid<blocks n b) (hl : lane<b) :
    mask n b width col pid lane channel=decide (pid*b+lane<n) := by
  obtain ⟨ps,ls,rs⟩:=lane_bounds n b width col pid lane valid hb hp hl
  have nonempty:0<n := by
    by_contra empty
    have hn:n=0 := by omega
    simp only [blocks,launchGrid,hn,Nat.zero_add] at hp
    have divzero:(b-1)/b=0 := Nat.div_eq_of_lt (by omega)
    omega
  have ib : (pid*b+lane)*width+col+channel<2^63 := by
    have mul:=Nat.mul_le_mul_right width rs
    have bound : (blocks n b*b-1)*width+col+10<2^63 := (valid nonempty).2.2.2.2
    omega
  have nb : n*width<2^63 := (valid nonempty).2.2.1
  rw [mask,offset_word _ _ _ _ _ _ ps ls,BitVec.slt_eq_decide,signed64 _ ib,signed64 _ nb]
  simp only [Int.ofNat_lt]
  congr 1
  apply propext
  constructor
  · intro active
    by_contra inactive
    have h:=Nat.mul_le_mul_right width (Nat.le_of_not_gt inactive)
    omega
  · intro active
    have h:=Nat.mul_le_mul_right width (Nat.succ_le_of_lt active)
    rw [Nat.succ_mul] at h
    omega

theorem coordinate_mask (n b width col pid lane channel : Nat)
    (valid : LaunchContract n width b col) (hb : 0<b ∧ b≤1024) (hc : channel<3)
    (hp : pid<blocks n b) (hl : lane<b) :
    mask n b 3 0 pid lane channel=decide (pid*b+lane<n) := by
  obtain ⟨ps,ls,rs⟩:=lane_bounds n b width col pid lane valid hb hp hl
  have nonempty:0<n := by
    by_contra empty
    have hn:n=0 := by omega
    simp only [blocks,launchGrid,hn,Nat.zero_add] at hp
    have divzero:(b-1)/b=0 := Nat.div_eq_of_lt (by omega)
    omega
  have bound : (blocks n b*b-1)*3+2<2^63 := (valid nonempty).2.2.2.1
  have ib : (pid*b+lane)*3+0+channel<2^63 := by omega
  have nb : n*3<2^63 := (valid nonempty).2.1
  rw [mask,offset_word _ _ _ _ _ _ ps ls,BitVec.slt_eq_decide,signed64 _ ib,signed64 _ nb]
  simp only [Int.ofNat_lt,Nat.add_zero]
  congr 1
  apply propext
  omega

/-- Full modular upstream address; inactive products may wrap harmlessly. -/
def pointer (base b sx sy col pid lane channel : Nat) : Nat :=
  (BitVec.ofNat 64 base +
    (((BitVec.ofNat 32 lane).signExtend 64 + BitVec.ofNat 64 b * (BitVec.ofNat 32 pid).signExtend 64) *
      BitVec.ofNat 64 sx + BitVec.ofNat 64 col * BitVec.ofNat 64 sy +
      BitVec.ofNat 64 channel * BitVec.ofNat 64 sy) * BitVec.ofNat 64 4).toNat

theorem pointer_exact (base b sx sy col pid lane channel : Nat)
    (hp : pid<2^31) (hl : lane<2^31)
    (fits : base+4*((pid*b+lane)*sx+(col+channel)*sy)<2^64) :
    pointer base b sx sy col pid lane channel=
      base+4*((pid*b+lane)*sx+(col+channel)*sy) := by
  simp only [pointer,widened pid hp,widened lane hl,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  have eq : base+((lane+b*pid)*sx+col*sy+channel*sy)*4=
      base+4*((pid*b+lane)*sx+(col+channel)*sy) := by ring
  rw [eq,BitVec.toNat_ofNat,Nat.mod_eq_of_lt fits]

#axiomsClean coordinate_mask
#axiomsClean mask_exact
#axiomsClean pointer_exact
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.Native
