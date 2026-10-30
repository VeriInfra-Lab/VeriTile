import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Addresses

/-! Finite source primitives. All arithmetic carriers below are actual 64-bit
words. Signed division/remainder, comparison, extension and modular addition
are explicit; bounds are used only in proofs of their interpretation. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
open VeriTile.Triton FP.Structural Whole
noncomputable section
set_option maxHeartbeats 2000000

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

theorem unsigned (n : Nat) (h : n<2^64) : (word n).toNat=n := Nat.mod_eq_of_lt h
theorem signed (n : Nat) (h : n<2^63) : (word n).toInt=(n : Int) := by
  have hn := unsigned n (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

def fromI32 (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64
def fromU32 (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).setWidth 64

theorem fromI32_eq (n : Nat) (h : n<2^31) : fromI32 n=word n := by
  apply BitVec.eq_of_toInt_eq
  rw [fromI32,BitVec.toInt_signExtend_of_le (by decide),signed n (by omega)]
  have hn : (BitVec.ofNat 32 n).toNat=n := Nat.mod_eq_of_lt (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

theorem fromU32_eq (n : Nat) (h : n<2^32) : fromU32 n=word n := by
  apply BitVec.eq_of_toNat_eq
  rw [fromU32,BitVec.toNat_setWidth,BitVec.toNat_ofNat,Nat.mod_eq_of_lt h,
    Nat.mod_eq_of_lt (by omega),unsigned n (by omega)]

inductive ScalarABI where | constexpr1 | i32 | u32 | i64
  deriving DecidableEq
def ScalarABI.Fits : ScalarABI → Nat → Prop
  | .constexpr1,n => n=1 | .i32,n => n<2^31 | .u32,n => n<2^32 | .i64,n => n<2^63
def castABI : ScalarABI → Nat → BitVec 64
  | .constexpr1,_ => word 1 | .i32,n => fromI32 n | .u32,n => fromU32 n | .i64,n => word n

theorem castABI_eq (kind : ScalarABI) (n : Nat) (h : kind.Fits n) : castABI kind n=word n := by
  cases kind
  · change n=1 at h; simp [castABI,h]
  · exact fromI32_eq n h
  · exact fromU32_eq n h
  · rfl

/-- Nonnegative Python scalar specialization, including Triton's u32 interval.
The profile theorem above also covers any other legal choice of these ABIs. -/
def scalar (n : Nat) : BitVec 64 :=
  if n=1 then castABI .constexpr1 n else
  if n<2^31 then castABI .i32 n else
  if n<2^32 then castABI .u32 n else castABI .i64 n

@[simp] theorem scalar_eq (n : Nat) : scalar n=word n := by
  unfold scalar
  split
  · rename_i h; exact castABI_eq .constexpr1 n h
  · split
    · rename_i h; exact castABI_eq .i32 n h
    · split
      · rename_i h; exact castABI_eq .u32 n h
      · rfl

theorem msb_false (n : Nat) (h : n<2^63) : (word n).msb=Bool.false := by
  rw [BitVec.msb_eq_false_iff_two_mul_lt,unsigned n (by omega)]; omega
theorem sdiv_eq (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    (word a).sdiv (word b)=word (a/b) := by
  rw [BitVec.sdiv_eq,msb_false a ha,msb_false b hb]
  apply BitVec.eq_of_toNat_eq
  change (word a / word b).toNat=(word (a/b)).toNat
  simp only [BitVec.toNat_udiv,unsigned a (by omega),unsigned b (by omega)]
  exact (unsigned (a/b) (lt_of_le_of_lt (Nat.div_le_self a b) (by omega))).symm
theorem srem_eq (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    (word a).srem (word b)=word (a%b) := by
  rw [BitVec.srem_eq,msb_false a ha,msb_false b hb]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_umod,unsigned a (by omega),unsigned b (by omega)]
  exact (unsigned (a%b) (lt_of_le_of_lt (Nat.mod_le a b) (by omega))).symm

theorem slt_eq (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    (word a).slt (word b)=decide (a<b) := by
  simp [BitVec.slt,signed a ha,signed b hb]

theorem sub_eq (a b : Nat) (h : b≤a) : word a-word b=word (a-b) := by
  have he : word (a-b)+word b=word a := by
    rw [← BitVec.ofNat_add,Nat.sub_add_cancel h]
  rw [← he,BitVec.add_sub_cancel]

def minimum (a b : BitVec 64) : BitVec 64 := if a.slt b then a else b
theorem minimum_eq (a b : Nat) (ha : a<2^63) (hb : b<2^63) :
    minimum (word a) (word b)=word (min a b) := by
  rw [minimum,slt_eq a b ha hb]
  by_cases h : a<b
  · simp [h,min_eq_left (Nat.le_of_lt h)]
  · simp [h,min_eq_right (Nat.le_of_not_gt h)]

/-- Source cdiv is evaluated as ((x + block) - 1) signed-divided by block. -/
def cdiv (x : BitVec 64) (block : Nat) : BitVec 64 :=
  (x+word block-word 1).sdiv (word block)
theorem cdiv_eq (x block : Nat) (positive : 0<block) (bound : x+block<2^63) :
    cdiv (word x) block=word ((x+block-1)/block) := by
  rw [cdiv,← BitVec.ofNat_add,sub_eq _ _ (by omega)]
  exact sdiv_eq _ _ (by omega) (by omega)

def safe (limit coordinate : BitVec 64) : BitVec 64 :=
  if coordinate.slt limit then coordinate else word 0
theorem safe_eq (limit coordinate : Nat) (hl : limit<2^63) (hc : coordinate<2^63) :
    safe (word limit) (word coordinate)=word (safeCoordinate limit coordinate) := by
  by_cases h : coordinate<limit <;>
    simp [safe,slt_eq coordinate limit hc hl,safeCoordinate,h]

def ptr (base bytes : Nat) (off : BitVec 64) : BitVec 64 :=
  word base + BitVec.ofInt 64 ((bytes : Int)*off.toInt)
theorem ptr_eq (base bytes off : Nat) : ptr base bytes (word off)=word (base+bytes*off) := by
  rw [ptr,BitVec.ofInt_mul,BitVec.ofInt_toInt]
  change word base+word bytes*word off=_
  simp only [← BitVec.ofNat_mul,← BitVec.ofNat_add]

/-- Full access interval, including the final byte, and actual finite pointer. -/
theorem bytes_exact (base bytes extent off : Nat) (positive : 0<bytes)
    (span : base+bytes*extent≤2^64) (inside : off<extent) :
    (ptr base bytes (word off)).toNat=base+bytes*off ∧
    (∀ byte, byte<bytes → base≤(ptr base bytes (word off)).toNat+byte ∧
      (ptr base bytes (word off)).toNat+byte<base+bytes*extent ∧
      (ptr base bytes (word off)).toNat+byte<2^64) ∧
    ((ptr base bytes (word off)).toNat-base)/bytes=off := by
  have hmul : bytes*(off+1)≤bytes*extent := Nat.mul_le_mul_left bytes inside
  have he : (ptr base bytes (word off)).toNat=base+bytes*off := by
    rw [ptr_eq,unsigned _ (by nlinarith)]
  refine ⟨he,?_,?_⟩
  · intro byte hb; rw [he]; constructor; omega
    constructor <;> nlinarith
  · rw [he,Nat.add_sub_cancel_left,Nat.mul_div_right _ positive]

#axiomsClean castABI_eq
#axiomsClean cdiv_eq
#axiomsClean bytes_exact
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Native
