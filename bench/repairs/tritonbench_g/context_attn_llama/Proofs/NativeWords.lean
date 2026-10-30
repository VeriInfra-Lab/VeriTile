import bench.repairs.tritonbench_g.context_attn_llama.Kernels
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Native
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
abbrev sword (n : Int) : BitVec 64 := BitVec.ofInt 64 n

theorem unsigned (n : Nat) (h : n < 2^64) : (word n).toNat = n := by
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt h]
theorem signed (n : Nat) (h : n < 2^63) : (word n).toInt = (n : Int) := by
  have hn := unsigned n (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
theorem signedInt (n : Int) (lo : -(2^63) ≤ n) (hi : n < 2^63) :
    (sword n).toInt = n := BitVec.toInt_ofInt_eq_self (by omega) lo hi

def fromI32 (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64

theorem fromI32_eq (n : Nat) (h : n < 2^31) : fromI32 n = word n := by
  have hn : (BitVec.ofNat 32 n).toNat = n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  have hi : (BitVec.ofNat 32 n).toInt = (n : Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
  change BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt = word n
  rw [hi]
  rfl

/-- Actual nonnegative scalar specialization: constexpr one, widened signed32,
and signed64. This covers positive head/group scalars and tensor strides. -/
def scalar (n : Nat) : BitVec 64 :=
  if n=1 then word 1 else if n < 2^31 then fromI32 n else word n
@[simp] theorem scalar_eq (n : Nat) : scalar n = word n := by
  unfold scalar
  split
  · rename_i h; rw [h]
  · split
    · rename_i h; exact fromI32_eq n h
    · rfl

inductive MetadataWidth where | int32 | int64 deriving DecidableEq

def MetadataWidth.bytes : MetadataWidth → Nat | .int32 => 4 | .int64 => 8
def MetadataWidth.Fits : MetadataWidth → Nat → Prop
  | .int32,n => n < 2^31 | .int64,n => n < 2^63
def metadata : MetadataWidth → Nat → BitVec 64
  | .int32,n => fromI32 n | .int64,n => word n

theorem metadata_eq (w : MetadataWidth) (n : Nat) (h : w.Fits n) : metadata w n = word n := by
  cases w
  · exact fromI32_eq n h
  · rfl
theorem metadata_lt (w : MetadataWidth) (n : Nat) (h : w.Fits n) : n < 2^63 := by
  cases w <;> simp only [MetadataWidth.Fits] at h <;> omega

theorem msb_false (n : Nat) (h : n < 2^63) : (word n).msb = Bool.false := by
  rw [BitVec.msb_eq_false_iff_two_mul_lt,unsigned n (by omega)]
  omega

theorem sdiv_eq (a b : Nat) (ha : a < 2^63) (hb : b < 2^63) :
    (word a).sdiv (word b) = word (a/b) := by
  rw [BitVec.sdiv_eq,msb_false a ha,msb_false b hb]
  apply BitVec.eq_of_toNat_eq
  change (word a / word b).toNat = (word (a/b)).toNat
  simp only [BitVec.toNat_udiv,unsigned a (by omega),unsigned b (by omega)]
  exact (unsigned (a/b) (lt_of_le_of_lt (Nat.div_le_self a b) (by omega))).symm

theorem srem_eq (a b : Nat) (ha : a < 2^63) (hb : b < 2^63) :
    (word a).srem (word b) = word (a%b) := by
  rw [BitVec.srem_eq,msb_false a ha,msb_false b hb]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_umod,unsigned a (by omega),unsigned b (by omega)]
  exact (unsigned (a%b) (lt_of_le_of_lt (Nat.mod_le a b) (by omega))).symm

theorem ult_exact (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) :
    (word a).ult (word b) = Bool.true ↔ a < b := by
  rw [BitVec.ult_iff_toNat_lt,unsigned a ha,unsigned b hb]
theorem ule_exact (a b : Nat) (ha : a < 2^64) (hb : b < 2^64) :
    (word a).ule (word b) = Bool.true ↔ a ≤ b := by
  rw [BitVec.ule_iff_toNat_le,unsigned a ha,unsigned b hb]
theorem slt_exact (a b : Nat) (ha : a < 2^63) (hb : b < 2^63) :
    (word a).slt (word b) = Bool.true ↔ a < b := by
  rw [BitVec.slt_iff_toInt_lt,signed a ha,signed b hb]
  exact Int.ofNat_lt

def ptr (base bytes : Nat) (off : BitVec 64) : BitVec 64 :=
  word base + sword ((bytes : Int)*off.toInt)

theorem byte_residue (bytes : Nat) (off : BitVec 64) :
    sword ((bytes : Int)*off.toInt) = word bytes*off := by
  change BitVec.ofInt 64 ((bytes : Int)*off.toInt) = word bytes*off
  rw [BitVec.ofInt_mul,BitVec.ofInt_toInt]
  rfl

theorem ptr_modular (base bytes off : Nat) : ptr base bytes (word off) = word (base+bytes*off) := by
  rw [ptr,byte_residue]
  simp only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]

/-- Python's page-table pointer expression contains two consecutive addptrs.
Both intermediate pointer values may wrap; their final residue is identical
to the sum of the element offsets. -/
def addptr (base : BitVec 64) (bytes : Nat) (off : BitVec 64) :=
  base + sword ((bytes : Int)*off.toInt)

theorem staged_ptr (base bytes : Nat) (a b : BitVec 64) :
    addptr (addptr (word base) bytes a) bytes b = ptr base bytes (a+b) := by
  simp only [addptr,ptr,byte_residue,BitVec.mul_add,BitVec.add_assoc]

theorem ptr_exact (base bytes off extent : Nat) (hp : 0 < bytes)
    (ho : off < extent) (hb : base+bytes*extent ≤ 2^64) :
    (ptr base bytes (word off)).toNat = base+bytes*off := by
  rw [ptr_modular]
  apply unsigned
  have hmul := Nat.mul_lt_mul_of_pos_left ho hp
  omega

/-- Masked-off lanes produce no memory event even when their address arithmetic
wraps. Active lanes use the byte-residue theorem, not a signed-address premise. -/
def access (on : Bool) (base bytes : Nat) (off : BitVec 64) : Option Nat :=
  if on then some (ptr base bytes off).toNat else none

theorem access_exact (on : Bool) (base bytes off extent : Nat) (hp : 0 < bytes)
    (hb : base+bytes*extent ≤ 2^64) (ho : on = Bool.true → off < extent) :
    access on base bytes (word off) = if on then some (base+bytes*off) else none := by
  cases on
  · rfl
  · simp [access,ptr_exact base bytes off extent hp (ho rfl) hb]

end
end VeriTile.Bench.Repairs.ContextAttnLlama.Native
