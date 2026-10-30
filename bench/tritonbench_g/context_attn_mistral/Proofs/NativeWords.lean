import bench.tritonbench_g.context_attn_mistral.Proofs.Kernels
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

abbrev word (n : Nat) : BitVec 64 := BitVec.ofNat 64 n
abbrev sword (n : Int) : BitVec 64 := BitVec.ofInt 64 n

theorem word_unsigned (n : Nat) (h : n < 2^64) : (word n).toNat = n := by
  rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt h]

theorem word_signed (n : Nat) (h : n < 2^63) : (word n).toInt = (n : Int) := by
  have hn := word_unsigned n (by omega)
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

theorem sword_signed (n : Int) (lo : -(2^63) ≤ n) (hi : n < 2^63) :
    (sword n).toInt = n := BitVec.toInt_ofInt_eq_self (by omega) lo hi

theorem i32_signed (n : Nat) (h : n < 2^31) : (BitVec.ofNat 32 n).toInt = (n : Int) := by
  have hn : (BitVec.ofNat 32 n).toNat = n := by
    rw [BitVec.toNat_ofNat,Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]

def fromI32 (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64
theorem fromI32_word (n : Nat) (h : n < 2^31) : fromI32 n = word n := by
  change BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt = word n
  rw [i32_signed n h]
  rfl

/-- Ordinary Triton scalar specialization: one may be constexpr; nonnegative
i32 values are widened before mixed arithmetic, larger int64 values retain
their bits. These are the stride, GQA-group and clamped-window ABI routes. -/
def scalarABI (n : Nat) : BitVec 64 :=
  if n = 1 then word 1 else if n < 2^31 then fromI32 n else word n
theorem scalarABI_word (n : Nat) : scalarABI n = word n := by
  unfold scalarABI
  split
  · rename_i h; rw [h]
  · split
    · rename_i h; exact fromI32_word n h
    · rfl

inductive MetadataWidth where
  | int32 | int64
  deriving DecidableEq

def MetadataWidth.bytes : MetadataWidth → Nat | .int32 => 4 | .int64 => 8
def MetadataWidth.Fits : MetadataWidth → Nat → Prop
  | .int32,n => n < 2^31 | .int64,n => n < 2^63
def metadataABI : MetadataWidth → Nat → BitVec 64
  | .int32,n => fromI32 n | .int64,n => word n

theorem metadata_word (w : MetadataWidth) (n : Nat) (h : w.Fits n) : metadataABI w n = word n := by
  cases w
  · exact fromI32_word n h
  · rfl
theorem metadata_signed (w : MetadataWidth) (n : Nat) (h : w.Fits n) :
    (metadataABI w n).toInt = (n : Int) := by
  rw [metadata_word w n h]
  apply word_signed
  cases w <;> simp only [MetadataWidth.Fits] at h <;> omega

theorem signed_div (head group : Nat) (hh : head < 2^31) (hg : group < 2^63) :
    (fromI32 head).sdiv (scalarABI group) = word (head/group) := by
  rw [fromI32_word head hh,scalarABI_word]
  have hn := word_unsigned head (by omega)
  have gn := word_unsigned group (by omega)
  have hm : (word head).msb = Bool.false := by
    rw [BitVec.msb_eq_false_iff_two_mul_lt,hn]; omega
  have gm : (word group).msb = Bool.false := by
    rw [BitVec.msb_eq_false_iff_two_mul_lt,gn]; omega
  rw [BitVec.sdiv_eq,hm,gm]
  apply BitVec.eq_of_toNat_eq
  change (word head / word group).toNat = (word (head/group)).toNat
  rw [BitVec.toNat_udiv,hn,gn]
  exact (word_unsigned (head/group)
    (lt_of_le_of_lt (Nat.div_le_self head group) (show head < 2^64 by omega))).symm

/-- Signed pointer offsets and unsigned byte arithmetic have the same residue,
even when an inactive lane's intermediate signed address wraps. -/
theorem signed_byte_residue (bytes : Nat) (offset : BitVec 64) :
    sword ((bytes : Int)*offset.toInt) = word bytes*offset := by
  change BitVec.ofInt 64 ((bytes : Int)*offset.toInt) = word bytes*offset
  rw [BitVec.ofInt_mul,BitVec.ofInt_toInt]
  rfl

def queryOffset (start row head dim sr sh sd : Nat) : BitVec 64 :=
  (word start+word row)*word sr+word head*word sh+word dim*word sd
def keyOffset (start index lane head dim sr sh sd : Nat) : BitVec 64 :=
  (word lane*word sr+word head*word sh+word dim*word sd)+(word start+word index)*word sr
def bytePointer (base bytes : Nat) (offset : BitVec 64) : BitVec 64 :=
  word base+sword ((bytes : Int)*offset.toInt)

theorem query_modular (start row head dim sr sh sd : Nat) :
    queryOffset start row head dim sr sh sd = word ((start+row)*sr+head*sh+dim*sd) := by
  simp only [queryOffset,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
theorem key_modular (start index lane head dim sr sh sd : Nat) :
    keyOffset start index lane head dim sr sh sd = word ((start+(index+lane))*sr+head*sh+dim*sd) := by
  simp only [keyOffset,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
  congr 1
  ring
theorem byte_modular (base bytes offset : Nat) :
    bytePointer base bytes (word offset) = word (base+bytes*offset) := by
  rw [bytePointer,signed_byte_residue]
  simp only [word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
theorem byte_exact (base bytes offset : Nat) (positive : 0 < bytes)
    (backing : base+bytes*offset+bytes ≤ 2^64) :
    (bytePointer base bytes (word offset)).toNat = base+bytes*offset := by
  rw [byte_modular]
  exact word_unsigned _ (by omega)

end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
