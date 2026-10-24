import Mathlib
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DequantizeRowwise.NativeAddress

def signed64 (n : Int) : Int := (BitVec.ofInt 64 n).toInt
def int32 (n : Int) : Int := (BitVec.ofInt 32 n).toInt
theorem signed64_exact (n : Int) (lo : -(2^63)≤n) (hi : n<2^63) : signed64 n=n :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi
theorem int32_exact (n : Int) (lo : -(2^31)≤n) (hi : n<2^31) : int32 n=n :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi

def pointer (bytes base : Nat) (index : Int) : Nat :=
  (BitVec.ofInt 64 ((base : Int)+(bytes : Int)*index)).toNat

theorem pointer_exact (bytes base count index : Nat) (positive : 0<bytes)
    (capacity : base+bytes*count≤2^64) (bound : index<count) :
    pointer bytes base index=base+bytes*index := by
  have high : base+bytes*index<2^64 := by
    have := Nat.mul_lt_mul_of_pos_left bound positive
    omega
  simp only [pointer,BitVec.toNat_ofInt]
  rw [Int.emod_eq_of_lt (by positivity) (by exact_mod_cast high)]
  norm_cast

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DequantizeRowwise.NativeAddress
