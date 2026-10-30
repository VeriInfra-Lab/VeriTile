import Mathlib
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.Int8Quantization.Words
abbrev Byte := BitVec 8
abbrev Word := Fin 4 → Byte
abbrev Memory := Nat → Byte
def signed64 (n : Int) := (BitVec.ofInt 64 n).toInt
def int32 (n : Int) := (BitVec.ofInt 32 n).toInt
theorem signed64_exact (n : Int) (lo : -(2^63)≤ n) (hi : n<2^63) : signed64 n=n :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi
theorem int32_exact (n : Int) (lo : -(2^31)≤ n) (hi : n<2^31) : int32 n=n :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi
end VeriTile.Bench.Repairs.Int8Quantization.Words
