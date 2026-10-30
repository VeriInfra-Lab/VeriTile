import Mathlib.Tactic
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.BinaryConversion

def key16 (x : BitVec 16) : BitVec 16 :=
  BitVec.ofNat 16 (if x.toNat % 32768 = 0 then 32768
    else if x.toNat < 32768 then x.toNat + 32768 else 65535 - x.toNat)

def key32 (x : BitVec 32) : BitVec 32 :=
  BitVec.ofNat 32 (if x.toNat % 2147483648 = 0 then 2147483648
    else if x.toNat < 2147483648 then x.toNat + 2147483648 else 4294967295 - x.toNat)

theorem key16_toNat (x : BitVec 16) :
    (key16 x).toNat = if x.toNat % 32768 = 0 then 32768
      else if x.toNat < 32768 then x.toNat + 32768 else 65535 - x.toNat := by
  have h : x.toNat < 65536 := x.isLt
  simp only [key16, BitVec.toNat_ofNat]
  split_ifs <;> omega

theorem key32_toNat (x : BitVec 32) :
    (key32 x).toNat = if x.toNat % 2147483648 = 0 then 2147483648
      else if x.toNat < 2147483648 then x.toNat + 2147483648 else 4294967295 - x.toNat := by
  have h : x.toNat < 4294967296 := x.isLt
  simp only [key32, BitVec.toNat_ofNat]
  split_ifs <;> omega
end VeriTile.Bench.TritonBenchG.MatrixReduction.BinaryConversion
