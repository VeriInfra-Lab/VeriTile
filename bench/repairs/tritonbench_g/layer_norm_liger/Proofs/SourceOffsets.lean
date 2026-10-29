import Mathlib.Data.Int.Cast.Basic

namespace VeriTile.Bench.Repairs.LayerNormLiger.SourceOffsets

@[simp] theorem product (a b : Nat) :
    ((a : Int) * (b : Int)).toNat = a*b := by
  rw [← Int.natCast_mul, Int.toNat_natCast]

@[simp] theorem product_add (a b c : Nat) :
    ((a : Int) * (b : Int) + (c : Int)).toNat = a*b+c := by
  rw [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

@[simp] theorem sum (a b : Nat) :
    ((a : Int) + (b : Int)).toNat = a+b := by
  rw [← Int.natCast_add, Int.toNat_natCast]

end VeriTile.Bench.Repairs.LayerNormLiger.SourceOffsets
