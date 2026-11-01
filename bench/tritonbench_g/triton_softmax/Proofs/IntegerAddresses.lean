import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.TritonSoftmax
@[simp] theorem integer_natural_product (a b : Nat) :
    ((a : Int)*(b : Int)).toNat=a*b := by norm_cast
@[simp] theorem integer_natural_row (base row stride : Nat) :
    ((base : Int)+(row : Int)*(stride : Int)).toNat=base+row*stride := by norm_cast
end VeriTile.Bench.Repairs.TritonSoftmax
