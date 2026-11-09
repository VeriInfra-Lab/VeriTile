import bench.tritonbench_g.softmax_triton3.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SoftmaxTriton3
/-- Nonnegative call-site dimensions retain the original signed multiplication
until the memory boundary. This equality is confined to that address projection. -/
@[simp] theorem natural_product_address (a b : Nat) :
    ((a : Int) * b).toNat = a * b := by
  rw [← Int.natCast_mul, Int.toNat_natCast]

@[simp] theorem natural_sum_address (a b : Nat) :
    ((a : Int) + b).toNat = a + b := by
  rw [← Int.natCast_add, Int.toNat_natCast]

@[simp] theorem natural_product_sum_address (a b c : Nat) :
    ((a : Int) * b + c).toNat = a * b + c := by
  rw [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]

#axiomsClean natural_product_address
#axiomsClean natural_sum_address
end VeriTile.Bench.Repairs.SoftmaxTriton3
