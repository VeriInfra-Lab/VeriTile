import bench.tritonbench_g.fp4_to_bf16.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Fp4ToBf16.SignedIndex
/-- Exact address/control identities; these use no floating-point rule. -/
theorem add_nat (a b : Nat) : (a : Int)+(b : Int)=((a+b : Nat) : Int) := by omega
theorem mul_nat (a b : Nat) : (a : Int)*(b : Int)=((a*b : Nat) : Int) := by exact_mod_cast rfl
theorem twice_nat (a : Nat) : (a : Int)*2=((a*2 : Nat) : Int) := by omega
theorem lt_nat (a b : Nat) : (a : Int)<(b : Int) ↔ a<b := by omega
#axiomsClean add_nat
#axiomsClean mul_nat
#axiomsClean twice_nat
#axiomsClean lt_nat
end VeriTile.Bench.Repairs.Fp4ToBf16.SignedIndex
