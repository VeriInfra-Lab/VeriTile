import bench.tritonbench_g.rbe_triton_transform.Proofs.Sources.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RbeTritonTransform.SignedIndex
/-- These identities concern only exact integer address/control arithmetic. -/
theorem nat_not_negative (n : Nat) : ¬ (n : Int) < 0 := by omega
theorem add_nat (a b : Nat) : (a : Int) + (b : Int) = ((a+b : Nat) : Int) := by omega
theorem mul_nat (a b : Nat) : (a : Int) * (b : Int) = ((a*b : Nat) : Int) := by exact_mod_cast rfl
theorem lt_nat (a b : Nat) : (a : Int) < (b : Int) ↔ a < b := by omega
theorem twice_nat (a : Nat) : (a : Int)*2 = ((a*2 : Nat) : Int) := by omega
theorem one_add_nat (a : Nat) : 1+(a : Int) = ((1+a : Nat) : Int) := by omega

theorem ceil_nat (n b : Nat) :
    (if (n : Int) + ((b : Int)-1) < 0 then
      -((1-(b : Int)+ -(n : Int))/(b : Int))
    else ((n : Int)+((b : Int)-1))/(b : Int)) = (((n+b-1)/b : Nat) : Int) := by
  by_cases hb : b=0
  · simp [hb]
  · have h : 1≤n+b := by omega
    have nonnegative : ¬ (n : Int)+((b : Int)-1)<0 := by omega
    rw [if_neg nonnegative]
    have numerator : (n : Int)+((b : Int)-1)=((n+b-1 : Nat) : Int) := by omega
    rw [numerator]
    exact (Int.natCast_ediv _ _).symm

theorem mod_nat (a b : Nat) :
    (a : Int) - (b : Int)*((a : Int)/(b : Int)) = ((a%b : Nat) : Int) := by
  rw [← Int.emod_def]
  exact (Int.natCast_emod _ _).symm

theorem mod_nat_pulled (a b : Nat) :
    (a : Int) - ((b*(a/b) : Nat) : Int) = ((a%b : Nat) : Int) := by
  calc
    _ = (a : Int)-(b : Int)*((a : Int)/(b : Int)) := by simp
    _ = _ := mod_nat a b

theorem mod128_nat (a : Nat) :
    (a : Int)-128*((a : Int)/128) = ((a%128 : Nat) : Int) := by
  exact mod_nat a 128

theorem mod128_nat_pulled (a : Nat) :
    (a : Int)-128*((a/128 : Nat) : Int) = ((a%128 : Nat) : Int) := by
  simpa only [Int.natCast_mul] using mod_nat_pulled a 128

theorem emod128_nat (a : Nat) : (a : Int)%128 = ((a%128 : Nat) : Int) := by
  exact (Int.natCast_emod a 128).symm

#axiomsClean ceil_nat
#axiomsClean mod_nat
end VeriTile.Bench.Repairs.RbeTritonTransform.SignedIndex
