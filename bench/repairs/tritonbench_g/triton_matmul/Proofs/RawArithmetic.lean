import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawControl

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section

/-- The Boolean cast contributes exactly 0 or 1 to the signed quotient. -/
theorem signedCeil_nat (n block : Nat) :
    signedCeil n block = ((n / block + if n % block = 0 then 0 else 1 : Nat) : Int) := by
  unfold signedCeil
  rw [← Int.ofNat_tdiv, ← Int.ofNat_tmod]
  by_cases h : n % block = 0 <;> simp [h]
  simpa only [Int.natCast_dvd_natCast,Nat.dvd_iff_mod_eq_zero] using h

theorem signedCeil_nonnegative (n block : Nat) : 0 ≤ signedCeil n block := by
  rw [signedCeil_nat]
  exact Int.natCast_nonneg _

/-- Both original host configurations use the same complete quotient/remainder
ceiling, including K = 0 and an incomplete final tile. -/
theorem signedCeil_configs (n : Nat) :
    signedCeil n 64 = (((n + 63) / 64 : Nat) : Int) ∧
    signedCeil n 128 = (((n + 127) / 128 : Nat) : Int) ∧
    signedCeil n 256 = (((n + 255) / 256 : Nat) : Int) := by
  simp only [signedCeil_nat]
  constructor
  · congr 1; split_ifs <;> omega
  constructor <;> congr 1 <;> split_ifs <;> omega

#axiomsClean signedCeil_nat
#axiomsClean signedCeil_nonnegative
#axiomsClean signedCeil_configs
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
