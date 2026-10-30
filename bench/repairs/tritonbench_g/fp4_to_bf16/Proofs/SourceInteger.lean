import bench.tritonbench_g.fp4_to_bf16.Proofs.FPInteger

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue
set_option maxHeartbeats 2000000

/-- Decoding the low signed32 payload cannot change its low byte. This lets
us relate the source uint8 subtraction to the existing independent arithmetic
lemmas without assuming any bound on a masked-load undef value. -/
theorem signed32_mod_byte (z : Int) : signed32 z % 256 = z % 256 := by
  have nonnegative : 0 ≤ z % 4294967296 := Int.emod_nonneg _ (by norm_num)
  have below : z % 4294967296 < 4294967296 := Int.emod_lt_of_pos _ (by norm_num)
  have value : ((z % 4294967296).toNat : Int) = z % 4294967296 := Int.toNat_of_nonneg nonnegative
  have bound : (z % 4294967296).toNat < 4294967296 := by omega
  simp only [signed32, IntegerExpr.signedBits, show (2^32 : Nat) = 4294967296 by norm_num,
    Nat.mod_eq_of_lt bound, value]
  split_ifs <;> omega

theorem minusBias_plain (n : Nat) :
    minusBias n = ((((positive n >>> 1 : Nat) : Int) - 1) % 256).toNat := by
  unfold minusBias
  have same := signed32_mod_byte ((positive n >>> 1 : Nat) : Int)
  congr 1
  omega

#axiomsClean signed32_mod_byte
#axiomsClean minusBias_plain
end VeriTile.Bench.Repairs.Fp4ToBf16.Source
