import bench.tritonbench_g.uniform_sampling.Proofs.ScalarSpec
import bench.tritonbench_g.uniform_sampling.Proofs.SourceDomain

/-! Integer representation obligations for the original uniform source.
These statements cover machine intermediates and negative signed64 inputs;
they do not substitute sampled Philox results for execution. -/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.WordArithmetic
open VeriTile.Triton ScalarSpec SourceDomain

/-- The fallback's negation and subtraction are themselves 32-bit operations.
Even at INT32_MIN, the final folded word agrees with the canonical decoder. -/
theorem fallback_machine_fold (x : Random.Word) :
    (if x.toInt < 0 then -x - 1 else x).toNat = Random.foldSign x := by
  by_cases h : x.toNat < 2^31
  · have hs : ¬ x.toInt < 0 := by rw [BitVec.toInt_neg_iff]; omega
    simp only [if_neg hs, Random.foldSign, if_pos h]
  · have hs : x.toInt < 0 := by rw [BitVec.toInt_neg_iff]; omega
    rw [if_pos hs]
    change (-x - (1#32)).toNat = _
    rw [← BitVec.not_eq_neg_add x, BitVec.toNat_not]
    simp only [Random.foldSign, if_neg h]

theorem high_is_arithmetic_shift (offset : Int) :
    counterHigh offset = BitVec.ofInt 32 (offset >>> 32) := by
  rw [Int.shiftRight_eq_div_pow]
  rfl

theorem signed64_edge_words :
    counterLow (-(2^63)) 0 = 0 ∧ counterHigh (-(2^63)) = 0x80000000 ∧
    counterLow (2^63-1) 0 = 0xffffffff ∧ counterHigh (2^63-1) = 0x7fffffff ∧
    (Random.seedKey (-1)).k0 = 0xffffffff ∧
    (Random.seedKey (-1)).k1 = 0xffffffff := by
  decide +kernel

/-- Full source i4 arithmetic before conversion to the unsigned counter. -/
def counterIndexBits (pid B lane : Nat) : Random.Word :=
  BitVec.ofNat 32 pid * BitVec.ofNat 32 B + BitVec.ofNat 32 lane

/-- Source off_0 arithmetic, retaining its two separate multiplications. -/
def outputZeroBits (pid B lane : Nat) : Random.Word :=
  BitVec.ofNat 32 pid * BitVec.ofNat 32 B * 4 + BitVec.ofNat 32 lane

def outputBits (pid B lane field : Nat) : Random.Word :=
  match field with
  | 0 => outputZeroBits pid B lane
  | n+1 => outputBits pid B lane n + BitVec.ofNat 32 B

theorem counter_bits_eq (pid B lane : Nat) :
    counterIndexBits pid B lane = BitVec.ofNat 32 (counterIndex pid B lane) := by
  unfold counterIndexBits counterIndex
  rw [BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]

theorem output_bits_eq (pid B lane field : Nat) :
    outputBits pid B lane field = BitVec.ofNat 32 ((pid*4+field)*B+lane) := by
  induction field with
  | zero =>
      unfold outputBits outputZeroBits
      change BitVec.ofNat 32 pid * BitVec.ofNat 32 B * BitVec.ofNat 32 4 +
        BitVec.ofNat 32 lane = _
      rw [BitVec.ofNat_mul_ofNat, BitVec.ofNat_mul_ofNat, BitVec.ofNat_add_ofNat]
      congr 1
      ring
  | succ n ih =>
      rw [outputBits, ih, BitVec.ofNat_add_ofNat]
      congr 1
      ring

theorem signed_i32_of_lt {n : Nat} (hn : n < 2^31) :
    (BitVec.ofNat 32 n).toInt = (n : Int) := by
  have hnat : (BitVec.ofNat 32 n).toNat = n := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hnat]; omega), hnat]

/-- Source machine results agree with the mathematical address formulas on
every launched lane, including each masked-off tail of the fourth store. -/
theorem machine_indices_exact {N B pid : Nat} (hB : LegalBlock B)
    (hN : N < 2^31) (hpid : pid < grid N B) (f : Fin 4) (i : Fin B) :
    (counterIndexBits pid B i.val).toInt = (counterIndex pid B i.val : Int) ∧
    (outputBits pid B i.val f.val).toInt = (outputIndex pid B f i : Int) := by
  rw [counter_bits_eq, output_bits_eq]
  exact ⟨signed_i32_of_lt (counter_index_lt_i32 hB hN hpid i.isLt),
    signed_i32_of_lt (launched_all_lanes_lt_i32 hB hN hpid f i)⟩

#axiomsClean fallback_machine_fold
#axiomsClean high_is_arithmetic_shift
#axiomsClean signed64_edge_words
#axiomsClean counter_bits_eq
#axiomsClean output_bits_eq
#axiomsClean machine_indices_exact

end VeriTile.Bench.TritonBenchG.UniformSampling.WordArithmetic
