import VeriTile.Triton.Random.DecoderCompatibility
import VeriTile.Triton.Random.DropoutFormula

/-! Published KATs, source-interpreter values, rounding boundaries, and trust checks.
These are deterministic software checks; they do not launch a GPU or admit a backend.
-/
namespace VeriTile.Bench.Tests.PhiloxRandom
open VeriTile.Triton.Random

-- Independent Random123 v1.14.0 published known-answer vectors.
-- https://github.com/DEShawResearch/random123/blob/v1.14.0/tests/kat_vectors
-- Ordinary `decide` elaborates proofs checked by Lean's kernel.
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

theorem kat_zero_7 : philox 7 ⟨0,0,0,0⟩ ⟨0,0⟩ =
    ⟨0x5f6fb709,0x0d893f64,0x4f121f81,0x4f730a48⟩ := by decide

theorem kat_ones_7 : philox 7 ⟨0xffffffff,0xffffffff,0xffffffff,0xffffffff⟩
    ⟨0xffffffff,0xffffffff⟩ =
    ⟨0x5207ddc2,0x45165e59,0x4d8ee751,0x8c52f662⟩ := by decide

theorem kat_pi_7 : philox 7 ⟨0x243f6a88,0x85a308d3,0x13198a2e,0x03707344⟩
    ⟨0xa4093822,0x299f31d0⟩ =
    ⟨0x4dfccaba,0x190a87f0,0xc47362ba,0xb6b5242a⟩ := by decide

theorem kat_zero_10 : philox 10 ⟨0,0,0,0⟩ ⟨0,0⟩ =
    ⟨0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8⟩ := by decide

theorem kat_ones_10 : philox 10 ⟨0xffffffff,0xffffffff,0xffffffff,0xffffffff⟩
    ⟨0xffffffff,0xffffffff⟩ =
    ⟨0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd⟩ := by decide

theorem kat_pi_10 : philox 10 ⟨0x243f6a88,0x85a308d3,0x13198a2e,0x03707344⟩
    ⟨0xa4093822,0x299f31d0⟩ =
    ⟨0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1⟩ := by decide

theorem negative_seed_words : seedKey (-1) = ⟨0xffffffff,0xffffffff⟩ := by decide

theorem high_seed_word : seedKey (2^32) = ⟨0,1⟩ := by decide

theorem offset_32_wrap : offsetCounter .bits32 (2^32) = ⟨0,0,0,0⟩ := by decide

theorem offset_64_high : offsetCounter .bits64 (2^32) = ⟨0,1,0,0⟩ := by decide

theorem offset_64_negative : offsetCounter .bits64 (-1) =
    ⟨0xffffffff,0xffffffff,0,0⟩ := by decide

theorem scale_bits : scale.toRat? = some (mkRat 16777215 (2^55)) := by decide +kernel

theorem uniform_zero : (uniform 0).raw = 0 := by decide +kernel

theorem uniform_allones : (uniform 0xffffffff).raw = 0 := by decide +kernel

theorem uniform_minsigned : (uniform 0x80000000).raw = 0x3f7fffff := by decide +kernel

theorem uniform_maxsigned : (uniform 0x7fffffff).raw = 0x3f7fffff := by decide +kernel

theorem rand_known_0 : (rand (0) (0) .bits32).raw = 0x3f4c4fd1 := by decide +kernel
theorem rand_known_1 : (rand (0) (1) .bits32).raw = 0x3d63666a := by decide +kernel
theorem rand_known_2 : (rand (123) (0) .bits32).raw = 0x3e091be6 := by decide +kernel
theorem rand_known_3 : (rand (123) (1) .bits32).raw = 0x3f387fd6 := by decide +kernel
theorem rand_known_4 : (rand (123) (1024) .bits32).raw = 0x3ebe465a := by decide +kernel
theorem rand_known_5 : (rand (512) (0) .bits32).raw = 0x3ee321bc := by decide +kernel
theorem rand_known_6 : (rand (-1) (0) .bits32).raw = 0x3f6548ed := by decide +kernel
theorem rand_known_7 : (rand (4294967296) (0) .bits32).raw = 0x3c88707c := by decide +kernel
theorem rand_known_8 : (rand (9223372036854775807) (0) .bits32).raw = 0x3ecb9d3e := by decide +kernel
theorem rand_known_9 : (rand (0) (4294967296) .bits64).raw = 0x3f55a18b := by decide +kernel
theorem rand_known_10 : (rand (123) (4294967297) .bits64).raw = 0x3f6f9331 := by decide +kernel
theorem rand_known_11 : (rand (-1) (-1) .bits64).raw = 0x3f1a31af := by decide +kernel
theorem rand_known_12 : (rand (-1) (-1) .bits32).raw = 0x3ef6a634 := by decide +kernel
theorem rand_known_13 : (rand (512) (-2) .bits64).raw = 0x3ee0b0fc := by decide +kernel
theorem rand_known_14 : (rand (123) (281474976710656) .bits64).raw = 0x3f4683d4 := by decide +kernel
theorem rand_known_15 : (rand (9223372036854775807) (-9223372036854775808) .bits64).raw = 0x3ec00cde := by decide +kernel


namespace DyadicExamples
open VeriTile.Triton.Random.Dyadic
theorem integer_tie_low : round24 (2^24 + 1) = 2^24 := by decide +kernel

theorem integer_tie_high : round24 (2^24 + 3) = 2^24 + 4 := by decide +kernel

theorem uniformDyadic_zero : uniformDyadic 0 = 0 := by decide +kernel

theorem uniformDyadic_max : uniformDyadic (2^31-1) = 1 - (1:ℚ)/2^24 := by decide +kernel

end DyadicExamples

#print axioms kat_zero_10
#print axioms kat_pi_10
#print axioms uniform_minsigned
#print axioms rand_known_15
#print axioms DyadicExamples.uniformDyadic_max
end VeriTile.Bench.Tests.PhiloxRandom

#print axioms VeriTile.Triton.Random.foldSign_lt
#print axioms VeriTile.Triton.Random.RoundBridge.canonical_uniform_decode
#print axioms VeriTile.Triton.Random.RoundBridge.canonical_uniform_toRat
#print axioms VeriTile.Triton.Random.RoundBridge.canonical_uniform_range
#print axioms VeriTile.Triton.Random.canonical_rand_toRat
#print axioms VeriTile.Triton.Random.canonical_rand_range
#print axioms VeriTile.Triton.Random.canonical_rand_core_decode
#print axioms VeriTile.Triton.Random.randRat_eq
#print axioms VeriTile.Triton.Random.rand_toRat_randRat
#print axioms VeriTile.Triton.Random.randRat_range
#print axioms VeriTile.Triton.Random.randReal_range
#print axioms VeriTile.Triton.Random.rand_core_decode
#print axioms VeriTile.Triton.Random.seededDropoutExpected_option_eq
#print axioms VeriTile.Triton.Random.seededDropoutExpected_defined
#print axioms VeriTile.Triton.Random.seededDropoutExpected_total
#print axioms VeriTile.Triton.Random.seededDropoutExpected_p_one
#print axioms VeriTile.Triton.Random.seededDropoutExpected_real_p_one
