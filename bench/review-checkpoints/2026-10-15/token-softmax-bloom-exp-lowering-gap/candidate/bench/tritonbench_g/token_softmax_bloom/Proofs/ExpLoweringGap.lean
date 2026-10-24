import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Tactic

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.ExpLoweringGap
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
abbrev Word := FP.Value .fp32

def inputA : Word := FP.Value.ofBits .fp32 0xbf400006
def inputB : Word := FP.Value.ofBits .fp32 0xbf400007
def log2e : Word := FP.Value.ofBits .fp32 0x3fb8aa3b
def sharedArgument : Word := FP.Value.ofBits .fp32 0xbf8a7fb1
def roundedExpA : Word := FP.Value.ofBits .fp32 0x3ef1da02
def roundedExpB : Word := FP.Value.ofBits .fp32 0x3ef1da01

/-- Two distinct, ordinary finite negative inputs collapse at the actual
compiled multiply. This statement contains no exponential or SFU callback. -/
theorem multiply_collision :
    FP.mul .ieeeValue inputA log2e = sharedArgument ∧
    FP.mul .ieeeValue inputB log2e = sharedArgument ∧ inputA ≠ inputB := by
  decide +kernel

/-- Both operands are exactly representable differences from a zero row max. -/
theorem input_decoding_and_subtraction :
    inputA.toRat? = some (-6291459 / 8388608 : Rat) ∧
    inputB.toRat? = some (-12582919 / 16777216 : Rat) ∧
    FP.sub .ieeeValue inputA (FP.Value.zero .fp32) = inputA ∧
    FP.sub .ieeeValue inputB (FP.Value.zero .fp32) = inputB := by
  decide +kernel

/-- Exact rational series bound from Mathlib. No host exp, numerical oracle,
GPU observation or assumed output participates in either inequality. -/
theorem real_exp_A_rounding_cell :
    (31699971 : ℝ) / 67108864 < Real.exp (-6291459 / 8388608) ∧
    Real.exp (-6291459 / 8388608) < (31699973 : ℝ) / 67108864 := by
  have hb := Real.exp_bound (x := (-6291459 / 8388608 : ℝ)) (by norm_num) (n := 16) (by norm_num)
  norm_num [Finset.sum_range_succ, Nat.factorial] at hb
  rcases abs_le.mp hb with ⟨hl, hu⟩
  constructor <;> linarith

theorem real_exp_B_rounding_cell :
    (31699969 : ℝ) / 67108864 < Real.exp (-12582919 / 16777216) ∧
    Real.exp (-12582919 / 16777216) < (31699971 : ℝ) / 67108864 := by
  have hb := Real.exp_bound (x := (-12582919 / 16777216 : ℝ)) (by norm_num) (n := 16) (by norm_num)
  norm_num [Finset.sum_range_succ, Nat.factorial] at hb
  rcases abs_le.mp hb with ⟨hl, hu⟩
  constructor <;> linarith

/-- The three neighboring fp32 words on each side give the certified midpoint
cells above. This exposes the exact nearest-even decision; neither value is a tie. -/
theorem output_cell_decoding :
    (FP.Value.ofBits .fp32 0x3ef1da00).toRat? = some (15849984 / 33554432 : Rat) ∧
    roundedExpB.toRat? = some (15849985 / 33554432 : Rat) ∧
    roundedExpA.toRat? = some (15849986 / 33554432 : Rat) ∧
    (FP.Value.ofBits .fp32 0x3ef1da03).toRat? = some (15849987 / 33554432 : Rat) ∧
    roundedExpA ≠ roundedExpB := by
  decide +kernel

/-- This rules out a proposed universal equality, independently of the unknown
implementation of ex2.approx. It is not a proposed replacement exp algorithm. -/
theorem no_ex2_function_matches_both (sfu : Word → Word) :
    ¬ (sfu (FP.mul .ieeeValue inputA log2e) = roundedExpA ∧
       sfu (FP.mul .ieeeValue inputB log2e) = roundedExpB) := by
  rcases multiply_collision with ⟨ha,hb,_⟩
  rw [ha,hb]
  rintro ⟨ha,hb⟩
  exact output_cell_decoding.2.2.2.2 (ha.symm.trans hb)

#axiomsClean multiply_collision
#axiomsClean input_decoding_and_subtraction
#axiomsClean real_exp_A_rounding_cell
#axiomsClean real_exp_B_rounding_cell
#axiomsClean output_cell_decoding
#axiomsClean no_ex2_function_matches_both
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.ExpLoweringGap
