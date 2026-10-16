import VeriTile.Triton
import Mathlib.Analysis.Real.Pi.Bounds
import VeriTile.Triton.Float.ScalarBackend
namespace VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := FP.Value .fp32

def halfWord : Word := FP.Value.ofBits .fp32 0x3f000000
def oneWord : Word := FP.Value.ofBits .fp32 0x3f800000
def cubicWord : Word := FP.Value.ofBits .fp32 0x3d372713
def sqrt2Word : Word := FP.Value.ofBits .fp32 0x3fb504f3
def sqrt2piWord : Word := FP.Value.ofBits .fp32 0x3f4c422a

/-- The old port's two irrational expressions are source tags for the two
actual host constants. Their native fp32 bit values are explicit here; this is
not an assertion that the host evaluates real sqrt exactly. -/
def literalWord (fallback : ℝ → Word) (x : ℝ) : Word :=
  if x=0 then FP.Value.zero .fp32
  else if x=1 then oneWord
  else if x=0.5 then halfWord
  else if x=0.044715 then cubicWord
  else if x=Real.sqrt 2 then sqrt2Word
  else if x=Real.sqrt (2/Real.pi) then sqrt2piWord
  else fallback x

@[simp] theorem literal_zero (f : ℝ → Word) : literalWord f 0=FP.Value.zero .fp32 := by simp [literalWord]
@[simp] theorem literal_one (f : ℝ → Word) : literalWord f 1=oneWord := by norm_num [literalWord]
@[simp] theorem literal_half (f : ℝ → Word) : literalWord f 0.5=halfWord := by norm_num [literalWord]
@[simp] theorem literal_cubic (f : ℝ → Word) : literalWord f 0.044715=cubicWord := by norm_num [literalWord]
private theorem sqrt2_gt_one : (1:ℝ)<Real.sqrt 2 := by
  have hs := Real.sq_sqrt (show (0:ℝ)≤2 by norm_num)
  have hn := Real.sqrt_nonneg (2:ℝ)
  nlinarith
private theorem sqrt2pi_bounds : (0.5:ℝ)<Real.sqrt (2/Real.pi) ∧ Real.sqrt (2/Real.pi)<1 := by
  have hpos : (0:ℝ)<2/Real.pi := div_pos (by norm_num) Real.pi_pos
  have hlo : (1:ℝ)/4 < 2/Real.pi := (lt_div_iff₀ Real.pi_pos).2 (by nlinarith [Real.pi_lt_four])
  have hhi : (2:ℝ)/Real.pi<1 := (div_lt_one Real.pi_pos).2 (by linarith [Real.pi_gt_three])
  have hs := Real.sq_sqrt hpos.le
  have hn := Real.sqrt_nonneg (2/Real.pi)
  constructor <;> nlinarith
@[simp] theorem literal_sqrt2 (f : ℝ → Word) : literalWord f (Real.sqrt 2)=sqrt2Word := by
  have h:=sqrt2_gt_one
  simp only [literalWord,ite_false,ite_true,if_pos rfl,show Real.sqrt 2≠0 by linarith,show Real.sqrt 2≠1 by linarith,
    show Real.sqrt 2≠0.5 by norm_num at *; linarith,show Real.sqrt 2≠0.044715 by norm_num at *; linarith]
@[simp] theorem literal_sqrt2pi (f : ℝ → Word) : literalWord f (Real.sqrt (2/Real.pi))=sqrt2piWord := by
  obtain ⟨hl,hh⟩:=sqrt2pi_bounds
  have h:=sqrt2_gt_one
  simp only [literalWord,ite_false,ite_true,if_pos rfl,show Real.sqrt (2/Real.pi)≠0 by linarith,
    show Real.sqrt (2/Real.pi)≠1 by linarith,
    show Real.sqrt (2/Real.pi)≠0.5 by linarith,
    show Real.sqrt (2/Real.pi)≠0.044715 by norm_num at *; linarith,
    show Real.sqrt (2/Real.pi)≠Real.sqrt 2 by linarith]
@[simp] theorem literal_zero_decimal (f : ℝ → Word) : literalWord f 0.0=FP.Value.zero .fp32 := by
  norm_num [literalWord]
@[simp] theorem literal_one_decimal (f : ℝ → Word) : literalWord f 1.0=oneWord := by
  norm_num [literalWord]
@[simp] theorem literal_sqrt2pi_decimal (f : ℝ → Word) : literalWord f (Real.sqrt (2.0/Real.pi))=sqrt2piWord := by
  convert literal_sqrt2pi f using 1 <;> norm_num
#axiomsClean literal_zero
#axiomsClean literal_one
#axiomsClean literal_half
#axiomsClean literal_cubic
#axiomsClean literal_sqrt2
#axiomsClean literal_sqrt2pi
end
end VeriTile.Bench.TritonBenchG.TritonLinearActivation.FPValue
