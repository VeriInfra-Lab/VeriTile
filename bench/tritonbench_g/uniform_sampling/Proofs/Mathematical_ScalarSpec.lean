import VeriTile.Triton.Random.DirectProofs
import VeriTile.Triton.Random.UniformValues
import VeriTile.Meta.StatementAudit

/-!
Independent, input-only scalar specification for the pinned uniform kernel.
There is no AST or execution dependency. Counter arithmetic is genuinely
modulo 2^32, while the high counter is the separately extracted offset word;
overflow of the lane addition does not carry into that high word. All four
Philox outputs are retained. The uniform decoder is the canonical two-rounding
fp32 decoder, including the signed bit reinterpretation and negative fold.

This module alone makes no source-correspondence or hardware-conformance claim.
The final real affine formula erases only the surrounding source arithmetic;
the generator's two internal fp32 rounding steps remain in `uniformReal`.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.ScalarSpec
open VeriTile.Triton

def counterLow (offset : Int) (i4 : Nat) : Random.Word :=
  BitVec.ofInt 32 offset + BitVec.ofNat 32 i4

/-- Arithmetic right shift by 32, followed by the source uint32 conversion. -/
def counterHigh (offset : Int) : Random.Word :=
  BitVec.ofInt 32 (offset / (2^32 : Int))

def counter (offset : Int) (i4 : Nat) : Random.Counter :=
  ⟨counterLow offset i4, counterHigh offset, 0, 0⟩

def words (seed offset : Int) (i4 : Nat) : Random.Counter :=
  Random.philox 10 (counter offset i4) (Random.seedKey seed)

def field (c : Random.Counter) : Random.Direct.Field → Random.Word
  | .c0 => c.c0
  | .c1 => c.c1
  | .c2 => c.c2
  | .c3 => c.c3

def word (seed offset : Int) (i4 : Nat) (f : Random.Direct.Field) : Random.Word :=
  field (words seed offset i4) f

noncomputable def value (seed offset : Int) (i4 : Nat) (f : Random.Direct.Field)
    (from_ upper : ℝ) : ℝ :=
  Random.uniformReal (word seed offset i4 f) * (upper - from_) + from_

/-- Exact source order for the unfused fp32 arithmetic interpretation. Native
contraction policy is a separate source-admission obligation. -/
def roundedValue (seed offset : Int) (i4 : Nat) (f : Random.Direct.Field)
    (from_ upper : FP.Value .fp32) : FP.Value .fp32 :=
  FP.add FP.Config.ieeeValue
    (FP.mul FP.Config.ieeeValue (Random.uniform (word seed offset i4 f))
      (FP.sub FP.Config.ieeeValue upper from_)) from_

/-- The affine contraction observed in the recorded native profiles. This is
a separate software scalar formula, not a hardware-conformance theorem. -/
def fusedValue (seed offset : Int) (i4 : Nat) (f : Random.Direct.Field)
    (from_ upper : FP.Value .fp32) : FP.Value .fp32 :=
  FP.fma FP.Config.ieeeValue (Random.uniform (word seed offset i4 f))
    (FP.sub FP.Config.ieeeValue upper from_) from_

/-- Connect the compatibility helper's signed bitcast/where to the canonical
decoder's integer fold, for every possible uint32 bit pattern. -/
theorem fallback_fold (x : Random.Word) :
    Random.foldSign x = (if x.toInt < 0 then -x.toInt - 1 else x.toInt).toNat := by
  have hx := x.isLt
  simp only [Random.foldSign, BitVec.toInt_eq_toNat_cond]
  split <;> split <;> split <;> omega

theorem low_wraps_without_carry :
    counterLow (-1) 1 = 0 ∧ counterHigh (-1) = 0xffffffff := by
  decide +kernel

theorem all_four_words {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (i : TileIndex [B]) :
    Random.Direct.pointwise
      (Random.Direct.run (Random.Direct.uniformPlan [B] 9)
        (Random.Direct.uniformInput (fun j => counterLow offset (i4 j))
          (fun _ => 0) (fun _ => 0) (counterHigh offset))
        (Random.seedKey seed)) i = words seed offset (i4 i) := by
  rw [Random.Direct.uniformPlan_pointwise, Random.Direct.pointwise_ofPointwise]
  rfl

theorem decoder_range (seed offset : Int) (i4 : Nat) (f : Random.Direct.Field) :
    0 ≤ Random.uniformRat (word seed offset i4 f) ∧
    Random.uniformRat (word seed offset i4 f) ≤ 1 - (1 : Rat) / 2^24 ∧
    Random.uniformRat (word seed offset i4 f) < 1 :=
  Random.uniformRat_range _

#axiomsClean fallback_fold
#axiomsClean low_wraps_without_carry
#axiomsClean all_four_words
#axiomsClean decoder_range

end VeriTile.Bench.TritonBenchG.UniformSampling.ScalarSpec
