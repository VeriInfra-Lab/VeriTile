import bench.tritonbench_g.rope_embedding.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def one : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def tiny : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000
def negativeZero : FP.Value .fp32 := FP.Value.ofBits .fp32 0x80000000

def expected (backward second : Bool) : FP.Value .fp32 := if backward && second then one.neg else one

theorem rounding (backward second : Bool) :
    (let sn := if backward then one.neg else one;
      if second then FP.add .ieeeValue (FP.mul .ieeeValue tiny one) (FP.mul .ieeeValue one sn)
      else FP.sub .ieeeValue (FP.mul .ieeeValue one one) (FP.mul .ieeeValue tiny sn)) = expected backward second := by
  cases backward <;> cases second <;> decide +kernel

theorem rounding_nonreal :
    one.toRat? = some 1 ∧ tiny.toRat? = some (1 / 33554432) ∧
    (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 := by decide +kernel

def witnessParameters : Parameters := ⟨"q", "cos", "sin", 2, 1, 1, 1, 2, 1, 1, 2⟩

def witnessInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun r o => .mk .fp32 (if r = ("q" : RegionName) ∧ o = 1 then tiny else one)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : TileIndex [witnessParameters.width] := (⟨0, by decide⟩, PUnit.unit)

theorem nonempty_rounding [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32))
    (backward second : Bool) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (source backward second witnessParameters) witnessInitial = some final ∧
      final.mem "q" (if second then 1 else 0) = .mk .fp32 (expected backward second) ∧
      final.mem "q" 2 = witnessInitial.mem "q" 2 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 := by
  obtain ⟨final, he, hp⟩ := software_correctness rest .ieeeValue backward second witnessParameters witnessInitial
  have hi : active witnessParameters witnessInitial lane := by
    simp [active, witnessParameters, witnessInitial, lane]
  refine ⟨final, he, ?_, ?_, rounding_nonreal.2.2.2⟩
  · have hv := hp.1 lane hi
    simpa [roundedValue, offset, coefficientOffset, witnessParameters, witnessInitial, lane,
      rounding backward second] using hv
  · apply hp.2 "q" 2
    right
    intro i hi
    have hc := hi.1
    simp only [witnessParameters] at hc
    have hz : i.1.val = 0 := by omega
    cases second <;> simp [offset, witnessParameters, witnessInitial, hz]

/-- Real fneg flips the sign without quieting a signaling-NaN payload. The
zero subtraction formerly used for unary minus does not have these bits. -/
theorem neg_special_bits :
    (FP.Value.zero .fp32).neg = negativeZero ∧
    (FP.Value.ofBits .fp32 0x7f800001).neg = FP.Value.ofBits .fp32 0xff800001 ∧
    FP.sub .ieeeValue (FP.Value.zero .fp32) (FP.Value.zero .fp32) = FP.Value.zero .fp32 ∧
    negativeZero ≠ FP.Value.zero .fp32 := by decide +kernel

def zeroInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) :=
  { witnessInitial with mem := fun r o => .mk .fp32 (if r = ("sin" : RegionName) then FP.Value.zero .fp32
     else if r = ("q" : RegionName) ∧ o = 1 then negativeZero else one) }

theorem zero_arithmetic :
    FP.add .ieeeValue (FP.mul .ieeeValue negativeZero one)
      (FP.mul .ieeeValue one (FP.Value.zero .fp32).neg) = negativeZero ∧
    FP.add .ieeeValue (FP.mul .ieeeValue negativeZero one)
      (FP.mul .ieeeValue one (FP.sub .ieeeValue (FP.Value.zero .fp32) (FP.Value.zero .fp32))) =
        FP.Value.zero .fp32 := by decide +kernel

theorem backward_signed_zero [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (source Bool.true Bool.true witnessParameters) zeroInitial = some final ∧
      final.mem "q" 1 = .mk .fp32 negativeZero ∧ negativeZero ≠ FP.Value.zero .fp32 := by
  obtain ⟨final, he, hp⟩ := software_correctness rest .ieeeValue Bool.true Bool.true witnessParameters zeroInitial
  have hi : active witnessParameters zeroInitial lane := by
    simp [active, witnessParameters, witnessInitial, zeroInitial, lane]
  refine ⟨final, he, ?_, neg_special_bits.2.2.2⟩
  have hv := hp.1 lane hi
  simpa [roundedValue, offset, coefficientOffset, witnessParameters, witnessInitial, zeroInitial, lane,
    zero_arithmetic.1] using hv

#axiomsClean rounding
#axiomsClean rounding_nonreal
#axiomsClean nonempty_rounding
#axiomsClean neg_special_bits
#axiomsClean backward_signed_zero
end
end VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
