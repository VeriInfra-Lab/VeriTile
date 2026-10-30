import bench.tritonbench_g.rotary_transform.Proofs.FPContract

namespace VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def bitsAt [Inhabited (FP.Value .fp32)] (s : State (FP.Value .fp32))
    (r : RegionName) (off : Nat) : FP.Value .fp32 := (s.mem r off).read .real

/-- Independent binary32 formula: every multiplication and the final add or
subtract rounds separately under cfg. No algebra callback appears here. -/
def roundedValue [Inhabited (FP.Value .fp32)] (cfg : FP.Config) (k : Kind)
    (p : Parameters) (s : State (FP.Value .fp32)) (second : Bool)
    (i : TileIndex [p.width]) : FP.Value .fp32 :=
  let x0 := bitsAt s p.input (inputOffset k p s Bool.false i)
  let x1 := bitsAt s p.input (inputOffset k p s Bool.true i)
  let c := if cacheActive p s i then bitsAt s p.cosine (cacheOffset p s i)
    else FP.Value.ofBits .fp32 0x3f800000
  let z := if cacheActive p s i then bitsAt s p.sine (cacheOffset p s i)
    else FP.Value.zero .fp32
  if second then FP.add cfg (FP.mul cfg x0 z) (FP.mul cfg x1 c)
  else FP.sub cfg (FP.mul cfg x0 c) (FP.mul cfg x1 z)

 theorem software_correctness [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) (k : Kind) (p : Parameters)
    (s : State (FP.Value .fp32)) (hstride : p.strideOutDimension ≠ 0) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest cfg) (source k p) s = some final ∧
      Post k p s final (roundedValue cfg k p s) := by
  obtain ⟨final, he, hp⟩ := correctness (FP.Software.scalarFP32 rest cfg) k p s hstride
  refine ⟨final, he, ?_, hp.2⟩
  intro second i hi
  simpa [value, maskedCast, roundedValue, bitsAt, hi,
    FP.Software.scalarFP32, FP.Software.fp32Context] using hp.1 second i hi

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessQuarterUlp : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem witness_rounding :
    FP.sub .ieeeValue (FP.mul .ieeeValue witnessOne witnessOne)
      (FP.mul .ieeeValue witnessOne witnessQuarterUlp) = witnessOne ∧
    FP.add .ieeeValue (FP.mul .ieeeValue witnessOne witnessQuarterUlp)
      (FP.mul .ieeeValue witnessOne witnessOne) = witnessOne ∧
    witnessOne.toRat? = some 1 ∧ witnessQuarterUlp.toRat? = some (1 / 33554432) ∧
    (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 := by decide +kernel

def witnessParameters : Parameters where
  output := "out"
  input := "input"
  cosine := "cos"
  sine := "sin"
  cuLengths := ⟨"cu"⟩
  headIndex := 0
  sequenceOffset := 0
  sequenceLength := 1
  halfDimension := 1
  cosineLength := 1
  strideOutBatch := 8
  strideOutSequence := 4
  strideOutHead := 2
  strideOutDimension := 1
  strideInBatch := 8
  strideInSequence := 4
  strideInHead := 2
  strideInDimension := 1
  blockRows := 1
  width := 2

/-- One active pair, one masked tail, an initially different destination, and
empty registers. Fixed/varlen metadata produce the same nonempty row here. -/
def roundingInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun r o => if r = ("cu" : RegionName) then .mk .nat (if o = 0 then 0 else 1)
    else .mk .real (if r = ("sin" : RegionName) ∨ r = ("out" : RegionName)
      then witnessQuarterUlp else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (k : Kind) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (source k witnessParameters) roundingInitial = some final ∧
      final.mem "out" 0 = .mk .real witnessOne ∧
      final.mem "out" 1 = .mk .real witnessOne ∧
      final.mem "out" 2 = roundingInitial.mem "out" 2 ∧
      (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 := by
  obtain ⟨final, he, hp⟩ := software_correctness rest .ieeeValue k witnessParameters
    roundingInitial (by decide)
  have hi : active k witnessParameters roundingInitial (⟨0, by decide⟩, PUnit.unit) := by
    cases k <;> simp [active, rowIndex, sequenceLength, startIndex, witnessParameters,
      roundingInitial, Region.cast, Cell.read]
  refine ⟨final, he, ?_, ?_, ?_, witness_rounding.2.2.2.2⟩
  · have hv := hp.1 Bool.false (⟨0, by decide⟩, PUnit.unit) hi
    cases k <;> simpa [roundedValue, bitsAt, inputOffset, outputOffset, inputBase, outputBase,
      cacheOffset, cacheActive, rowIndex, headIndex, startIndex, witnessParameters,
      roundingInitial, Region.cast, Cell.read, witness_rounding.1] using hv
  · have hv := hp.1 Bool.true (⟨0, by decide⟩, PUnit.unit) hi
    cases k <;> simpa [roundedValue, bitsAt, inputOffset, outputOffset, inputBase, outputBase,
      cacheOffset, cacheActive, rowIndex, headIndex, startIndex, witnessParameters,
      roundingInitial, Region.cast, Cell.read, witness_rounding.2.1] using hv
  · apply hp.2 "out" 2
    right
    intro second i hi
    have hil := hi.2
    simp only [witnessParameters] at hil
    have hzero : i.1.val = 0 := by omega
    cases k <;> cases second <;>
      simp [outputOffset, outputBase, rowIndex, headIndex, startIndex,
        witnessParameters, roundingInitial, Region.cast, Cell.read, hzero]

#axiomsClean software_correctness
#axiomsClean witness_rounding
#axiomsClean nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.RotaryTransform.FPRow
