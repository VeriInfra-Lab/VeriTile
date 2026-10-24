import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordContract
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
local instance scaledWitnessDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev witnessParameters (even : Bool) : Parameters where
  input := "A"
  weight := "B"
  scale := "S"
  output := "C"
  rows := 3
  columns := 3
  inner := if even then 4 else 3
  inputRowStride := 10
  inputInnerStride := 1
  weightInnerStride := 10
  weightColumnStride := 1
  outputRowStride := 999
  outputColumnStride := 999
  scaleRowStride := 999
  scaleColumnStride := 999
  blockRows := 2
  blockColumns := 2
  blockInner := 2
  groupRows := 2
  steps := if even then 2 else 3
  even := even

def witnessInitial : State Word where
  mem := fun r o =>
    if r="A" then .mk .int (if o=20 then 2 else if o=21 then -3 else if o=22 then 5 else if o=23 then 1 else -128)
    else if r="B" then .mk .int (if o=2 then -5 else if o=12 then 6 else if o=22 then 7 else if o=32 then 2 else 127)
    else if r="S" then .mk .real (if o=2 then FP.Value.ofBits .fp32 0x3fc00000 else FP.Value.ofBits .fp32 0xffa12345)
    else .mk .int 777
  regs := fun _ _ _ => none
  pids := fun _ => 3
  numPids := fun _ => 4
  undef := fun d _ _ => match d with
    | .real => FP.Value.ofBits .fp32 0xffa12345
    | d' => defaultValue d'

private theorem witness_address (b : Bool) (i : CIndex (witnessParameters b))
    (hi : active (witnessParameters b) witnessInitial i) : i.1.val=0 ∧ i.2.1.val=0 := by
  simp [active,outputRow,outputColumn,rowBlock,columnBlock,firstRow,rowsInGroup,
    blocksM,blocksN,witnessParameters,witnessInitial] at hi
  omega

/-- Both branch outputs are nonintegral fp32 products. The actual signed32
store truncates 13.5 to 13 and 10.5 to 10, respectively. -/
theorem witness_cast (b : Bool) :
    toSigned 32 (scaled .ieeeValue (witnessParameters b) witnessInitial (0,0,PUnit.unit)) =
      some (if b then 13 else 10) := by cases b <;> decide +kernel

theorem witness_defined (b : Bool) : InputDefined .ieeeValue (witnessParameters b) witnessInitial := by
  intro i hi
  have ha := witness_address b i hi
  rcases i with ⟨r,c,u⟩
  have hr : r=0 := Fin.ext ha.1
  have hc : c=0 := Fin.ext ha.2
  cases hr; cases hc; cases u
  rw [witness_cast]
  rfl

/-- The scale load has no other. Its inactive signaling NaN really reaches
an undefined integer conversion; no padding-finiteness assumption is hidden. -/
theorem inactive_cast_undefined (b : Bool) :
    toSigned 32 (FP.mul .ieeeValue
      (FP.round .fp32 .ieeeValue (accumulation (witnessParameters b) witnessInitial
        (witnessParameters b).steps (1,1,PUnit.unit) : Rat))
      (rawScale (witnessParameters b) witnessInitial (1,1,PUnit.unit)))=none := by
  cases b <;> decide +kernel

theorem bit_success (rest : Algebra Word) (u : Nat → Word → Int) (b : Bool) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue u) (source (witnessParameters b)) witnessInitial=some final ∧
      final.mem "C" 8 = .mk .int (if b then 13 else 10) ∧
      final.mem "C" 9 = .mk .int 777 := by
  obtain ⟨final,he,hv,hf⟩ := correctness rest .ieeeValue u (witnessParameters b) witnessInitial
    (witness_defined b) (output_injective _ _)
  refine ⟨final,he,?_,?_⟩
  · have h := hv (0,0,PUnit.unit) (by simp [active,outputRow,outputColumn,rowBlock,columnBlock,
      firstRow,rowsInGroup,blocksM,blocksN,witnessParameters,witnessInitial])
    have ho : outputOffset (witnessParameters b) witnessInitial (0,0,PUnit.unit)=8 := by
      simp [outputOffset,outputRow,outputColumn,rowBlock,columnBlock,firstRow,rowsInGroup,
        blocksM,blocksN,witnessParameters,witnessInitial]
    rw [ho] at h
    unfold expected at h
    rw [witness_cast] at h
    exact h
  · have h := hf "C" 9 (Or.inr (by
      intro i hi
      have ha := witness_address b i hi
      simp [outputOffset,outputRow,outputColumn,rowBlock,columnBlock,firstRow,rowsInGroup,
        blocksM,blocksN,witnessParameters,witnessInitial,ha.1,ha.2]))
    exact h
#axiomsClean witness_cast
#axiomsClean witness_defined
#axiomsClean inactive_cast_undefined
#axiomsClean bit_success
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
