import bench.tritonbench_g.int8_matmul_quantization.Proofs.MatmulComplete
namespace VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
local instance matmulWitnessDefault : Inhabited Word := ⟨FP.Value.zero .fp32⟩
abbrev witnessParameters : Parameters where
  input := "A"
  weight := "B"
  inputScale := "AS"
  weightScale := "BS"
  output := "C"
  rows := 3
  columns := 3
  inner := 3
  inputRowStride := 10
  inputInnerStride := 1
  inputScaleStride := 1
  weightInnerStride := 10
  weightColumnStride := 1
  weightScaleStride := 1
  outputRowStride := 10
  outputColumnStride := 1
  blockRows := 2
  blockColumns := 2
  blockInner := 2
  groupRows := 2
  steps := 3

def witnessInitial : State Word where
  mem := fun r o =>
    if r="A" then .mk .int (if o=20 then 2 else if o=21 then -3 else if o=22 then 5 else -128)
    else if r="B" then .mk .int (if o=2 then -5 else if o=12 then 6 else if o=22 then 7 else 127)
    else if r="AS" then .mk .fp16 (if o=2 then FP.Value.ofBits .fp32 0x3800 else FP.Value.ofBits .fp32 0x7d01)
    else if r="BS" then .mk .real (if o=2 then FP.Value.ofBits .fp32 0x40000000 else FP.Value.ofBits .fp32 0xffa12345)
    else .mk .fp16 (FP.Value.ofBits .fp32 0x1234)
  regs := fun _ _ _ => none
  pids := fun i => if i=0 then 3 else 0
  numPids := fun i => if i=0 then 4 else 1
  undef := fun d _ _ => defaultValue d

private theorem witness_address (i : CIndex witnessParameters)
    (hi : active witnessParameters witnessInitial i) :
    i.1.val=0 ∧ i.2.1.val=0 := by
  simp [active, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
    witnessParameters, witnessInitial] at hi
  omega

/-- Three blocks: full, ragged, and entirely masked. Mixed signs produce
7; the actual half(0.5) and fp32(2) scales preserve 7 through both rounded
products, then the final half encoding is 0x4700. -/
theorem witness_value :
    outputValue .ieeeValue witnessParameters witnessInitial (0,0,PUnit.unit) =
      FP.Value.ofBits .fp32 0x4700 := by decide +kernel

theorem bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue) (source witnessParameters) witnessInitial = some final ∧
      final.mem "C" 22 = .mk .fp16 (FP.Value.ofBits .fp32 0x4700) ∧
      final.mem "C" 23 = .mk .fp16 (FP.Value.ofBits .fp32 0x1234) := by
  have hinj : ∀ i j : CIndex witnessParameters,
      active witnessParameters witnessInitial i → active witnessParameters witnessInitial j →
      outputOffset witnessParameters witnessInitial i=outputOffset witnessParameters witnessInitial j → i=j := by
    intro i j hi hj _
    have ha := witness_address i hi
    have hb := witness_address j hj
    rcases i with ⟨r,c,u⟩
    rcases j with ⟨r',c',u'⟩
    have hr : r=r' := Fin.ext (ha.1.trans hb.1.symm)
    have hc : c=c' := Fin.ext (ha.2.trans hb.2.symm)
    cases hr; cases hc; cases u; cases u'; rfl
  obtain ⟨final, he, hv, hf⟩ := correctness rest .ieeeValue witnessParameters witnessInitial hinj
  refine ⟨final, he, ?_, ?_⟩
  · have h := hv (0,0,PUnit.unit) (by decide)
    change final.mem "C" 22 = .mk .fp16
      (outputValue .ieeeValue witnessParameters witnessInitial (0,0,PUnit.unit)) at h
    exact h.trans (congrArg (Cell.mk .fp16) witness_value)
  · have h := hf "C" 23 (Or.inr (by
      intro i hi
      have ha := witness_address i hi
      simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
        witnessParameters, witnessInitial, ha.1, ha.2]))
    exact h
#axiomsClean witness_value
#axiomsClean bit_success
end
end VeriTile.Bench.TritonBenchG.Int8MatmulQuantization.FPMatmul
