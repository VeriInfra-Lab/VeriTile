import bench.tritonbench_g.token_attn_reduceV.FPCorrect
namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def witnessSmall : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000
def witnessTwo : FP.Value .fp32 := FP.Value.ofBits .fp32 0x40000000

def witnessParameters : Parameters where
  prob := "p"
  value := "v"
  output := "out"
  tokens := "tokens"
  request := "request"
  start := "start"
  length := "length"
  requestBatchStride := 1
  requestTokenStride := 1
  probHeadStride := 3
  probTokenStride := 1
  valueBatchStride := 1
  valueHeadStride := 3
  valueDimStride := 1
  outputBatchStride := 1
  outputHeadStride := 1
  outputDimStride := 1
  kvGroup := 1
  width := 1
  block := 2

def witnessState : State (FP.Value .fp32) where
  mem := fun r o =>
    if r = ("length" : RegionName) then .mk .nat 3
    else if r = ("tokens" : RegionName) then .mk .nat o
    else if r = ("request" : RegionName) ∨ r = ("start" : RegionName) then .mk .nat 0
    else .mk .fp32 (if r = ("p" : RegionName) ∧ o = 1 then witnessSmall else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Two blocks, one padded leaf in the second block; the small positive term
is lost at a genuine rounded addition, so the result differs from exact Real
summation. Kernel reduction checks this bit-pattern computation. -/
theorem witness_bits :
    bitOutput .ieeeValue witnessParameters witnessState (⟨0, by decide⟩, PUnit.unit) = witnessTwo ∧
      witnessSmall.toRat? = some (1/33554432) ∧
      (2+1/33554432 : Rat) ≠ 2 := by decide +kernel

theorem full_stream_nonempty_rounding
    (rest : Algebra (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (withSum (software rest .ieeeValue) (softwareSum .ieeeValue))
        (kernel witnessParameters) witnessState = some t ∧
      t.mem "out" 0 = .mk .fp32 witnessTwo ∧
      t.mem "out" 1 = witnessState.mem "out" 1 ∧
      (2+1/33554432 : Rat) ≠ 2 := by
  have hinj : Function.Injective (outputOffset witnessParameters witnessState) := by
    intro i j _
    apply Prod.ext
    · apply Fin.ext
      have hi := i.1.isLt
      have hj := j.1.isLt
      change i.1.val < 1 at hi
      change j.1.val < 1 at hj
      omega
    · exact Subsingleton.elim _ _
  obtain ⟨t,he,hv,hf⟩ := FPCorrect.fp32_software_stream_correctness rest .ieeeValue
    witnessParameters witnessState hinj
  refine ⟨t,he,?_,?_,witness_bits.2.2⟩
  · have h := hv (⟨0, by decide⟩,PUnit.unit)
    simpa only [witness_bits.1] using h
  · apply hf
    right
    intro i
    have hi : i.1.val = 0 := by have := i.1.isLt; change i.1.val < 1 at this; omega
    simp [outputOffset, witnessParameters, witnessState, hi]

#axiomsClean witness_bits
#axiomsClean full_stream_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
