import bench.tritonbench_g.token_attn_mistral.FPCorrect
namespace VeriTile.Bench.TritonBenchG.TokenAttnMistral.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (withSum software softwareSum)
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def witnessOne : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000


def witnessParameters : Parameters where
  prob := "p"
  value := "v"
  output := "out"
  tokens := "tokens"
  request := "request"
  start := "start"
  length := "length"
  attentionLength := "attentionLength"
  unusedStart := "unusedStart"
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
  window := 2
  width := 1
  block := 2

def witnessState : State (FP.Value .fp32) where
  mem := fun r o =>
    if r = ("length" : RegionName) then .mk .nat 2
    else if r = ("attentionLength" : RegionName) then .mk .nat 1
    else if r = ("tokens" : RegionName) then .mk .nat o
    else if r = ("request" : RegionName) ∨ r = ("start" : RegionName) then .mk .nat 0
    else .mk .fp32 (if r = ("v" : RegionName) ∧ o = 1 then FP.Value.infinity .fp32 else witnessOne)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Probability lane 1 is padded while V lane 1 is live and infinite.
The full BLOCK_N multiply yields NaN, as does the final output. Replacing the
block by a sum over only live probabilities would incorrectly return one. -/
theorem witness_bits :
    bitProbability witnessParameters witnessState 0 1 = FP.Value.zero .fp32 ∧
      bitValue witnessParameters witnessState 0 1 0 = FP.Value.infinity .fp32 ∧
      bitOutput .ieeeValue witnessParameters witnessState (⟨0, by decide⟩, PUnit.unit) = FP.Value.qNaN .fp32 := by
  decide +kernel

theorem full_stream_distinct_masks_nan
    (rest : Algebra (FP.Value .fp32)) :
    ∃ t, FP.Structural.exec (withSum (software rest .ieeeValue) (softwareSum .ieeeValue))
        (kernel witnessParameters) witnessState = some t ∧
      t.mem "out" 0 = .mk .fp32 (FP.Value.qNaN .fp32) ∧
      t.mem "out" 1 = witnessState.mem "out" 1 := by
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
  refine ⟨t,he,?_,?_⟩
  · have h := hv (⟨0, by decide⟩,PUnit.unit)
    simpa only [witness_bits.2.2] using h
  · apply hf
    right
    intro i
    have hi : i.1.val = 0 := by have := i.1.isLt; change i.1.val < 1 at this; omega
    simp [outputOffset, witnessParameters, witnessState, hi]

#axiomsClean witness_bits
#axiomsClean full_stream_distinct_masks_nan
end
end VeriTile.Bench.TritonBenchG.TokenAttnMistral.FPStages
