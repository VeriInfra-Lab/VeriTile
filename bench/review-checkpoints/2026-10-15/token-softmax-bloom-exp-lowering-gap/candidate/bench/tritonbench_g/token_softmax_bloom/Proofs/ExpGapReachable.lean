import bench.tritonbench_g.token_softmax_bloom.Proofs.ExpLoweringGap
import bench.tritonbench_g.token_softmax_bloom.Proofs.TypedIO
import bench.tritonbench_g.softmax_reducev.Proofs.FPSoftware

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.ExpLoweringGap
open VeriTile.Triton FP.Structural FP.Equational
open SoftmaxReducev.FPValue (softwareBackend treeReduce treeValue maximumWord)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def probeParameters : Typed.Parameters where
  input := "logits"
  output := "output"
  start := "start"
  length := "length"
  inputHeadStride := 64
  inputTokenStride := 2
  outputHeadStride := 64
  outputTokenStride := 3
  width := 32

/-- Separate buffers and nonzero logical offsets are used for the experimental
instance only. The original parametric contract is not narrowed to this case. -/
def probeInitial (x : Word) : State Word where
  mem := fun r o =>
    if r = "start" then .mk .nat 3
    else if r = "length" then .mk .nat 2
    else if r = "logits" then .mk .fp32 (if o = 72 then x else FP.Value.zero .fp32)
    else .mk .fp32 (FP.Value.ofBits .fp32 0xdeadbeef)
  regs := fun _ _ _ => none
  pids := fun i => if i = 0 then 2 else if i = 1 then 1 else 0
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

def probeRow (x : Word) (i : TileIndex [32]) : Word :=
  if i.1.val < 2 then
    if i.1.val = 1 then x else FP.Value.zero .fp32
  else FP.Value.infinity .fp32 Bool.true

def probeMaximum (x : Word) : Word :=
  treeReduce ReductionPlan.withSeed (maximumWord .ieeeValue)
    (FP.Value.infinity .fp32 Bool.true) ⟨0,by simp⟩ Bool.false (probeRow x) ()

/-- The typed input row, including every eager masked lane, has no dependence
on the unimplemented exp function or unrelated algebra callbacks. -/
theorem probe_source_row (rest : Algebra Word) (exp32 : Word → Word) (x : Word) :
    Typed.inputRow (softwareBackend rest .ieeeValue exp32
      ReductionPlan.withSeed ReductionPlan.withSeed) .fp32 probeParameters (probeInitial x) = probeRow x := by
  funext i
  simp [Typed.inputRow, Typed.startIndex, Typed.count, probeParameters, probeInitial,
    probeRow, softwareBackend, FP.Software.scalarFP32, Region.cast, Cell.read,
    FloatDType.toTileDType, ofFloat, toFloat]
  split_ifs <;> (try simp_all)
  all_goals omega

/-- Both rows have ordinary finite active values 0 and x; all thirty padded
lanes are -Inf. The selected explicit software tree produces +0 as maximum. -/
theorem probe_maximums :
    probeMaximum inputA = FP.Value.zero .fp32 ∧
    probeMaximum inputB = FP.Value.zero .fp32 := by
  decide +kernel

def sourceExpArgument (rest : Algebra Word) (exp32 : Word → Word) (x : Word) : Word :=
  let M := softwareBackend rest .ieeeValue exp32 ReductionPlan.withSeed ReductionPlan.withSeed
  let row := Typed.inputRow M .fp32 probeParameters (probeInitial x)
  M.binary (some .fp32) .real .sub (row (⟨1,by decide⟩,()))
    (M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false row ())

theorem backend_sub (rest : Algebra Word) (exp32 : Word → Word) (a b : Word) :
    (softwareBackend rest .ieeeValue exp32 ReductionPlan.withSeed ReductionPlan.withSeed).binary
      (some .fp32) .real .sub a b = FP.sub .ieeeValue a b := rfl

theorem backend_max (rest : Algebra Word) (exp32 : Word → Word) (xs : TileIndex [32] → Word) :
    (softwareBackend rest .ieeeValue exp32 ReductionPlan.withSeed ReductionPlan.withSeed).reduceMax
      (some .fp32) ⟨0,by simp⟩ Bool.false xs () =
      treeReduce ReductionPlan.withSeed (maximumWord .ieeeValue)
        (FP.Value.infinity .fp32 Bool.true) ⟨0,by simp⟩ Bool.false xs () := rfl

/-- These exact inputs really reach the exponent position of the old complete
row reference. No exp result or final output is assumed here. -/
theorem source_reaches_both_inputs (rest : Algebra Word) (exp32 : Word → Word) :
    sourceExpArgument rest exp32 inputA = inputA ∧
    sourceExpArgument rest exp32 inputB = inputB := by
  simp only [sourceExpArgument,probe_source_row,backend_sub,backend_max]
  change FP.sub .ieeeValue (probeRow inputA (⟨1,by decide⟩,())) (probeMaximum inputA) = inputA ∧
    FP.sub .ieeeValue (probeRow inputB (⟨1,by decide⟩,())) (probeMaximum inputB) = inputB
  rw [probe_maximums.1,probe_maximums.2]
  simpa only [probeRow, if_pos (by decide : 1 < 2), if_pos rfl] using
    input_decoding_and_subtraction.2.2

/-- Therefore even on legal two-active-lane, width-32 rows, the proposed
"multiply then arbitrary SFU equals correctly-rounded exp" binding cannot hold
at both exponent operands. This is not a claim that final GPU outputs differ. -/
theorem reachable_binding_impossible (rest : Algebra Word)
    (exp32 sfu : Word → Word) :
    ¬ (sfu (FP.mul .ieeeValue (sourceExpArgument rest exp32 inputA) log2e) = roundedExpA ∧
       sfu (FP.mul .ieeeValue (sourceExpArgument rest exp32 inputB) log2e) = roundedExpB) := by
  rw [(source_reaches_both_inputs rest exp32).1,(source_reaches_both_inputs rest exp32).2]
  exact no_ex2_function_matches_both sfu

#axiomsClean backend_sub
#axiomsClean backend_max
#axiomsClean probe_source_row
#axiomsClean probe_maximums
#axiomsClean source_reaches_both_inputs
#axiomsClean reachable_binding_impossible
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.ExpLoweringGap
