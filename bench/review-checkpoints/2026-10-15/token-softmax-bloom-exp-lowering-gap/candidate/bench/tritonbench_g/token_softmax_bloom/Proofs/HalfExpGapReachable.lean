import bench.tritonbench_g.token_softmax_bloom.Proofs.ExpGapReachable
import bench.tritonbench_g.quantize_kv_transform.Proofs.MixedBits

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.ExpLoweringGap
open VeriTile.Triton FP.Structural FP.Equational
open SoftmaxReducev.FPValue (softwareBackend treeReduce maximumWord)
open QuantizeKvTransform
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def halfA : Word := FP.Value.ofBits .fp32 0x8006
def halfB : Word := FP.Value.ofBits .fp32 0x8007
def halfMax : Word := FP.Value.ofBits .fp32 0x3a00

def halfBackend (rest : Algebra Word) (exp32 : Word → Word) : Algebra Word :=
  let base := softwareBackend rest .ieeeValue exp32 ReductionPlan.withSeed ReductionPlan.withSeed
  { base with cast := fun p src dst x => match src,dst with
      | .fp16,.real | .fp16,.fp32 => MixedBits.widen x
      | .real,.fp16 | .fp32,.fp16 => MixedBits.narrow .ieeeValue x
      | .fp16,.fp16 => MixedBits.embed (MixedBits.half x)
      | _,_ => base.cast p src dst x }

def halfProbeInitial (x : Word) : State Word :=
  { (probeInitial x) with
    mem := fun r o =>
      if r = "start" then .mk .nat 3
      else if r = "length" then .mk .nat 2
      else if r = "logits" then .mk .fp16 (if o = 72 then x else halfMax)
      else .mk .fp16 (FP.Value.ofBits .fp32 0xbeef) }

def halfProbeRow (x : Word) (i : TileIndex [32]) : Word :=
  if i.1.val < 2 then MixedBits.widen (if i.1.val = 1 then x else halfMax)
  else FP.Value.infinity .fp32 Bool.true

def halfProbeMaximum (x : Word) : Word :=
  treeReduce ReductionPlan.withSeed (maximumWord .ieeeValue)
    (FP.Value.infinity .fp32 Bool.true) ⟨0,by simp⟩ Bool.false (halfProbeRow x) ()

theorem half_inf_padding :
    MixedBits.widen (MixedBits.narrow .ieeeValue (FP.Value.infinity .fp32 Bool.true)) =
      FP.Value.infinity .fp32 Bool.true := by decide +kernel

/-- The original fp16 storage profile reaches both collision operands: use
max=0.75 and the negative binary16 subnormals with payloads 6 and 7. Widening
preserves those small values; it is not a half-as-fp32 identity assumption. -/
theorem half_subtraction_inputs :
    FP.sub .ieeeValue (MixedBits.widen halfA) (MixedBits.widen halfMax) = inputA ∧
    FP.sub .ieeeValue (MixedBits.widen halfB) (MixedBits.widen halfMax) = inputB ∧
    (MixedBits.widen halfMax).toRat? = some (3/4 : Rat) ∧
    (MixedBits.widen halfA).toRat? = some (-3/8388608 : Rat) ∧
    (MixedBits.widen halfB).toRat? = some (-7/16777216 : Rat) := by
  decide +kernel

theorem half_probe_source_row (rest : Algebra Word) (exp32 : Word → Word) (x : Word) :
    Typed.inputRow (halfBackend rest exp32) .fp16 probeParameters (halfProbeInitial x) = halfProbeRow x := by
  funext i
  simp [Typed.inputRow, Typed.startIndex, Typed.count, probeParameters, probeInitial,
    halfProbeInitial,halfProbeRow,halfBackend,softwareBackend, FP.Software.scalarFP32,
    Region.cast, Cell.read,FloatDType.toTileDType, ofFloat, toFloat,half_inf_padding]
  split_ifs <;> (try simp_all)
  all_goals first | omega | exact half_inf_padding

theorem half_probe_maximums :
    halfProbeMaximum halfA = MixedBits.widen halfMax ∧
    halfProbeMaximum halfB = MixedBits.widen halfMax := by
  decide +kernel

def halfSourceExpArgument (rest : Algebra Word) (exp32 : Word → Word) (x : Word) : Word :=
  let M := halfBackend rest exp32
  let row := Typed.inputRow M .fp16 probeParameters (halfProbeInitial x)
  M.binary (some .fp32) .real .sub (row (⟨1,by decide⟩,()))
    (M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false row ())

theorem half_backend_sub (rest : Algebra Word) (exp32 : Word → Word) (a b : Word) :
    (halfBackend rest exp32).binary (some .fp32) .real .sub a b = FP.sub .ieeeValue a b := rfl

theorem half_backend_max (rest : Algebra Word) (exp32 : Word → Word)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Word .real shape) :
    (halfBackend rest exp32).reduceMax (some .fp32) ax keep xs =
      treeReduce ReductionPlan.withSeed (maximumWord .ieeeValue)
        (FP.Value.infinity .fp32 Bool.true) ax keep xs := rfl

theorem half_source_argument_formula (rest : Algebra Word) (exp32 : Word → Word) (x : Word) :
    halfSourceExpArgument rest exp32 x = FP.sub .ieeeValue (MixedBits.widen x) (halfProbeMaximum x) := by
  simp only [halfSourceExpArgument]
  rw [half_probe_source_row]
  rw [half_backend_sub,half_backend_max]
  rfl

theorem half_source_reaches_both_inputs (rest : Algebra Word) (exp32 : Word → Word) :
    halfSourceExpArgument rest exp32 halfA = inputA ∧
    halfSourceExpArgument rest exp32 halfB = inputB := by
  rw [half_source_argument_formula,half_source_argument_formula,half_probe_maximums.1,half_probe_maximums.2]
  exact ⟨half_subtraction_inputs.1,half_subtraction_inputs.2.1⟩

theorem half_reachable_binding_impossible (rest : Algebra Word) (exp32 sfu : Word → Word) :
    ¬ (sfu (FP.mul .ieeeValue (halfSourceExpArgument rest exp32 halfA) log2e) = roundedExpA ∧
       sfu (FP.mul .ieeeValue (halfSourceExpArgument rest exp32 halfB) log2e) = roundedExpB) := by
  rw [(half_source_reaches_both_inputs rest exp32).1,(half_source_reaches_both_inputs rest exp32).2]
  exact no_ex2_function_matches_both sfu

#axiomsClean half_backend_sub
#axiomsClean half_backend_max
#axiomsClean half_source_argument_formula
#axiomsClean half_inf_padding
#axiomsClean half_subtraction_inputs
#axiomsClean half_probe_source_row
#axiomsClean half_probe_maximums
#axiomsClean half_source_reaches_both_inputs
#axiomsClean half_reachable_binding_impossible
end
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.ExpLoweringGap
