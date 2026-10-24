import bench.tritonbench_g.softmax_reducev.Proofs.FPBitContract
namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

/-- Only a witness exponential table: the reachable arguments are zero and
negative infinity. No claim about its values at other arguments is made. -/
def witnessExp (x : Word) : Word :=
  if x.isInf && x.sign then FP.Value.zero .fp32
  else if x.isZero then FP.Value.ofBits .fp32 0x3f800000
  else FP.Value.qNaN .fp32

def witnessParameters : Parameters where
  logics := "L"
  values := "V"
  output := "O"
  locations := ⟨"loc"⟩
  starts := ⟨"start"⟩
  lengths := ⟨"length"⟩
  maxLength := 4
  logicHeadStride := 30
  logicSequenceStride := 1
  valueBatchStride := 4
  valueHeadStride := 20
  valueDimStride := 1
  outputBatchStride := 10
  outputHeadStride := 2
  outputDimStride := 1
  locationBatchStride := 10
  locationSequenceStride := 1
  dimensions := 2
  tokens := 2
  sentinel := -1

def witnessInitial : State Word where
  mem := fun r o =>
    if r = ("length" : RegionName) then .mk .nat 3
    else if r = ("start" : RegionName) then .mk .nat 5
    else if r = ("loc" : RegionName) then .mk .int (if o=11 then 2 else if o=12 then 0 else 1)
    else .mk .fp32 (if r = ("L" : RegionName) then FP.Value.zero .fp32
      else if r = ("V" : RegionName) then FP.Value.ofBits .fp32
        (if o=48 then 0x40000000 else if o=49 then 0x40400000
         else if o=40 then 0x40800000 else if o=41 then 0x40a00000
         else if o=44 then 0x40c00000 else if o=45 then 0x40e00000 else 0x42c60000)
      else FP.Value.ofBits .fp32 0x42c60000)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 1 else 2
  numPids := fun _ => 3
  undef := fun d _ _ => defaultValue d

def firstLane : Index witnessParameters := (⟨0,by decide⟩,())
def secondLane : Index witnessParameters := (⟨1,by decide⟩,())

theorem witness_output :
    wordOutput .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed
      witnessParameters witnessInitial firstLane = FP.Value.ofBits .fp32 0x40800000 ∧
    wordOutput .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed
      witnessParameters witnessInitial secondLane = FP.Value.ofBits .fp32 0x40a00000 := by
  decide +kernel

theorem witness_layout : Function.Injective (outputOffset witnessParameters witnessInitial) := by
  intro i j hij
  apply Prod.ext
  · apply Fin.ext
    simpa [outputOffset,witnessParameters,witnessInitial] using hij
  · exact Subsingleton.elim _ _

/-- Two dimensions and two loop iterations, including a masked final logit,
signed sentinel, permuted gathered rows, nonzero program coordinates, terminal
division, distinct output cells, and untouched cells. -/
theorem nonempty_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec
      (softwareBackend rest .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed)
      (source witnessParameters) witnessInitial = some final ∧
      final.mem "O" 14 = .mk .fp32 (FP.Value.ofBits .fp32 0x40800000) ∧
      final.mem "O" 15 = .mk .fp32 (FP.Value.ofBits .fp32 0x40a00000) ∧
      final.mem "O" 16 = witnessInitial.mem "O" 16 ∧
      final.mem "V" 48 = witnessInitial.mem "V" 48 := by
  obtain ⟨final,he,hp⟩ := software_correctness rest .ieeeValue witnessExp
    ReductionPlan.withSeed ReductionPlan.withSeed witnessParameters witnessInitial (by decide) witness_layout
  refine ⟨final,he,?_,?_,?_,?_⟩
  · have hv := hp.1 firstLane
    change final.mem "O" 14 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) witness_output.1)
  · have hv := hp.1 secondLane
    change final.mem "O" 15 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) witness_output.2)
  · apply hp.2 "O" 16
    right
    intro i
    simp [outputOffset,witnessParameters,witnessInitial]
    have h := i.1.isLt
    change i.1.val < 2 at h
    omega
  · exact hp.2 "V" 48 (Or.inl (by decide))
/-- A second instance satisfies the original positive/divisible/max-length
launch scope: S=4, BLOCK_N=2. The partial-block witness above is additional. -/
def acceptedWitnessInitial : State Word :=
  { witnessInitial with mem := fun r o => if r = ("length" : RegionName) then .mk .nat 4
      else witnessInitial.mem r o }

theorem accepted_witness_output :
    wordOutput .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed
      witnessParameters acceptedWitnessInitial firstLane = FP.Value.ofBits .fp32 0x40900000 ∧
    wordOutput .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed
      witnessParameters acceptedWitnessInitial secondLane = FP.Value.ofBits .fp32 0x40b00000 := by
  decide +kernel

theorem accepted_witness_launch :
    0 < sequenceLength witnessParameters acceptedWitnessInitial ∧
    sequenceLength witnessParameters acceptedWitnessInitial ≤ witnessParameters.maxLength ∧
    sequenceLength witnessParameters acceptedWitnessInitial % witnessParameters.tokens = 0 := by
  decide +kernel

theorem accepted_bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec
      (softwareBackend rest .ieeeValue witnessExp ReductionPlan.withSeed ReductionPlan.withSeed)
      (source witnessParameters) acceptedWitnessInitial = some final ∧
      final.mem "O" 14 = .mk .fp32 (FP.Value.ofBits .fp32 0x40900000) ∧
      final.mem "O" 15 = .mk .fp32 (FP.Value.ofBits .fp32 0x40b00000) ∧
      final.mem "O" 16 = acceptedWitnessInitial.mem "O" 16 ∧
      final.mem "V" 48 = acceptedWitnessInitial.mem "V" 48 := by
  have hw : Function.Injective (outputOffset witnessParameters acceptedWitnessInitial) := witness_layout
  obtain ⟨final,he,hp⟩ := software_correctness rest .ieeeValue witnessExp
    ReductionPlan.withSeed ReductionPlan.withSeed witnessParameters acceptedWitnessInitial (by decide) hw
  refine ⟨final,he,?_,?_,?_,?_⟩
  · have hv := hp.1 firstLane
    change final.mem "O" 14 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) accepted_witness_output.1)
  · have hv := hp.1 secondLane
    change final.mem "O" 15 = _ at hv
    exact hv.trans (congrArg (Cell.mk .fp32) accepted_witness_output.2)
  · apply hp.2 "O" 16
    right
    intro i
    simp [outputOffset,witnessParameters,acceptedWitnessInitial,witnessInitial]
    have h := i.1.isLt
    change i.1.val < 2 at h
    omega
  · exact hp.2 "V" 48 (Or.inl (by decide))
#axiomsClean accepted_witness_output
#axiomsClean accepted_witness_launch
#axiomsClean accepted_bit_success

#axiomsClean witness_output
#axiomsClean witness_layout
#axiomsClean nonempty_bit_success
end
end VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
