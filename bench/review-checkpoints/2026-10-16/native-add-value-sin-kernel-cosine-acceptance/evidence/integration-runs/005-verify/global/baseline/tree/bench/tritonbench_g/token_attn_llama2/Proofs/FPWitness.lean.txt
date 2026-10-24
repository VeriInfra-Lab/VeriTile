import bench.tritonbench_g.token_attn_llama2.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def one : Word := FP.Value.ofBits .fp32 0x3f800000
def tiny : Word := FP.Value.ofBits .fp32 0x33000000
def witnessScale : Rat := 1 + 1 / 33554432

def witnessParameters : Parameters where
  q := "Q"
  k := "K"
  scale := (witnessScale : ℝ)
  locations := ⟨"B_Loc"⟩
  starts := ⟨"B_Start_Loc"⟩
  lengths := ⟨"B_Seqlen"⟩
  output := "Out"
  maxLength := 2
  locationBatchStride := 2
  locationSequenceStride := 1
  queryBatchStride := 2
  queryHeadStride := 2
  queryDimStride := 1
  keyBatchStride := 2
  keyHeadStride := 2
  keyDimStride := 1
  outputHeadStride := 8
  outputSequenceStride := 1
  groups := 1
  dimensions := 2
  tokens := 2

def witnessInitial [Inhabited Word] : State Word where
  mem := fun r o =>
    if r = ("B_Seqlen" : RegionName) then .mk .nat 1
    else if r = ("B_Start_Loc" : RegionName) then .mk .nat 5
    else if r = ("B_Loc" : RegionName) then .mk .nat 3
    else .mk .fp32 (if r = ("Q" : RegionName) ∧ o = 1 then tiny else one)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def witnessLane : Index witnessParameters := (⟨0, by decide⟩, PUnit.unit)

theorem witness_scalar_rounding :
    FP.round .fp32 .ieeeValue witnessScale = one ∧
    FP.mul .ieeeValue (FP.add .ieeeValue
      (FP.mul .ieeeValue one one)
      (FP.add .ieeeValue (FP.mul .ieeeValue tiny one) (FP.Value.zero .fp32))) one = one ∧
    witnessScale * witnessScale ≠ 1 := by decide +kernel

theorem witness_score [Inhabited Word] :
    wordScore .ieeeValue ReductionPlan.withSeed (FP.round .fp32 .ieeeValue witnessScale)
      witnessParameters witnessInitial witnessLane = one := by
  simp [wordScore, treeSum, wordProducts, ReductionPlan.withSeed,
    FP.Equational.ReductionTree.ofList, treeValue, queryOffset, keyOffset, keyLocation,
    active, windowIndex, token, length, witnessParameters, witnessInitial, witnessLane,
    List.finRange_succ, Cell.read, Region.cast, witness_scalar_rounding.1]
  exact witness_scalar_rounding.2.1

theorem witness_layout [Inhabited Word] : WritesApart witnessParameters witnessInitial := by
  intro i j hi hj hij
  apply Prod.ext
  · apply Fin.ext
    simpa [outputOffset, witnessParameters, witnessInitial, token, Cell.read, Region.cast] using hij
  · exact Subsingleton.elim _ _

/-- Two dimensions, two token lanes, a nontrivial B_Loc gather, an inactive
lane, and an inexact host-scalar conversion. A witness never changes the
scope of the parameterized theorems. -/
theorem nonempty_bit_success [Inhabited Word] (rest : Algebra Word) :
    ∃ final, FP.Structural.exec
      (softwareBackend rest .ieeeValue ReductionPlan.withSeed (rationalEncoder .ieeeValue witnessScale))
      (source witnessParameters) witnessInitial = some final ∧
      final.mem "Out" 5 = .mk .fp32 one ∧
      final.mem "Out" 6 = witnessInitial.mem "Out" 6 ∧
      final.mem "Q" 0 = witnessInitial.mem "Q" 0 ∧ witnessScale * witnessScale ≠ 1 := by
  obtain ⟨final, he, hp⟩ := rational_correctness rest .ieeeValue ReductionPlan.withSeed
    witnessScale witnessParameters witnessInitial witness_layout
  have hb : blockActive witnessParameters witnessInitial := by
    simp [blockActive, length, witnessParameters, witnessInitial, Cell.read, Region.cast]
  have hi : active witnessParameters witnessInitial witnessLane := by
    simp [active, windowIndex, token, length, witnessParameters, witnessInitial, witnessLane, Cell.read, Region.cast]
  refine ⟨final, he, ?_, ?_, ?_, witness_scalar_rounding.2.2⟩
  · have hv := hp.1 witnessLane hb hi
    change final.mem "Out" 5 = .mk .fp32
      (wordScore .ieeeValue ReductionPlan.withSeed (FP.round .fp32 .ieeeValue witnessScale)
        witnessParameters witnessInitial witnessLane) at hv
    exact hv.trans (congrArg (Cell.mk .fp32) witness_score)
  · apply hp.2 "Out" 6
    right; right
    intro i hi
    simp [active, windowIndex, token, length, witnessParameters, witnessInitial, Cell.read, Region.cast] at hi
    simp [outputOffset, token, witnessParameters, witnessInitial, Cell.read, Region.cast]
    omega
  · exact hp.2 "Q" 0 (Or.inr (Or.inl (by decide)))

#axiomsClean witness_scalar_rounding
#axiomsClean witness_score
#axiomsClean witness_layout
#axiomsClean nonempty_bit_success
end
end VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
