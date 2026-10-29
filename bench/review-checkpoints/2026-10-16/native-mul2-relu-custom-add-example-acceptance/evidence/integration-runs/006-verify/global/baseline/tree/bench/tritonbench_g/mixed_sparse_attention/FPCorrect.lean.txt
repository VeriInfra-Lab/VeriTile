import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPWitness
import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPShape
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPCorrect
open VeriTile.Triton FP.Structural
open FPStages
noncomputable section

/-- The originally accepted forward scope includes all eight fixed block
iterations, the one column iteration for BLOCK_N≥16, and writeback. -/
specification forward_fp16 {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (c : Config) (hBN : 16≤c.BN) (s : State α) (hactive : rowStart c s < sequenceLength c s) :
    ∃ final, FP.Structural.exec (backend M D) (kernel c) s = some final ∧
      final.mem = (outputMemory M c s (acceptedValues M D c s)).mem := by
  simpa only [one_column_iteration M D c s hBN] using whole_fp16 M D c s (by omega) hactive

specification forward_nonempty {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α) :
    ∃ final, FP.Structural.exec (backend M D) (kernel witnessConfig) witnessInitial = some final ∧
      final.mem = (outputMemory M witnessConfig witnessInitial
        (finalValues M D witnessConfig witnessInitial)).mem ∧
      (∃ i : TileIndex [16,16], rowStart witnessConfig (witnessInitial : State α)+i.1.val <
        sequenceLength witnessConfig (witnessInitial : State α)) := by
  exact FPStages.forward_nonempty M D

specification epilogue_fp16 {α : Type} [Inhabited α] (M : Algebra α) (AccPre LPre : RegionName)
    (c : Config) (s : State α) :
    ∃ final, FP.Structural.exec M (epilogueSource AccPre LPre c) s = some final ∧
      final.mem = (outputMemory M c s (epilogueInputs M c s AccPre LPre)).mem := by
  exact FPStages.epilogue_fp16 M AccPre LPre c s

specification epilogue_software_with_cast [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (toHalf : FP.Value .fp32 → FP.Value .fp32) (AccPre LPre : RegionName)
    (c : Config) (s : State (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (softwareEpilogue rest cfg toHalf) (epilogueSource AccPre LPre c) s = some final ∧
      final.mem = (roundedEpilogueMemory cfg toHalf AccPre LPre c s).mem := by
  exact FPStages.epilogue_software_with_cast rest cfg toHalf AccPre LPre c s

specification epilogue_nonempty_rounding [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (toHalf : FP.Value .fp32 → FP.Value .fp32) :
    ∃ final, FP.Structural.exec (softwareEpilogue rest .ieeeValue toHalf)
      (epilogueSource "acc" "den" epilogueWitnessConfig) epilogueWitnessInitial = some final ∧
      final.mem "out" 0 = .mk .fp16 (toHalf (FP.Value.ofBits .fp32 0x3eaaaaab)) ∧
      final.mem "out" 1 = epilogueWitnessInitial.mem "out" 1 ∧
      (11184811/33554432 : Rat) ≠ 1/3 := by
  exact FPStages.epilogue_nonempty_rounding rest toHalf

#axiomsClean forward_fp16
#axiomsClean forward_nonempty
#axiomsClean epilogue_fp16
#axiomsClean epilogue_software_with_cast
#axiomsClean epilogue_nonempty_rounding
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPCorrect
