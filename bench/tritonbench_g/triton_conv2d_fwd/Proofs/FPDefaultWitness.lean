import bench.tritonbench_g.triton_conv2d_fwd.BranchesCorrect
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPWitness

namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The existing nonempty 2x2x2 witness has eight source updates. Default fp32
operands preserve the low bit that the half-cast branch rounds away. -/
theorem default_multiblock_value :
    bitOutput FPValue.witnessParameters (FPValue.witnessState (FP.Value.zero .fp32))
      FPValue.lane = FP.Value.ofBits .fp32 0x41001000 := by
  decide +kernel

theorem fp16_flags_have_different_values :
    bitOutput FPValue.witnessParameters (FPValue.witnessState (FP.Value.zero .fp32)) FPValue.lane ≠
      FPValue.bitOutput FPValue.witnessParameters (FPValue.witnessState (FP.Value.zero .fp32)) FPValue.lane := by
  rw [default_multiblock_value, FPValue.multiblock_value]
  decide

/-- Actual full default-branch execution, with a concrete active output. This
witness supports the arbitrary-shape theorem; it does not restrict that theorem. -/
theorem default_multiblock_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend (software rest) sequentialDot)
      (source FPValue.witnessParameters) (FPValue.witnessState (FP.Value.zero .fp32)) = some final ∧
      final.mem "O" 0 = .mk .fp32 (FP.Value.ofBits .fp32 0x41001000) := by
  obtain ⟨final, he, hv, _⟩ := software_correctness rest FPValue.witnessParameters
    (FPValue.witnessState (FP.Value.zero .fp32)) (FPValue.witness_injective _)
  have output := hv FPValue.lane (by decide)
  rw [default_multiblock_value] at output
  exact ⟨final, he, output⟩

#axiomsClean default_multiblock_value
#axiomsClean fp16_flags_have_different_values
#axiomsClean default_multiblock_execution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.FPDefault
