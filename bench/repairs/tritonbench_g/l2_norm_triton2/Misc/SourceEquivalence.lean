import bench.repairs.tritonbench_g.l2_norm_triton2.Kernels
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Kernel
import bench.repairs.tritonbench_g.l2_norm_triton1.Misc.SourceEquivalence

namespace VeriTile.Bench.Repairs.L2NormTriton2.SourceEquivalence
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev Program := L2NormTriton1.SourceEquivalence.Program
abbrev Equivalent := L2NormTriton1.SourceEquivalence.Equivalent

def originalForward (N B : Nat) : Program := ⟨forward N B⟩
def candidateForward (N B : Nat) : Program := ⟨forwardCommuted N B⟩
def originalBackward (N B : Nat) : Program := ⟨backward N B⟩
def candidateBackward (N B : Nat) : Program := ⟨backwardCommuted N B⟩

theorem forward_equivalence (N B : Nat) :
    originalForward N B ≡[([] : Spec.Assumptions ComputeStmt)] candidateForward N B :=
  L2NormTriton1.SourceEquivalence.source_equivalence N B

/-- Successful complete executions yield exactly the same full state. Only
integer row-base products change, so no numerical assumption is needed for
NaNs, infinities, signed zeros, overflow, or any runtime FP32 epsilon word. -/
theorem backward_equivalence (N B : Nat) :
    originalBackward N B ≡[([] : Spec.Assumptions ComputeStmt)] candidateBackward N B := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent) rfl
  · rfl
  intro α _ A row eps before
  obtain ⟨after,success⟩ := Kernel.executes A N B row eps before
  exact ⟨after,success,(Kernel.commuted_execution A N B row eps before).trans success⟩

#print_fp_assumptions forward_equivalence
#print_fp_assumptions backward_equivalence
#axiomsClean forward_equivalence
#axiomsClean backward_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.SourceEquivalence
