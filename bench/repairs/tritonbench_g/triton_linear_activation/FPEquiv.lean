import bench.repairs.tritonbench_g.triton_linear_activation.Proofs.Pair

namespace VeriTile.Bench.Repairs.TritonLinearActivation.FPEquiv
open VeriTile Triton FP.Structural KernelSource KernelContract KernelValues
open scoped VeriTile.Spec
noncomputable section

def original {α β : Type} (M : Algebra β) (D : Dot β) (encode : α → β) (observe : β → α)
    (mode : Mode) (p : Parameters) (pid : Nat) :=
  ProgramIO.Implementation.original (LinearIO.programIO M D encode observe mode p pid) (fun _ => True)

def shifted {α β : Type} (M : Algebra β) (D : Dot β) (encode : α → β) (observe : β → α)
    (mode : Mode) (p : Parameters) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (Pair.variant M D encode observe mode p pid) (fun _ => True)

/-- One complete source invocation, including both storage modes, all activation,
bias and saved-output branches and the complete descending K loop. The candidate
uses exact integer shift strength reduction at GROUP_M=8, with the original
formula for every other GROUP_M. No numerical rule or finite-value guard is used. -/
specification kernel_equiv {α β : Type} [Inhabited β] (M : Algebra β) (D : Dot β)
    (encode : α → β) (observe : β → α) (mode : Mode) (p : Parameters) (pid : Nat)
    (positive : 0 < p.blockInner) (strides : CanonicalStrides p) (separate : p.saved ≠ p.output) :
    original M D encode observe mode p pid ≡[[]] shifted M D encode observe mode p pid :=
  ProgramIO.Implementation.of_equivalent (Pair.variant M D encode observe mode p pid)
    (Pair.equivalent M D encode observe mode p pid positive strides separate)

#axiomsClean kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.TritonLinearActivation.FPEquiv
