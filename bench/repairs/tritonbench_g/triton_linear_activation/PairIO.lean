import bench.repairs.tritonbench_g.triton_linear_activation.IO
import bench.repairs.tritonbench_g.triton_linear_activation.CandidateKernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.TritonLinearActivation.Pair
open VeriTile Triton FP.Structural KernelSource KernelValues
noncomputable section

def invocation {α : Type} (M : Algebra α) (D : Dot α) (mode : Mode) (p : Parameters) (pid : Nat) : Invocation α :=
  { LinearIO.invocation M D mode p pid with kernel := CandidateSource.source mode p }

def variant {α β : Type} (M : Algebra β) (D : Dot β) (encode : α → β) (observe : β → α)
    (mode : Mode) (p : Parameters) (pid : Nat) :
    ProgramIO.Variant (LinearIO.programIO M D encode observe mode p pid) where
  schedule := fun _ => [invocation M D mode p pid]
  observe := fun _ _ => observe

end
end VeriTile.Bench.Repairs.TritonLinearActivation.Pair
