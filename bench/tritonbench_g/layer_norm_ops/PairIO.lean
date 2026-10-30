import bench.tritonbench_g.layer_norm_ops.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LayerNormOps.PairIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Same complete forward IO; the candidate reuses its column mask. -/
def forwardCandidate {α : Type} (M : Algebra α) (p : ForwardParameters) (pid : Nat) :
    ProgramIO.Variant (Interface.forwardIO M p pid) where
  schedule := fun _ => [Interface.invocation M (forwardMaskReusedKernel p) pid]
  observe := fun _ _ => id

/-- Same bounded backward row block and optional IO; reuse the scalar row start. -/
def backwardCandidate {α : Type} (M : Algebra α) (p : BackwardParameters) (pid : Nat) :
    ProgramIO.Variant (Interface.backwardIO M p pid) where
  schedule := fun _ => [Interface.invocation M (backwardRowStartReusedKernel p) pid]
  observe := fun _ _ => id

def forwardOriginal {α : Type} (M : Algebra α) (p : ForwardParameters) (pid : Nat) :=
  ProgramIO.Implementation.original (Interface.forwardIO M p pid) (fun _ => True)
def forwardMaskReused {α : Type} (M : Algebra α) (p : ForwardParameters) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (forwardCandidate M p pid) (fun _ => True)
def backwardOriginal {α : Type} (M : Algebra α) (p : BackwardParameters) (pid : Nat) :=
  ProgramIO.Implementation.original (Interface.backwardIO M p pid) (fun _ => True)
def backwardRowStartReused {α : Type} (M : Algebra α) (p : BackwardParameters) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (backwardCandidate M p pid) (fun _ => True)

end
end VeriTile.Bench.Repairs.LayerNormOps.PairIO
