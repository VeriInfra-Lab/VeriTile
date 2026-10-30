import bench.tritonbench_g.cross_entropy2.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.CrossEntropy2.PairIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Same public inputs, scalar ABI, output windows and observer; only the
integer column product in the actual complete invocation changes. -/
def forwardCandidate {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :
    ProgramIO.Variant (IO.forward M p row block args id id) where
  schedule := fun _ => [IO.launch M p (forwardIndexCommutedSyntax p) row block args]
  observe := fun _ _ => id

def backwardCandidate {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :
    ProgramIO.Variant (IO.backward M p row block args id id) where
  schedule := fun _ => [IO.launch M p (backwardIndexCommutedSyntax p) row block args]
  observe := fun _ _ => id

def forwardOriginal {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.original (IO.forward M p row block args id id) (fun _ => True)
def forwardIndexCommuted {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.ofVariant (forwardCandidate M p row block args) (fun _ => True)
def backwardOriginal {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.original (IO.backward M p row block args id id) (fun _ => True)
def backwardIndexCommuted {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.ofVariant (backwardCandidate M p row block args) (fun _ => True)

end
end VeriTile.Bench.Repairs.CrossEntropy2.PairIO
