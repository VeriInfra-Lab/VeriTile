import bench.tritonbench_g.cross_entropy_ops.IO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.CrossEntropyOps.PairIO
open VeriTile.Triton FP.Structural Interface
noncomputable section

def candidateInvocation {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) (backward : Bool) : Invocation α :=
  { invocation M p args backward row block with
    kernel := if backward then backwardIndexCommutedSyntax p else forwardIndexCommutedSyntax p }

def forwardCandidate {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :
    ProgramIO.Variant (Forward.programIO M p args row block) where
  schedule := fun _ => [candidateInvocation M p row block args Bool.false]
  observe := fun _ _ => id

def backwardCandidate {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :
    ProgramIO.Variant (Backward.programIO M p args row block) where
  schedule := fun _ => [candidateInvocation M p row block args Bool.true]
  observe := fun _ _ => id

def forwardOriginal {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.original (Forward.programIO M p args row block) (fun _ => True)
def forwardIndexCommuted {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.ofVariant (forwardCandidate M p row block args) (fun _ => True)
def backwardOriginal {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.original (Backward.programIO M p args row block) (fun _ => True)
def backwardIndexCommuted {α : Type} (M : Algebra α) (p : Parameters)
    (row block : Nat) (args : RuntimeArgs.Values α) :=
  ProgramIO.Implementation.ofVariant (backwardCandidate M p row block args) (fun _ => True)
end
end VeriTile.Bench.Repairs.CrossEntropyOps.PairIO
