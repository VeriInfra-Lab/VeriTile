import bench.tritonbench_g.rope_transform.IO
import bench.tritonbench_g.rope_transform.Proofs.IndexCSE
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RopeTransform.PairIO
open VeriTile.Triton FP.Structural
noncomputable section

def candidateInvocation {β : Type} (M : Algebra β) (r : Request) (row : Nat) : Invocation β :=
  { PublicIO.invocation M r row with kernel := IndexCSE.candidate r.parameters }

/-- The complete original grid, storage windows, encoding and observer are
retained. The invocation executes the entire readable index-CSE candidate. -/
def candidate {α β : Type} (M : Algebra β) (encode : α → β)
    (observe : β → α) (r : Request) : ProgramIO.Variant (PublicIO.sourceIO M encode observe r) where
  schedule := fun _ => (List.range r.rows).map (candidateInvocation M r)
  observe := fun _ _ => observe

def original {α β : Type} (M : Algebra β) (encode : α → β) (observe : β → α) (r : Request) :=
  ProgramIO.Implementation.original (PublicIO.sourceIO M encode observe r) (fun _ => True)
def indexShared {α β : Type} (M : Algebra β) (encode : α → β) (observe : β → α) (r : Request) :=
  ProgramIO.Implementation.ofVariant (candidate M encode observe r) (fun _ => True)

end
end VeriTile.Bench.Repairs.RopeTransform.PairIO
