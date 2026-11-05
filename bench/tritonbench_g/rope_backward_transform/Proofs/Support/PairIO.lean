import bench.tritonbench_g.rope_backward_transform.IO
import bench.tritonbench_g.rope_backward_transform.Proofs.IndexCSE
import VeriTile.Triton.Float.ReadProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.PairIO
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete candidate invocation: only the source index-CSE program changes. -/
def candidateInvocation (op : Operations A) (r : Request) (row : Nat) : Invocation A :=
  { PublicIO.invocation op r row with kernel := IndexCSE.candidate (parameters r) }

/-- The public inputs, write footprint, odd-tail result reads and observer are
identical; the actual full row schedule executes the complete candidate source. -/
def candidate (op : Operations A) (project : A → A) (r : Request) :
    ReadProgramIO.Variant (PublicIO.sourceIO op project r) where
  schedule := fun _ => (KernelContract.launchRows r).reverse.map (candidateInvocation op r)
  observe := fun _ _ => project

def original (op : Operations A) (project : A → A) (r : Request) :=
  ReadProgramIO.Implementation.original (PublicIO.sourceIO op project r) (fun _ => True)
def indexShared (op : Operations A) (project : A → A) (r : Request) :=
  ReadProgramIO.Implementation.ofVariant (candidate op project r) (fun _ => True)

end
end VeriTile.Bench.Repairs.RopeBackwardTransform.PairIO
