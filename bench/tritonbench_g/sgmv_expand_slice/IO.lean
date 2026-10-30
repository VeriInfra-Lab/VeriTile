import bench.tritonbench_g.sgmv_expand_slice.Proofs.IO
import bench.tritonbench_g.sgmv_expand_slice.Candidate
import VeriTile.Triton.Float.ReadProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
open VeriTile Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (M : Algebra α) (a : SourceABI) (p : KernelArgs) (j : Job) : Invocation α :=
  { PublicIO.invocation M a p j with kernel := IndexReuse.kernel a {p with batch := j.batch} }

def variant (M : Algebra α) (a : SourceABI) (p : KernelArgs) (shape : Shape) (segments : List Segment) :
    ReadProgramIO.Variant (PublicIO.io M a p shape segments) where
  schedule := fun _ => (jobs shape.columns segments).map (invocation M a p)
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexPair
