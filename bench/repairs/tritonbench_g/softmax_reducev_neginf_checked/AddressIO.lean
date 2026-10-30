import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.IO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Candidate
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.AddressIO
open VeriTile Triton FP.Structural
noncomputable section
variable {A B : Type} [Inhabited A] [Inhabited B]

def invocation (M : Algebra B) (c : Host.Call) (j : Host.Job) : Invocation B :=
  { PublicIO.invocation M c j with kernel := EarlyAddress.source (Host.parameters c) }

def variant (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
    (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
    (counts : Fin c.batch → Nat) (work : List Host.Job) :
    ProgramIO.Variant (PublicIO.sourceIO M encode c observe counts work) where
  schedule := fun _ => work.map (fun j => invocation (M j) c j)
  observe := (PublicIO.sourceIO M encode c observe counts work).observe

end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.AddressIO
