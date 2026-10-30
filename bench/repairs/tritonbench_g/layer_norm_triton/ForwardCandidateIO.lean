import bench.repairs.tritonbench_g.layer_norm_triton.IO
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceCandidate

namespace VeriTile.Bench.Repairs.LayerNormTriton.ForwardEquivalenceIO
open VeriTile.Triton FP.Structural SourceModel Interface
noncomputable section
variable {α : Type} [Inhabited α]

def candidate (m : Model α) (stride N B row : Nat) (eps : ℝ)
    (observe : RegionName → Nat → α → α) :
    ProgramIO.Variant (Forward.io m stride N B row eps observe) where
  schedule := fun _ => [job m (SourceCandidate.forward stride N B eps) row]
  observe := (Forward.io m stride N B row eps observe).observe

end
end VeriTile.Bench.Repairs.LayerNormTriton.ForwardEquivalenceIO
