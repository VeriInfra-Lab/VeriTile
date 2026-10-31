import bench.tritonbench_g.lightning_attention.Proofs.InterForwardEntry
import bench.tritonbench_g.lightning_attention.Proofs.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def querySubDelta (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (input : Memory α) (acc : Matrix α p.E p.D) : Matrix α p.E p.D :=
  fun i => add M (acc i) (dot D (subVT M p head row input) (subK M p head row input) i)

end
end VeriTile.Bench.Repairs.LightningAttention
