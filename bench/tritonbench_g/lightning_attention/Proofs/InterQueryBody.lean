import bench.tritonbench_g.lightning_attention.Proofs.InterForwardEntry
import bench.tritonbench_g.lightning_attention.Proofs.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def querySubMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) : Memory α :=
  if block=0 then m else queryUpdate M D p head row carry m

theorem querySubMemory_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) :
    Inputs (querySubMemory M D p head row block carry m) m := by
  unfold querySubMemory
  split
  · exact Inputs.refl _
  · exact writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _

end
end VeriTile.Bench.Repairs.LightningAttention
