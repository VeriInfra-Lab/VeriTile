import bench.tritonbench_g.lightning_attention.Proofs.InterReverseEntry
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
def keyValueSubMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) : Memory α :=
  if block=0 then m else keyValueUpdate M D p head row carry m

theorem keyValueSubMemory_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) :
    Inputs (keyValueSubMemory M D p head row block carry m) m := by
  unfold keyValueSubMemory
  split
  · exact Inputs.refl _
  · exact (writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _).trans
      (writeTrans_inputs _ _ (by decide) _ _ _ _ _ _ _)

end
end VeriTile.Bench.Repairs.LightningAttention
