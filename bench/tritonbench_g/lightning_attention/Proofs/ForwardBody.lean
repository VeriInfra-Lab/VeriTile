import bench.tritonbench_g.lightning_attention.Proofs.ForwardEntry
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem pair_eta (i : TileIndex [a,b]) : (i.1,i.2.1,PUnit.unit)=i := by
  rcases i with ⟨a,b,⟨⟩⟩
  rfl

end
end VeriTile.Bench.Repairs.LightningAttention
