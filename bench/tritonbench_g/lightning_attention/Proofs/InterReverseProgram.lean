import bench.tritonbench_g.lightning_attention.Proofs.InterReverseLoops
import bench.tritonbench_g.lightning_attention.Proofs.InterForwardProgram
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem InterRReady.set_i (p : Parameters) (head row1 row2 block : Nat)
    (input : Memory α) (carry : Matrix α p.D p.E) (s : State α)
    (ctx : InterRReady p head row1 row2 input carry s) :
    InterRReady p head row1 row2 input carry (s.setReg "reverse_i" .nat [] (fun _ => block)) := by
  constructor
  · exact ctx.inputs
  all_goals simp [State.setReg,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.dk,ctx.dv,ctx.rows1,ctx.rows2,ctx.total]

end
end VeriTile.Bench.Repairs.LightningAttention
