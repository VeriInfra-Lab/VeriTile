import bench.tritonbench_g.lightning_attention.Proofs.InterForwardLoops
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem InterFReady.set_i (p : Parameters) (head row1 row2 block : Nat)
    (input : Memory α) (carry : Matrix α p.E p.D) (s : State α)
    (ctx : InterFReady p head row1 row2 input carry s) :
    InterFReady p head row1 row2 input carry (s.setReg "i" .nat [] (fun _ => block)) := by
  constructor
  · exact ctx.inputs
  all_goals simp [State.setReg,ctx.qkOffset,ctx.vOffset,ctx.oOffset,ctx.query,ctx.incoming,ctx.key,ctx.value,ctx.rows1,ctx.rows2,ctx.total]

theorem run_single_range (M : Algebra α) (idx : RegName) (stop : Nat) (body : List ComputeStmt)
    (s t : State α) (hr : FP.Structural.range M idx 0 stop 1 body s=some t) :
    run M [.forRangeDyn idx (.constNat 0) (.constNat stop) (.constNat 1) body] s=some t := by
  simp only [run,step,evalOp_unfold,bind,Option.bind,hr]

end
end VeriTile.Bench.Repairs.LightningAttention
