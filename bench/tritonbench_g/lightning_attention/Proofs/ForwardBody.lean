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
def fwdBodyExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (s : State α) (block : Nat) (carry : Matrix α p.D p.BM)
    (ctx : FwdReady p pid input s) (rows : FwdRows p block s) (hc : FwdCarry p carry s) :
    {t : State α // run (engine (withDot M D)) (fwdParts p).body
      (s.setReg "i" .nat [] (fun _ => block))=some t} := by
  have hq := ctx.inputs "query" (by simp)
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  simp only [FwdRows] at rows
  simp only [FwdCarry] at hc
  refine ⟨?_,?_⟩
  swap
  simp [fwdParts,forward,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
    ctx.query,ctx.key,ctx.value,ctx.output,ctx.causal,rows,hc,hq,hk,hv]
  rfl

theorem fwdBody_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (s : State α) (block : Nat) (carry : Matrix α p.D p.BM)
    (ctx : FwdReady p pid input s) (rows : FwdRows p block s) (hc : FwdCarry p carry s) :
    (fwdBodyExecution M D p pid input s block carry ctx rows hc).val.mem=
      writeOutput M p pid block (fwdOutput M D p pid input block carry) s.mem := by
  unfold writeOutput scatter ChunkDeltaFwd.scatter fwdOutput fwdScore fwdQ fwdK fwdV
  simp [fwdBodyExecution,masked_scatter_memory,pair_eta,
    fwdActive,fwdOffset,read,zero,add,cast,dot,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Nat.add_assoc]
  norm_num

theorem fwdBody_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input : Memory α) (s : State α) (block : Nat) (carry : Matrix α p.D p.BM)
    (ctx : FwdReady p pid input s) (rows : FwdRows p block s) (hc : FwdCarry p carry s) :
    let t := (fwdBodyExecution M D p pid input s block carry ctx rows hc).val
    FwdReady p pid input t ∧ FwdRows p (block+1) t ∧
      FwdCarry p (fwdNext M D p pid input block carry) t := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · rw [fwdBody_memory]
      exact (writeOutput_inputs _ _ _ _ _ _).trans ctx.inputs
    · simpa [fwdBodyExecution] using ctx.query
    · simpa [fwdBodyExecution] using ctx.key
    · simpa [fwdBodyExecution] using ctx.value
    · simpa [fwdBodyExecution] using ctx.output
    · simpa [fwdBodyExecution] using ctx.causal
  · simp [FwdRows,fwdBodyExecution,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    funext i
    simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  · simp [FwdCarry,fwdBodyExecution,fwdNext,fwdK,fwdV,read,zero,add,cast,dot,
      FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Nat.add_assoc]
    unfold fwdK fwdV
    funext i
    simp [read,zero,cast,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Nat.add_assoc,pair_eta]
    norm_num
#axiomsClean fwdBody_memory
#axiomsClean fwdBody_ready
end
end VeriTile.Bench.Repairs.LightningAttention
