import bench.repairs.tritonbench_g.chunk_delta_fwd.Entry
import bench.repairs.tritonbench_g.chunk_delta_fwd.Memory
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def innerExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c b : Nat)
    (frozen acc : HValues p α) (ctx : Ready pid input current)
    (hh : HasH p frozen current) (ha : HasAcc p acc current)
    (ht : current.regs .nat [] "i_t"=some (fun _ => c)) :
    {t : State α // run (engine (withDot M D)) (inner p).body
      (current.setReg "i_c" .nat [] (fun _ => b))=some t} := by
  have hkey := ctx.inputs "key" (by simp)
  have hw := ctx.inputs "weight" (by simp)
  have hv := ctx.inputs "value" (by simp)
  simp only [HasH] at hh
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [inner,outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
    ctx.ik,ctx.iv,ctx.bh,hh,ha,ht,hkey,hw,hv]
  rfl

theorem inner_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c b : Nat)
    (frozen acc : HValues p α) (ctx : Ready pid input current)
    (hh : HasH p frozen current) (ha : HasAcc p acc current)
    (ht : current.regs .nat [] "i_t"=some (fun _ => c)) :
    (innerExecution M D p pid input current c b frozen acc ctx hh ha ht).val.mem=
      writeCorrected M p pid c b (corrected M D p pid input c b frozen) current.mem := by
  simp [innerExecution,writeCorrected,scatter,masked_scatter_memory,cActive,cOffset,
    corrected,valueLoad,weightLoad,zero,cast,time,column,weightBase,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem inner_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c b : Nat)
    (frozen acc : HValues p α) (ctx : Ready pid input current)
    (hh : HasH p frozen current) (ha : HasAcc p acc current)
    (ht : current.regs .nat [] "i_t"=some (fun _ => c)) :
    let t := (innerExecution M D p pid input current c b frozen acc ctx hh ha ht).val
    Ready pid input t ∧ HasH p frozen t ∧
      HasAcc p (accumulatedStep M D p pid input c b frozen acc) t ∧
      t.regs .nat [] "i_t"=some (fun _ => c) := by
  dsimp only
  refine ⟨?_,?_,?_,?_⟩
  · constructor
    · rw [inner_memory]
      exact (corrected_inputs _ _ _ _ _ _ _).trans ctx.inputs
    · simpa [innerExecution] using ctx.ik
    · simpa [innerExecution] using ctx.iv
    · simpa [innerExecution] using ctx.bh
  · simpa [HasH,innerExecution] using hh
  · simp [HasAcc,innerExecution,accumulatedStep,keyLoad,corrected,valueLoad,weightLoad,
      zero,cast,time,column,weightBase,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    rfl
  · simpa [innerExecution] using ht

#axiomsClean inner_memory
#axiomsClean inner_values
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
