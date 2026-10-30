import bench.tritonbench_g.chunk_delta_fwd.Sources.Inner
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def historyExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current) (hh : HasH p frozen current) :
    {t : State α // run (engine (withDot M D)) (inner p).before
      (current.setReg "i_t" .nat [] (fun _ => c))=some t} := by
  simp only [HasH] at hh
  refine ⟨?_,?_⟩
  swap
  simp [inner,outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
    ctx.ik,ctx.iv,ctx.bh,hh]
  rfl

theorem history_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current) (hh : HasH p frozen current) :
    (historyExecution M D p pid input current c frozen ctx hh).val.mem=
      writeHistory M p pid c frozen current.mem := by
  simp [historyExecution,writeHistory,scatter,masked_scatter_memory,hActive,hOffset,
    cast,column,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

theorem history_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (current : State α) (c : Nat)
    (frozen : HValues p α) (ctx : Ready pid input current) (hh : HasH p frozen current) :
    let t := (historyExecution M D p pid input current c frozen ctx hh).val
    Ready pid input t ∧ HasH p frozen t ∧ HasAcc p (fun _ => zero M) t ∧
      t.regs .nat [] "i_t"=some (fun _ => c) := by
  dsimp only
  refine ⟨?_,?_,?_,?_⟩
  · constructor
    · rw [history_memory]
      exact (history_inputs _ _ _ _ _ _).trans ctx.inputs
    · simpa [historyExecution] using ctx.ik
    · simpa [historyExecution] using ctx.iv
    · simpa [historyExecution] using ctx.bh
  · simpa [HasH,historyExecution] using hh
  · simp [HasAcc,historyExecution,zero]
  · simp [historyExecution]

def updateState (M : Algebra α) (p : Parameters) (s : State α)
    (h acc : HValues p α) :=
  s.setReg "b_h" .real [p.BK,p.BV]
    (fun i => M.binary (some .fp32) .real .add (h i) (acc i))

theorem update_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (s : State α) (h acc : HValues p α) (hh : HasH p h s) (ha : HasAcc p acc s) :
    run (engine (withDot M D)) (inner p).after s=some (updateState M p s h acc) := by
  simp only [HasH] at hh
  simp only [HasAcc] at ha
  simp [inner,outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,hh,ha,
    updateState,numeric,ofFloat,toFloat,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem update_ready (M : Algebra α) (p : Parameters) (pid : PID) (input : Memory α)
    (s : State α) (h acc : HValues p α) (ctx : Ready pid input s) :
    Ready pid input (updateState M p s h acc) := by
  constructor
  · exact ctx.inputs
  · simpa [updateState] using ctx.ik
  · simpa [updateState] using ctx.iv
  · simpa [updateState] using ctx.bh

#axiomsClean history_memory
#axiomsClean history_values
#axiomsClean update_run
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
