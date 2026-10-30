import bench.tritonbench_g.chunk_delta_fwd.Sources.OuterEdges
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def finishState (M : Algebra α) (p : Parameters) (pid : PID) (s : State α)
    (h : HValues p α) : State α :=
  if p.storeFinal then
    let ptr := BlockPtr.mk "final" (pid.2*p.K*p.V) [p.K,p.V] [p.BK,p.BV]
      [p.V,1] [0,pid.1*p.BV]
    let base := s.setReg "p_ht" .blockPtr [p.BK,p.BV] (fun _ => ptr)
    (TileShape.allIndices [p.BK,p.BV]).foldl (fun t i =>
      if hActive p pid i then t.write "final" (fOffset p pid i) (.mk .real (cast M (h i))) else t) base
  else s

theorem finish_run (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (pid : PID) (input : Memory α) (s : State α) (h : HValues p α)
    (ctx : Ready pid input s) (hh : HasH p h s) :
    run (engine (withDot M D)) (outer p).after s=some (finishState M p pid s h) := by
  simp only [HasH] at hh
  cases hp : p.storeFinal <;>
    simp [outer,kernel,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      ctx.ik,ctx.iv,ctx.bh,hh,hp,finishState,hActive,fOffset,column,cast]

theorem finish_memory (M : Algebra α) (p : Parameters) (pid : PID) (s : State α)
    (h : HValues p α) : (finishState M p pid s h).mem=writeFinal M p pid h s.mem := by
  cases hp : p.storeFinal <;>
    simp [finishState,writeFinal,hp,masked_scatter_memory,scatter]

theorem finish_ready (M : Algebra α) (p : Parameters) (pid : PID) (input : Memory α)
    (s : State α) (h : HValues p α) (ctx : Ready pid input s) :
    Ready pid input (finishState M p pid s h) := by
  constructor
  · rw [finish_memory]
    exact (final_inputs _ _ _ _ _).trans ctx.inputs
  · cases hp : p.storeFinal <;> simpa [finishState,hp] using ctx.ik
  · cases hp : p.storeFinal <;> simpa [finishState,hp] using ctx.iv
  · cases hp : p.storeFinal <;> simpa [finishState,hp] using ctx.bh

#axiomsClean finish_run
#axiomsClean finish_memory
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
