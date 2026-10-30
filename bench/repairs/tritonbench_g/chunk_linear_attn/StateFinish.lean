import bench.repairs.tritonbench_g.chunk_linear_attn.StateBody
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def stateFinish (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) :
    {t : State α // run (engine (withDot M D)) (stateParts p scale rev seed final).after s=some t} := by
  cases rev <;> cases final
  all_goals
    simp only [HasCarry,carryName,Bool.false_eq_true,if_false,if_true] at hh
    refine ⟨?_,?_⟩
    swap
    simp [stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      ctx.ik,ctx.iv,ctx.bh,hh]
    rfl

theorem stateFinish_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input : Memory α) (s : State α)
    (h : HValues p α) (ctx : StateReady pid input s) (hh : HasCarry p rev h s) :
    (stateFinish M D p scale rev seed final pid input s h ctx hh).val.mem=
      writeStateFinal M p pid rev final h s.mem := by
  cases rev <;> cases final <;>
    simp [stateFinish,writeStateFinal,scatter,masked_scatter_memory,hActive,fOffset,finalName,
      cast,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

#axiomsClean stateFinish_memory
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
