import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardUpdate
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkLinearAttn (cast scatter scatter_frame hActive fOffset)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def writeInitial (M : Algebra α) (p : Parameters) (pid : PID) (initial : Bool)
    (carry : Carry p α) (memory : Memory α) : Memory α :=
  if initial then scatter (TileShape.allIndices [p.BK,p.BV]) "initialGradient" (hActive p pid)
    (fOffset p pid) (fun i=>.mk .real (cast M (carry i))) memory else memory

theorem initial_inputs (M : Algebra α) (p : Parameters) (pid : PID) (initial : Bool)
    (carry : Carry p α) (memory : Memory α) : InputEq (writeInitial M p pid initial carry memory) memory := by
  cases initial
  · intro r hr o; rfl
  · intro r hr o
    apply scatter_frame
    left
    intro he
    subst r
    simp at hr

def finishExecution (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s) :
    {t : State α // run (engine (withDot M D)) (parts p scale terminal initial).after s=some t} := by
  cases initial
  all_goals
    simp only [HasCarry] at hc
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,Source.backwardH,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,
      ctx.ik,ctx.iv,ctx.bh,hc]
    rfl

theorem finish_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (terminal initial : Bool) (pid : PID) (input : Memory α) (s : State α)
    (carry : Carry p α) (ctx : Ready M p pid input s) (hc : HasCarry p carry s) :
    (finishExecution M D p scale terminal initial pid input s carry ctx hc).val.mem=
      writeInitial M p pid initial carry s.mem := by
  cases initial <;>
    simp [finishExecution,writeInitial,scatter,masked_scatter_memory,hActive,fOffset,
      VeriTile.Bench.Repairs.ChunkLinearAttn.cast,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

#axiomsClean finish_memory
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
