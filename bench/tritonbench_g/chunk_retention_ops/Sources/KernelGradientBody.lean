import bench.tritonbench_g.chunk_retention_ops.Sources.KernelGradientEntry
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
open KernelBackward (Parameters PID Matrix)
open VeriTile.Bench.Repairs.ChunkLinearAttn (zero scalar cast add mul load rawLoad dot scatter scatter_frame transpose valueBlocks)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def gradientBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (heads : Nat) (input : Memory α) (s : State α) (vb : Nat) (a : GradAcc p α)
    (ctx : GradientReady p pid heads input s) (fixed : GradientFixed M D p pid input scale s)
    (ha : HasGrad p a s) :
    {t : State α // run (engine (withDot M D)) (gradientParts p scale).body
      (s.setReg "i_v" .nat [] (fun _ => vb))=some t} := by
  have hv := ctx.inputs "v" (by simp)
  have hg := ctx.inputs "incoming" (by simp)
  have hh := ctx.inputs "h" (by simp)
  have hd := ctx.inputs "dhistory" (by simp)
  refine ⟨?_,?_⟩
  swap
  simp [gradientParts,kernel,Source.backwardQKV,ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,transposeValues,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,ctx.ik,ctx.it,ctx.bh,ctx.nh,hv,hg,hh,hd,fixed.query,fixed.key,fixed.score,fixed.dq,fixed.dk,fixed.ds,
    ha.query,ha.key,ha.score,FP.Structural.store]
  rfl

theorem gradientBody_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (heads : Nat) (input : Memory α) (s : State α) (vb : Nat) (a : GradAcc p α)
    (ctx : GradientReady p pid heads input s) (fixed : GradientFixed M D p pid input scale s)
    (ha : HasGrad p a s) :
    (gradientBody M D p scale pid heads input s vb a ctx fixed ha).val.mem=
      gradValueWrite M p pid heads vb (gradValue M D p pid input scale vb) s.mem := by
  simp [gradientBody,gradValueWrite,scatter,masked_scatter_memory,gradValue,gradValueActive,gradValueOffset,
    gradHistoryGradient,valueTile,load,rawLoad,zero,VeriTile.Bench.Repairs.ChunkLinearAttn.cast,add,VeriTile.Bench.Repairs.ChunkLinearAttn.mul,dot,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  unfold VeriTile.Bench.Repairs.ChunkDeltaFwd.scatter
  unfold gradValueActive gradValueOffset
  unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
  have hb : Int.toNat (((pid.x:Int)*heads+pid.z)*((p.T:Int)*p.V))=
      (pid.x*heads+pid.z)*(p.T*p.V) := by norm_cast
  have historyBase : ((pid.z:Int)*((p.NT:Int)*p.K*p.V)+(pid.y:Int)*p.K*p.V).toNat =
      pid.z*(p.NT*p.K*p.V)+pid.y*p.K*p.V := by norm_cast
  have incomingBase : ((pid.z:Int)*((p.T:Int)*(p.V:Int))).toNat = pid.z*(p.T*p.V) := by norm_cast
  have matrix_eta {a b : Nat} (i : TileIndex [a,b]) : (i.1,i.2.1,PUnit.unit)=i := by
    rcases i with ⟨r,c,⟨⟩⟩
    rfl
  simp [zero,hb,historyBase,incomingBase,matrix_eta]

theorem gradientBody_values (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (heads : Nat) (input : Memory α) (s : State α) (vb : Nat) (a : GradAcc p α)
    (ctx : GradientReady p pid heads input s) (fixed : GradientFixed M D p pid input scale s)
    (ha : HasGrad p a s) :
    let t := (gradientBody M D p scale pid heads input s vb a ctx fixed ha).val
    GradientReady p pid heads input t ∧ GradientFixed M D p pid input scale t ∧
      HasGrad p (gradNext M D p pid input scale vb a) t := by
  dsimp only
  refine ⟨?_,?_,?_⟩
  · constructor
    · rw [gradientBody_memory]
      exact (gradValueWrite_inputs _ _ _ _ _ _ _).trans ctx.inputs
    · simpa [gradientBody] using ctx.ik
    · simpa [gradientBody] using ctx.it
    · simpa [gradientBody] using ctx.bh
    · simpa [gradientBody] using ctx.nh
  · constructor
    · simpa [gradientBody] using fixed.dq
    · simpa [gradientBody] using fixed.dk
    · simpa [gradientBody] using fixed.ds
    · simpa [gradientBody] using fixed.query
    · simpa [gradientBody] using fixed.key
    · simpa [gradientBody] using fixed.score
  · have historyBase : ((pid.z:Int)*((p.NT:Int)*p.K*p.V)+(pid.y:Int)*p.K*p.V).toNat =
        pid.z*(p.NT*p.K*p.V)+pid.y*p.K*p.V := by norm_cast
    have stateBase : ((pid.z:Int)*((p.NT:Int)*p.K*p.V)+((pid.y:Int)+1)*p.K*p.V).toNat =
        pid.z*(p.NT*p.K*p.V)+(pid.y+1)*p.K*p.V := by norm_cast
    constructor <;>
      simp [gradientBody,gradNext,gradHistory,gradHistoryGradient,valueTile,transpose,load,rawLoad,
        zero,historyBase,stateBase,VeriTile.Bench.Repairs.ChunkLinearAttn.cast,add,VeriTile.Bench.Repairs.ChunkLinearAttn.mul,scalar,dot,transposeValues,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      funext i
      unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
      simp [zero,transpose,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      try unfold VeriTile.Bench.Repairs.ChunkLinearAttn.transpose
      rfl

#axiomsClean gradientBody_memory
#axiomsClean gradientBody_values
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelGradient
