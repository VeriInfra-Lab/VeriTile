import bench.repairs.tritonbench_g.chunk_linear_attn.OutputModel
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure OutputReady (p : Parameters) (pid : PID) (input : Memory α) (s : State α) : Prop where
  memory : s.mem=input
  iv : s.regs .nat [] "i_v"=some (fun _ => pid.x)
  it : s.regs .nat [] "i_t"=some (fun _ => pid.y)
  bh : s.regs .nat [] "i_bh"=some (fun _ => pid.z)
  mask : s.regs .bool [p.BT,p.BT] "m_s"=some (fun i => decide (i.2.1.val ≤ i.1.val))
def HasOutput (p : Parameters) (a : OutputAcc p α) (s : State α) :=
  s.regs .real [p.BT,p.BV] "b_o"=some a.output ∧ s.regs .real [p.BT,p.BT] "b_s"=some a.score

def outputEntry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    {t : State α // run (engine (withDot M D)) (outputParts p scale).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  rfl

theorem outputEntry_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ) (s : State α) :
    OutputReady p ⟨s.pids 0,s.pids 1,s.pids 2⟩ s.mem (outputEntry M D p scale s).val ∧
    HasOutput p (outputInitial M p) (outputEntry M D p scale s).val := by
  constructor
  · constructor <;> simp [outputEntry] <;> rfl
  · simp [HasOutput,outputInitial,outputEntry,zero]

def outputBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α) (kb : Nat) (a : OutputAcc p α)
    (ctx : OutputReady p pid input s) (ha : HasOutput p a s) :
    {t : State α // run (engine (withDot M D)) (outputParts p scale).body
      (s.setReg "i_k" .nat [] (fun _ => kb))=some t} := by
  have ho := ha.1
  have hs := ha.2
  refine ⟨?_,?_⟩
  swap
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,ctx.iv,ctx.it,ctx.bh,ctx.memory,ho,hs]
  rfl

theorem outputBody_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α) (kb : Nat) (a : OutputAcc p α)
    (ctx : OutputReady p pid input s) (ha : HasOutput p a s) :
    let t := (outputBody M D p scale pid input s kb a ctx ha).val
    OutputReady p pid input t ∧ HasOutput p (outputNext M D p pid input kb a) t := by
  dsimp only
  constructor
  · constructor
    · simp [outputBody]
    · simpa [outputBody] using ctx.iv
    · simpa [outputBody] using ctx.it
    · simpa [outputBody] using ctx.bh
    · simpa [outputBody] using ctx.mask
  · simp [HasOutput,outputBody,outputNext,queryTile,forwardHistoryTile,keyTile,load,rawLoad,
      zero,add,dot,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    constructor <;> funext i <;>
      simp [outputNext,queryTile,forwardHistoryTile,keyTile,load,rawLoad,zero,add,dot,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
      simp [zero]

def outputFinish (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α) (a : OutputAcc p α)
    (ctx : OutputReady p pid input s) (ha : HasOutput p a s) :
    {t : State α // run (engine (withDot M D)) (outputParts p scale).after s=some t} := by
  have ho := ha.1
  have hs := ha.2
  refine ⟨?_,?_⟩
  swap
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,ctx.iv,ctx.it,ctx.bh,ctx.mask,ctx.memory,ho,hs,FP.Structural.store]
  rfl

theorem outputFinish_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (input : Memory α) (s : State α) (a : OutputAcc p α)
    (ctx : OutputReady p pid input s) (ha : HasOutput p a s) :
    (outputFinish M D p scale pid input s a ctx ha).val.mem=
      outputWrite M p pid (outputResult M D p pid input scale a) input := by
  simp [outputFinish,outputWrite,scatter,masked_scatter_memory,outputResult,outputActive,outputOffset,
    valueTile,load,rawLoad,zero,cast,add,mul,scalar,dot,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  dsimp [VeriTile.Bench.Repairs.ChunkDeltaFwd.scatter,outputActive,outputOffset]
  unfold VeriTile.Bench.Repairs.ChunkLinearAttn.load VeriTile.Bench.Repairs.ChunkLinearAttn.rawLoad
  simp [zero]

#axiomsClean outputEntry_ready
#axiomsClean outputBody_ready
#axiomsClean outputFinish_memory
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
