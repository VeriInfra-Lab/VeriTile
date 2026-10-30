import bench.tritonbench_g.chunk_gla_simple.Proofs.FPValue

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

structure Context {α : Type} (c : Shape) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  bh : current.regs .nat [] "i_bh"=some (fun _ => initial.pids 2)
  it : current.regs .nat [] "i_t"=some (fun _ => initial.pids 1)
  rows : current.regs .nat [c.BT] "rows"=some (fun i => row c initial i.1)
  cols : current.regs .nat [c.BV] "cols"=some (fun i => column c initial i.1)
  mask : current.regs .bool [c.BT,c.BT] "m_s"=some (fun i => decide (i.2.1.val ≤ i.1.val))

def HasAcc {α : Type} (c : Shape) (acc : Acc α c) (s : State α) : Prop :=
  s.regs .real [c.BT,c.BV] "b_o"=some acc.1 ∧ s.regs .real [c.BT,c.BT] "b_s"=some acc.2

def entryExecution {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α) :
    {t : State α // run (engine M) (Typed.parts c scale).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [Typed.parts,kernel,ChunkGlaSimpleMinDotTile.source,fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem entryContext {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α) :
    Context c s (entryExecution M c scale s).val ∧
      HasAcc c (fun _ => zero M,fun _ => zero M) (entryExecution M c scale s).val := by
  constructor
  · constructor <;> simp [entryExecution,row,column,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex] <;> rfl
  · simp [HasAcc,entryExecution,zero]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial current : State α) (block : Nat) (acc : Acc α c)
    (ctx : Context c initial current) (ha : HasAcc c acc current) :
    {t : State α // run (engine M) (Typed.parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => block))=some t} := by
  have ho := ha.1
  have hs := ha.2
  refine ⟨?_,?_⟩
  swap
  simp [Typed.parts,kernel,ChunkGlaSimpleMinDotTile.source,fp32Original,ComputeKernel.surfaceBody,Source.dynamicRange?,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.bh,ctx.it,ctx.rows,ctx.cols,ctx.mask,ctx.memory,ho,hs,d.qh_eq,d.qk_eq]
  rfl

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial current : State α) (block : Nat) (acc : Acc α c)
    (ctx : Context c initial current) (ha : HasAcc c acc current) :
    let t := (bodyExecution M c scale d initial current block acc ctx ha).val
    Context c initial t ∧ HasAcc c (update M c d initial block acc) t := by
  dsimp only
  constructor
  · constructor <;> simp [bodyExecution,ctx.memory,ctx.bh,ctx.it,ctx.rows,ctx.cols,ctx.mask]
  · simp [HasAcc,bodyExecution,update,qValue,kValue,hValue,qOffset,kOffset,hOffset,key,row,column,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    constructor <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPValue
