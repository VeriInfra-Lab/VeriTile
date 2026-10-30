import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeValue

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

structure Context {α : Type} (c : Shape) (base : Nat) (initial current : State α) : Prop where
  key : ∀ off,current.mem "k" off=initial.mem "k" off
  value : ∀ off,current.mem "v" off=initial.mem "v" off
  gate : ∀ off,current.mem "g" off=initial.mem "g" off
  seed : ∀ off,current.mem "h0" off=initial.mem "h0" off
  bh : current.regs .nat [] "i_bh"=some (fun _=>head c base initial)
  ik : current.regs .nat [] "i_k"=some (fun _=>kblock c base initial)
  iv : current.regs .nat [] "i_v"=some (fun _=>vblock c base initial)

def HasAcc {α : Type} (c : Shape) (s : State α) (a : HIndex c→α) : Prop :=
  s.regs .real [c.BK,c.BV] "b_h"=some a

def entryExecution {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (base : Nat)
    (gateK hasSeed storeFinal : Bool) (s : State α) :
    {t : State α // run (engine M) (parts c base gateK hasSeed storeFinal).before s=some t} := by
  cases hasSeed
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (config := {maxSteps:=150000}) [parts,source,NativeSource.hOriginal,NativeSource.hSource,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg,integralFloorDiv,integralMod]
    rfl

theorem entryContext {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (base : Nat)
    (gateK hasSeed storeFinal : Bool) (s : State α) :
    Context c base s (entryExecution M c base gateK hasSeed storeFinal s).val ∧
    HasAcc c (entryExecution M c base gateK hasSeed storeFinal s).val (initial M c base s hasSeed) := by
  cases hasSeed
  all_goals
    constructor
    · constructor <;> simp [entryExecution,head,kblock,vblock,linear,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex] <;> rfl
    · simp [HasAcc,entryExecution,initial,zero,load,stateOffset,head,kblock,vblock,linear,
        FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
      rfl

#axiomsClean entryContext
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
