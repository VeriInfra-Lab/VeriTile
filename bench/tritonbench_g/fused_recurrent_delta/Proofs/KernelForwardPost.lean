import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelStateMemory
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardPost
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 14000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def execute (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : ForwardTile s α) (u : State α) (carry : u.regs .real [s.FBV,s.BK] "h"=some h)
    (static : KernelStatic.Forward s bh iv u) :
    {v : State α // run (engine M) (first .forward s f scale).after u=some v} := by
  obtain ⟨head,key,value,mask⟩ := static
  cases hf : f.final <;> cases hb : f.vectorBeta
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [first,source,Source.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hf,hb,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=250000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod,head,key,value,mask,carry]
    try rfl

theorem memory (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : ForwardTile s α) (u : State α) (carry : u.regs .real [s.FBV,s.BK] "h"=some h)
    (static : KernelStatic.Forward s bh iv u) :
    (execute M s f scale bh iv h u carry static).val.mem=
      (if f.final then KernelStateMemory.store M s "ht" bh iv h u else u).mem := by
  rcases f with ⟨vector,initial,final,ig,tg,default⟩
  cases vector <;> cases final <;> cases static
  all_goals simp only [execute,KernelStateMemory.store,KernelReference.stored,KernelReference.cast,
    KernelStatic.mask,stateAddress,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    TileShape.dropInsertedIndex,Nat.zero_mul,Nat.zero_add,Nat.add_assoc,
    Bool.false_eq_true,ite_false,ite_true,State.setReg_mem]
  all_goals try rfl
  all_goals apply KernelStore.masked_mem
  all_goals rfl

#axiomsClean execute
#axiomsClean memory
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardPost
