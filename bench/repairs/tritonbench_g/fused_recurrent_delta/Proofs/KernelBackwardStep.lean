import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackwardRegisters
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardStep
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def reverse (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : KernelBackwardRegisters.Reverse s f bh iv t h u) :
    {v : State α // run (engine M) (first .backward s f scale).body u=some v} := by
  obtain ⟨hq,hk,hv,hdo,hdk,hdv,hb,hdb,hmk,hmv,hh⟩ := regs
  cases hbv : f.vectorBeta
  all_goals
    simp only [hbv,Bool.false_eq_true,ite_false,ite_true] at hb hdb
    refine ⟨?_,?_⟩
    swap
    simp only [first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hbv,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=350000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod,hq,hk,hv,hdo,hdk,hdv,hb,hdb,hmk,hmv,hh,
      keyPtr,valPtr,betaPtr,dbetaPtr]
    try rfl

def recompute (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : Tile s α) (u : State α) (regs : KernelBackwardRegisters.Recompute s f bh iv t h u) :
    {v : State α // run (engine M) (second s f scale).body u=some v} := by
  obtain ⟨hk,hv,hdo,hdq,hdk,hdv,hb,hmk,hmv,hh⟩ := regs
  cases hbv : f.vectorBeta
  all_goals
    simp only [hbv,Bool.false_eq_true,ite_false,ite_true] at hb
    refine ⟨?_,?_⟩
    swap
    simp only [second,first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hbv,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=350000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod,hk,hv,hdo,hdq,hdk,hdv,hb,hmk,hmv,hh,
      keyPtr,valPtr,betaPtr]
    try rfl

#axiomsClean reverse
#axiomsClean recompute
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelBackwardStep
