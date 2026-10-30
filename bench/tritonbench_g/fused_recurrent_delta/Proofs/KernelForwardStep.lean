import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelRegisters
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardStep
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 20000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def execute (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv t : Nat)
    (h : ForwardTile s α) (u : State α) (regs : KernelRegisters.Forward s f bh iv t h u) :
    {v : State α // run (engine M) (first .forward s f scale).body u=some v} := by
  obtain ⟨hq,hk,hv,ho,hb,hmk,hmv,hh⟩ := regs
  cases hbv : f.vectorBeta
  all_goals
    simp only [hbv,Bool.false_eq_true,ite_false,ite_true] at hb
    refine ⟨?_,?_⟩
    swap
    simp only [first,source,Source.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hbv,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=300000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod,hq,hk,hv,ho,hb,hmk,hmv,hh,
      keyPtr,valPtr,betaPtr]
    try rfl

#axiomsClean execute
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelForwardStep
