import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelProgram
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelPrefix
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def forward (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (u : State α) :
    {v : State α // run (engine M) (first .forward s f scale).before u=some v} := by
  cases hi : f.initial <;> cases hb : f.vectorBeta
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [first,source,Source.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hi,hb,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=250000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod]
    try rfl

def backward (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (u : State α) :
    {v : State α // run (engine M) (first .backward s f scale).before u=some v} := by
  cases ht : f.terminalGrad <;> cases hb : f.vectorBeta
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,ht,hb,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=250000}) [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
      FP.Structural.store,integralFloorDiv,integralMod]
    try rfl

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.KernelPrefix
