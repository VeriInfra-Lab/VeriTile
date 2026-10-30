import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPProgram
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPValues

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPSetup
open VeriTile.Triton FP.Structural FPProgram VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Constructive execution of all source metadata-prefix statements. -/
def preguard (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α) : {s : State α //
    run (machine M P) (prelude h) (entry h batch split data)=some s} := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [prelude,code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,
      ComputeKernel.surfaceBody,List.drop,run,step,FP.Structural.evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,entry,Data.cells,State.setReg,Cell.read,Region.cast]
    rfl

/-- Constructive execution of the complete source's offset/pointer/accumulator
setup for both admitted input precisions. -/
def preloop (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split slot : Nat) (data : Data α) (metadata : data.indices batch=(slot:Int)) :
    {s : State α // run (machine M P) (setup h)
      (preguard M P h batch split data).val=some s} := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [setup,active,code,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,Kernels.original_fp16_i64,Kernels.original_bf16_i64,
      ComputeKernel.surfaceBody,List.drop,Source.conditional?,Source.intRange?,
      Option.map_some,Option.getD_some,beq_self_eq_true]
    dsimp [preguard]
    simp (config := {maxSteps := 200000}) [Source.intRange?,Option.map,Option.getD,
      run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      machine,Algebra.withDefaultPrecision,resolvePrecision,numeric,
      integralFloorDiv,integralMod,numericLt,numericLe,numericEq,natLt,bop,
      State.setReg,Cell.read,ofFloat,toFloat,Region.cast,entry,Data.cells,metadata]
    try rfl

#axiomsClean preguard
#axiomsClean preloop
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPSetup
