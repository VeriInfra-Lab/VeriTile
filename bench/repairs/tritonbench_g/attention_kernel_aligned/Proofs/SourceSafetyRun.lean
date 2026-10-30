import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.SourceSafetySteps
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
open VeriTile Triton FP.Structural Safety Source Full FPValue
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
theorem prefix_run (A : Algebra α) (P : Primitives α) (p : Parameters) (s : State α) :
    run (backend A P) (queryPrefix p) s=some (prefixResult P p s) := by
  simp [queryPrefix,prologue,source,typedSurface,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,numeric,backend,prefixResult,
    initialAccumulator,qPointer,kPointer,vPointer,integralMod,natLt,Fin.isLt,
    TileShape.indexToList,scaleValue,row,headBase,kvBase,valueBase,outputBase,ofFloat,toFloat,
    integralFloorDiv,Region.cast,ComputeDType.eraseDType,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Option.bind]
  rfl

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.SourceSafety
