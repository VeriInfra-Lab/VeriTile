import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawLoop

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (RawSource.accumulation M D p st initial) current) :
    {t : State α // run (backend M D) (epilogue p st) current = some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [epilogue,Source.typed,Source.halfSource,Source.byteSource,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
       numeric,bop,natLt,numericEq,numericLt,numericLe,ComputeDType.eraseDType,ofFloat,toFloat,
       FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
       integralFloorDiv,integralMod,ctx.memory,ctx.strideCM,ctx.strideCN,
       ctx.am,ctx.bn,ctx.mm,ctx.nn,ha,backend]
     rfl)

#axiomsClean epilogueExecution
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
