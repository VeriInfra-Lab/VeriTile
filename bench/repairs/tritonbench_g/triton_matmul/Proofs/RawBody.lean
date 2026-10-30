import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawLoopModel

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    {t : State α // run (backend M D) (body p st)
      (current.setReg "k" .int [] (fun _ => (k : Int))) = some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [body,Source.typed,Source.halfSource,Source.byteSource,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
       numeric,bop,natLt,numericEq,numericLt,ComputeDType.eraseDType,ofFloat,toFloat,
       FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
       integralFloorDiv,integralMod,ctx.memory,ctx.dimK,ctx.strideAM,ctx.strideAK,
       ctx.strideBK,ctx.strideBN,ctx.offk,ctx.am,ctx.bn,ha,backend]
     rfl)

#axiomsClean bodyExecution
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
