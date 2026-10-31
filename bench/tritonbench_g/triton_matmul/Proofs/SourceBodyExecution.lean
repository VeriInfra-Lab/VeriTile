import bench.tritonbench_g.triton_matmul.Proofs.SourceCoordinates
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p→α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    {t : State α // run (backend M D) (iteration p st)
      (current.setReg "k" .int [] (fun _=>(k : Int)))=some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_,?_⟩
     swap
     simp [iteration,typed,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.call,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.rawBody,dtype,
       ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
       ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
       evalOffsets,store,numeric,bop,natLt,numericLt,numericLe,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.metadata.1,ctx.metadata.2.1,ctx.metadata.2.2.1,ctx.metadata.2.2.2.1,
       ctx.metadata.2.2.2.2.1,ctx.metadata.2.2.2.2.2.1,ctx.metadata.2.2.2.2.2.2.1,
       ctx.metadata.2.2.2.2.2.2.2.1,ctx.metadata.2.2.2.2.2.2.2.2,ctx.memory,ctx.offk,ctx.am,ctx.bn,
       ctx.mm,ctx.nn,ha,backend]
     rfl)

#axiomsClean bodyExecution
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
