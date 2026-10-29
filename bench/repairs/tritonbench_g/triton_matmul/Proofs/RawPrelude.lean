import bench.repairs.tritonbench_g.triton_matmul.Proofs.RawControl
import bench.tritonbench_g.triton_matmul.Proofs.FPTypedSource

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    {t : State α // run (backend M D) (prelude p st) s = some t} := by
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [prelude,Source.typed,Source.halfSource,Source.byteSource,ComputeKernel.surfaceBody,
       run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericEq,numericLt,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]
     rfl)

#axiomsClean preludeExecution
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.RawSource
