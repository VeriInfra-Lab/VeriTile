import bench.tritonbench_g.triton_matmul.Proofs.SourceDecomposition
import bench.tritonbench_g.triton_matmul.Sources.Numerics
import VeriTile.Triton.Float.Integer
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def setup (p : Parameters) (st : Storage) := (typed p st).surfaceBody.take 22

def prefixExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    {t : State α // run (backend M D) (setup p st) s=some t} := by
  cases st <;>
    (refine ⟨?_,?_⟩
     swap
     simp [setup,typed,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.call,VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source.rawBody,dtype,
       ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
       ComputeKernel.surfaceBody,run,step,stepAlg,evalExpr,evalComputeOp,evalOp_unfold,
       evalOffsets,store,numeric,bop,natLt,numericLt,numericLe,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]
     rfl)

#axiomsClean prefixExecution
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
