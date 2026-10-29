import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Model
namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def preludeExecution (P : Source.Arithmetic α α) (p : Parameters) (s : State α) :
    {t : State α // run (backend P) (prelude p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  rw [prelude_raw]
  simp [rawPrelude,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,backend,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
