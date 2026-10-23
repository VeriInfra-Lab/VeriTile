import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Registers
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Coordinates

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def tilePreludeExecution (P : Source.Arithmetic α α) (p : Parameters)
    (pid ordinal : Nat) (s : State α) (h : Registers p pid s) :
    {t : State α // run (backend P) (tilePrelude p)
      (s.setReg "tile_ordinal" .nat [] (fun _ => ordinal))=some t} := by
  refine ⟨?_,?_⟩
  swap
  rw [tilePrelude_raw]
  simp [rawTilePrelude,TypedSource.kernel,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,backend,h.start,h.rows,h.columns,
    h.group,h.span,h.lanesM,h.lanesN,h.lanesK,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.Core
