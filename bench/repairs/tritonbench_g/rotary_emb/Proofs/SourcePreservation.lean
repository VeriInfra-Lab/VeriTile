import bench.repairs.tritonbench_g.rotary_emb.Proofs.LegacyPastedSurface
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
namespace VeriTile.Bench.Repairs.RotaryEmb.Kernels
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-! Full old/new observable execution under arbitrary floating primitives.
Only internal natural/signed index carriers differ; all memory and failure
observations agree. Current full source safety and IO scopes replay directly. -/

/-- Complete observable execution for all source parameters and states. -/
theorem memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : Layout) (s : State A) :
    (FP.Structural.exec M (kernel p) s).map (·.mem) = (FP.Structural.exec M (LegacyPastedSurface.kernel p) s).map (·.mem) := by
  simp (config := { maxSteps := 3000000 }) [FP.Structural.exec,kernel,LegacyPastedSurface.kernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,State.setReg,masked_scatter_memory]

#axiomsClean memory_preserved

/-- Complete observable execution for all source parameters and states. -/
theorem candidate_memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : Layout) (s : State A) :
    (FP.Structural.exec M (indexCommuted p) s).map (·.mem) = (FP.Structural.exec M (LegacyPastedSurface.indexCommuted p) s).map (·.mem) := by
  simp (config := { maxSteps := 3000000 }) [FP.Structural.exec,indexCommuted,LegacyPastedSurface.indexCommuted,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,State.setReg,masked_scatter_memory]

#axiomsClean candidate_memory_preserved

end VeriTile.Bench.Repairs.RotaryEmb.Kernels
