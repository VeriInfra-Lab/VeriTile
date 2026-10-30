import bench.repairs.tritonbench_g.index_select_cat.Proofs.LegacyPastedSurface
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.IndexSelectCat
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-! All current source casts remain in the raw quotation. The complete
old/new execution preserves success/failure and every memory cell, permitting
an exact natural/signed representation change of temporary indices. -/

/-- Full observable execution for arbitrary payload primitives and initial state. -/
theorem memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : Layout) (s : State A) :
    (FP.Structural.exec M (kernel p) s).map (·.mem) =
      (FP.Structural.exec M (LegacyPastedSurface.kernel p) s).map (·.mem) := by
  simp (config := { maxSteps := 3000000 }) [FP.Structural.exec,kernel,LegacyPastedSurface.kernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,State.setReg,masked_scatter_memory]
  all_goals try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
  all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
#axiomsClean memory_preserved

/-- Full observable execution for arbitrary payload primitives and initial state. -/
theorem candidate_memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : Layout) (s : State A) :
    (FP.Structural.exec M (indexCommutedKernel p) s).map (·.mem) =
      (FP.Structural.exec M (LegacyPastedSurface.indexCommutedKernel p) s).map (·.mem) := by
  simp (config := { maxSteps := 3000000 }) [FP.Structural.exec,indexCommutedKernel,LegacyPastedSurface.indexCommutedKernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,State.setReg,masked_scatter_memory]
  all_goals try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
  all_goals try simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt]
#axiomsClean candidate_memory_preserved

end VeriTile.Bench.Repairs.IndexSelectCat
