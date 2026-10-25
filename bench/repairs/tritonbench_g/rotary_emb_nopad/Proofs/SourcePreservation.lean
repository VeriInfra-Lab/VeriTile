import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.LegacyPastedSurface
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.TypedDefinitions
namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Kernels
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
open VeriTile.Bench.Repairs.RotaryEmbNopad.Typed (int_half dimension_projection)
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-! These entire-program observations preserve success/failure and every memory
cell for arbitrary floating primitives and initial states. The old source used
an Int register where the current frontend uses its exact nonnegative Nat
projection; the transient register files are therefore not asserted equal.
The actual source safety and public Real/FP/IO/frame results are replayed directly. -/

/-- Complete observable execution, including every memory cell and failure. -/
theorem uncached_memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : UncachedLayout) (s : State A) :
    (FP.Structural.exec M (uncached p) s).map (·.mem) = (FP.Structural.exec M (LegacyPastedSurface.uncached p) s).map (·.mem) := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  have signedLeader :
      ((s.pids 0 : Int)-(p.KV_GROUP_NUM : Int)*((s.pids 0 : Int)/(p.KV_GROUP_NUM : Int))=0) ↔
        s.pids 0%p.KV_GROUP_NUM=0 := by
    rw [←Int.emod_def,←Int.natCast_emod,Int.natCast_eq_zero]
  by_cases leader : s.pids 0 % p.KV_GROUP_NUM = 0
  all_goals simp [leader,signedLeader,FP.Structural.exec,uncached,LegacyPastedSurface.uncached,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericLe,natLe,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,Int.natCast_ediv,headNonneg,State.setReg,masked_scatter_memory]
  all_goals try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
  all_goals try simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,Int.toNat_natCast,Int.ofNat_lt]

#axiomsClean uncached_memory_preserved

/-- Complete observable execution, including every memory cell and failure. -/
theorem uncached_shared_memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : UncachedLayout) (s : State A) :
    (FP.Structural.exec M (uncachedShared p) s).map (·.mem) = (FP.Structural.exec M (LegacyPastedSurface.uncachedShared p) s).map (·.mem) := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  have signedLeader :
      ((s.pids 0 : Int)-(p.KV_GROUP_NUM : Int)*((s.pids 0 : Int)/(p.KV_GROUP_NUM : Int))=0) ↔
        s.pids 0%p.KV_GROUP_NUM=0 := by
    rw [←Int.emod_def,←Int.natCast_emod,Int.natCast_eq_zero]
  by_cases leader : s.pids 0 % p.KV_GROUP_NUM = 0
  all_goals simp [leader,signedLeader,FP.Structural.exec,uncachedShared,LegacyPastedSurface.uncachedShared,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericLe,natLe,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,Int.natCast_ediv,headNonneg,State.setReg,masked_scatter_memory]
  all_goals try dsimp only [TileShape.insertAxis,TileShape.dropInsertedIndex]
  all_goals try simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,Int.toNat_natCast,Int.ofNat_lt]

#axiomsClean uncached_shared_memory_preserved

/-- Complete observable execution, including every memory cell and failure. -/
theorem cached_memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : CachedLayout) (s : State A) :
    (FP.Structural.exec M (cached p) s).map (·.mem) = (FP.Structural.exec M (LegacyPastedSurface.cached p) s).map (·.mem) := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  by_cases head : s.pids 0 < p.Q_HEAD_NUM
  all_goals simp [head,FP.Structural.exec,cached,LegacyPastedSurface.cached,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericLe,natLe,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,Int.natCast_ediv,headNonneg,State.setReg,masked_scatter_memory]
  all_goals try simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,Int.toNat_natCast,Int.ofNat_lt]

#axiomsClean cached_memory_preserved

/-- Complete observable execution, including every memory cell and failure. -/
theorem cached_shared_memory_preserved {A : Type} [Inhabited A] (M : Algebra A) (p : CachedLayout) (s : State A) :
    (FP.Structural.exec M (cachedShared p) s).map (·.mem) = (FP.Structural.exec M (LegacyPastedSurface.cachedShared p) s).map (·.mem) := by
  have headNonneg : ¬(s.pids 0 : Int)<0 := by omega
  by_cases head : s.pids 0 < p.Q_HEAD_NUM
  all_goals simp [head,FP.Structural.exec,cachedShared,LegacyPastedSurface.cachedShared,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    numericLt,natLt,numericLe,natLe,numericEq,bop,numeric,store,Region.cast,ComputeDType.eraseDType,
    ofFloat,toFloat,integralFloorDiv,integralMod,Int.natCast_ediv,headNonneg,State.setReg,masked_scatter_memory]
  all_goals try simp only [int_half,←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv,Int.toNat_natCast,Int.ofNat_lt]

#axiomsClean cached_shared_memory_preserved

end VeriTile.Bench.Repairs.RotaryEmbNopad.Kernels
