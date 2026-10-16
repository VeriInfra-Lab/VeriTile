import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealAccess
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealLoopProgram
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealSchedule

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealPureSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

theorem setup_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealGroup.setup p) s := by
  apply TraceProofs.all_safe
  simp only [RealGroup.setup,RealProgram.body,RealSource.kernel,Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType,
    List.getElem!_cons_zero,List.getElem!_cons_succ,RealProgram.loopBody,RealProgram.thenBody,
    List.take,List.drop]
  simp [Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem prefix_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealPrefix.code p) s := by
  apply TraceProofs.all_safe
  simp only [RealPrefix.code,RealSource.kernel,Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType,
    List.getElem!_cons_zero,List.getElem!_cons_succ,RealProgram.loopBody,RealProgram.thenBody,
    List.take,List.drop]
  simp [Stmt.TraceSafe,Op.SafeAt.eq_def]
  repeat' first
    | intro _
    | constructor
    | (apply TraceProofs.all_safe; simp [Stmt.TraceSafe,Op.SafeAt.eq_def])
    | split

theorem schedule_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealProgram.schedule p) s := by
  rw [RealSchedule.matches_source]
  apply TraceProofs.all_safe
  simp [RealControl.update,RealSchedule.condition,Stmt.TraceSafe,Op.SafeAt.eq_def,setup_safe]
  intro t
  split <;> simp [setup_safe]

theorem store_prepare_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds ((RealStore.code p).take 5) s := by
  apply TraceProofs.all_safe
  simp only [RealStore.code,RealProgram.body,RealSource.kernel,Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType,
    List.getElem!_cons_zero,List.getElem!_cons_succ,RealProgram.loopBody,RealProgram.thenBody,
    List.take,List.drop]
  simp [Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem reset_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds ((RealStore.code p).drop 6) s := by
  apply TraceProofs.all_safe
  simp only [RealStore.code,RealProgram.body,RealSource.kernel,Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType,
    List.getElem!_cons_zero,List.getElem!_cons_succ,RealProgram.loopBody,RealProgram.thenBody,
    List.take,List.drop]
  simp [Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem dot_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealProgram.dot p) s := by
  rw [RealDot.matches_source]
  apply TraceProofs.all_safe
  simp [RealDot.update,Stmt.TraceSafe,Op.SafeAt.eq_def]

theorem flatten_ok (p : Parameters) (v : Bool) :
    (RealSource.kernel p v).FlattenOk := by
  cases v <;>
    simp [RealSource.kernel,Parameters.source,TypedSource.kernel,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeKernel.fromKernelBody,ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType,Except.bind,bind,
    List.map_cons,List.map_nil,FloatDType.toTileDType,
    List.getElem!_cons_zero,List.getElem!_cons_succ,RealProgram.loopBody,RealProgram.thenBody,
    List.take,List.drop,Kernel.FlattenOk,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealPureSafety
