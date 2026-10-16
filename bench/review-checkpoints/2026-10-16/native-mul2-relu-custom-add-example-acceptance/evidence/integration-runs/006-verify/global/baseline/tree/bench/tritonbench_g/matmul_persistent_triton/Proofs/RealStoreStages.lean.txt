import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealAccess

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStoreStages
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem partition (p : Parameters) : RealStore.code p =
    (RealStore.code p).take 5 ++ [RealAccess.output p] ++ (RealStore.code p).drop 6 := by
  simp only [RealStore.code,RealProgram.body,RealSource.kernel,Parameters.source,
    TypedSource.kernel,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,FloatDType.toTileDType]
  rfl

theorem prepares (p : Parameters) (s : BlockState) (r c : Nat)
    (values : Fin p.BM → Fin p.BN → ℝ)
    (hr : s.regs .int [] "pid_m" = some (Tile.scalar (r : Int)))
    (hc : s.regs .int [] "pid_n" = some (Tile.scalar (c : Int)))
    (hv : s.regs .real [p.BM,p.BN] "accumulator" = some (Tile.mat fun i j => some (values i j))) :
    stepStmts ((RealStore.code p).take 5) s = some (RealStore.prepared p s r c values) := by
  simp only [RealStore.code,RealProgram.body,RealSource.kernel,Parameters.source,
    TypedSource.kernel,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,List.map_cons,List.map_nil,FloatDType.toTileDType,
    List.getElem!_cons_zero,List.getElem!_cons_succ,RealProgram.loopBody,RealProgram.thenBody,List.take]
  simp [RealStore.prepared,RealStore.rows,RealStore.columns,RealStore.offset,RealStore.active,
    stepStmts,stepStmt,evalOp.eq_def,hr,hc,hv,bind,Option.bind,
    Tile.scalar,Tile.mat,Tile.vec,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.expandDim,
    TileShape.dropInsertedIndex,NumericDType.add,NumericDType.mul,ComparableDType.lt,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,-Tile.scalar_eta]
  simp only [← Int.natCast_mul,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
  try rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealStoreStages
