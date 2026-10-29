import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealSource

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealSourceSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FastRmsLayernorm
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem forward_trace (h : Call) (bounds : RegionBounds) (u : BlockState) :
    Kernel.TraceSafe bounds (Kernels.forward h).eraseDType.toAlgKernel u ↔
      Kernel.TraceSafe bounds (ForwardRun.original h.gemma "y" "x" "w" "r" h.cols h.cols 1 1 h.cols h.block h.eps) u := by
  cases mode : h.gemma
  all_goals
    simp [Kernel.TraceSafe,Kernels.forward,Kernels.regularSource,Kernels.gemmaSource,TypedSource.forward,TypedSource.gemma,ForwardRun.original,mode,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,
    exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,BlockState.readMem]
    <;> simp [-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add]

theorem forward_flat (h : Call) : (Kernels.forward h).eraseDType.toAlgKernel.FlattenOk := by
  cases mode : h.gemma
  all_goals
    simp [Kernel.FlattenOk,Kernels.forward,Kernels.regularSource,Kernels.gemmaSource,TypedSource.forward,TypedSource.gemma,mode,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

theorem backward_trace (h : Call) (bounds : RegionBounds) (u : BlockState) :
    Kernel.TraceSafe bounds (Kernels.backward h).eraseDType.toAlgKernel u ↔
      Kernel.TraceSafe bounds (BackwardRun.original h.gemma "dx" "x" "w" "r" "dw" h.cols h.cols 1 1 1 h.cols h.block h.eps) u := by
  cases mode : h.gemma
  all_goals
    simp [Kernel.TraceSafe,Kernels.backward,Kernels.inputGradientSource,Kernels.inputGradientBody,TypedSource.backward,BackwardRun.original,mode,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,
    exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,BlockState.setReg,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.ptrSub,Tile.uop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,BlockState.readMem]
    <;> simp [-Nat.cast_mul,-Nat.cast_add,← Int.natCast_mul,← Int.natCast_add]

theorem backward_flat (h : Call) : (Kernels.backward h).eraseDType.toAlgKernel.FlattenOk := by
  cases mode : h.gemma
  all_goals
    simp [Kernel.FlattenOk,Kernels.backward,Kernels.inputGradientSource,Kernels.inputGradientBody,TypedSource.backward,mode,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Except.bind,bind,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#axiomsClean forward_trace
#axiomsClean backward_trace
#axiomsClean forward_flat
#axiomsClean backward_flat
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealSourceSafety
