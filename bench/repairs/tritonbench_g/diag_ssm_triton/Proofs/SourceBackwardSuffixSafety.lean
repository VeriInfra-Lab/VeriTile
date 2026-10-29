import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardPrevious
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem suffix_safe (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (h : Previous p length t a b) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds ((forwardLoop p length).drop 4) a ↔
    Stmt.TraceSafeList bounds ((historicalLoop p length).drop 4) b := by
  rcases h with ⟨⟨h, hTa, hTb, ⟨offsets, hOa, hOb⟩, ⟨gy, hYa, hYb⟩⟩, ⟨prev, hPa, hPb⟩⟩
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨state, hSa, hSb⟩, ⟨gradient, hGa, hGb⟩, ⟨transition, hLa, hLb⟩⟩
  simp only [forwardLoop, forwardBody, Kernels.backwardTyped,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, List.map_cons, List.map_nil,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [List.getElem?_cons_zero, List.getElem?_cons_succ]
  simp only [historicalLoop, RealReference.diagSsmBackwardLoopBody, List.take, eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq, List.drop, eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MemAccess.SafeAt, MaskOpt.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.gt, Tile.cop,
    bind, Option.bind, true_and, and_true, String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hGa, hGb, hLa, hLb, hOa, hOb, hYa, hYb, hPa, hPb]
  simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
    hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hLa, hLb,
    BlockState.readMemValue, Region.cast_cast, true_and, and_true]
  simp only [Tile.scalar, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
    Nat.mul_assoc, BlockState.readMemAs, memory]
  simp only [BlockState.writeMemTyped_real, BlockState.foldl_writeMem_masked_regs,
    bind, Option.bind, Option.map_some, true_and, and_true]
  all_goals rfl

#axiomsClean suffix_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
