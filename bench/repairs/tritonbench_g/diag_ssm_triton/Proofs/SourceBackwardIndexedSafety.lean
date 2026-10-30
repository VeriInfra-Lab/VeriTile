import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardPrevious
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem indexed_safe (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (h : BackwardRelated p length a b) (ht : t < length) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds ((forwardLoop p length).take 3)
      (a.setReg "i" .int [] (Tile.scalar (t : Int))) ↔
    Stmt.TraceSafeList bounds ((historicalLoop p length).take 3)
      (b.setReg "i" .nat [] (Tile.scalar t)) := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨state, hSa, hSb⟩, ⟨gradient, hGa, hGb⟩, ⟨transition, hLa, hLb⟩⟩
  have time_cast : (length : Int) - 1 - (t : Int) = (length - 1 - t : Nat) := by omega
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
  simp only [historicalLoop, RealReference.diagSsmBackwardLoopBody, List.take, List.drop,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MemAccess.SafeAt, MaskOpt.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.gt, Tile.cop,
    bind, Option.bind, true_and, and_true, String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hGa, hGb, hLa, hLb, Tile.scalar, time_cast]
  simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
    hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hLa, hLb,
    BlockState.readMemValue, Region.cast_cast, true_and, and_true]
  simp only [Tile.scalar, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
    Nat.mul_assoc, BlockState.readMemAs, memory]
  all_goals rfl

#axiomsClean indexed_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
