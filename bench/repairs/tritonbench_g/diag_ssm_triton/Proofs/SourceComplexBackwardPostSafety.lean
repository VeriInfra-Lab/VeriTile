import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardLoop
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem post_safe (p : BackwardLayout) (length : Nat) (a b : BlockState)
    (h : BackwardRelated p length a b) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (backwardPost p length) a ↔
    Stmt.TraceSafeList bounds (ComplexBackwardSource.postlude p.gs p.gl p.B) b := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨sr, hSa, hSb⟩, ⟨si, hIa, hIb⟩,
    ⟨gr, hGa, hGb⟩, ⟨gi, hJa, hJb⟩, ⟨lr, hLa, hLb⟩, ⟨li, hKa, hKb⟩⟩
  simp only [backwardPost, forwardBody, Kernels.complexBackwardTyped,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, List.map_cons, List.map_nil,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [ComplexBackwardSource.postlude, Erasure.complexBackward,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeKernel.eraseDType, ComputeKernel.fromKernelBody, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil,
    Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
  simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, List.drop,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  simp only [
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MemAccess.SafeAt, MaskOpt.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.gt, Tile.cop,
    bind, Option.bind, Option.map_some, FloatDType.toTileDType, FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.map, true_and, and_true, String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hIa, hIb, hGa, hGb, hJa, hJb, hLa, hLb, hKa, hKb, BlockState.writeMemTyped_real, BlockState.foldl_writeMem_masked_regs, BlockState.foldl_writeMem_prop_masked_regs]
  try simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
    hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hLa, hLb,
    BlockState.readMemValue, Region.cast_cast, true_and, and_true]
  try simp only [Tile.scalar, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
    Nat.mul_assoc, BlockState.readMemAs, memory]
  all_goals try simp (config := {maxSteps := 2000000}) (disch := positivity) only [Int.toNat_add, Int.toNat_mul, Int.toNat_natCast, Int.ofNat_eq_natCast, Int.reduceToNat]
  all_goals rfl

#axiomsClean post_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
