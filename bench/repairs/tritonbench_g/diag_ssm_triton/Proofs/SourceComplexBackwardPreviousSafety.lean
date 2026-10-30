import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardPrevious
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

private theorem previous_offset_safe (x batch dim : Nat) :
    ((x : Int) - (batch : Int) * (dim : Int)).toNat = x - batch * dim := by
  rw [← Int.natCast_mul]
  omega
private theorem sub_cast_safe (x y : Nat) : ((x : Int) - (y : Int)).toNat = x-y := by omega
private theorem positive_cast_safe (n : Nat) : ((n : Int) > 0) ↔ 0 < n := by omega

theorem previous_safe (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (h : Indexed p length t a b) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (((forwardLoop p length).drop 4).take 1) a ↔
    Stmt.TraceSafeList bounds (((historicalLoop p length).drop 4).take 1) b := by
  rcases h with ⟨h, hTa, hTb, ⟨offsets, hOa, hOb⟩, ⟨gy, hYa, hYb⟩, ⟨gyi, hZa, hZb⟩⟩
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨sr, hSa, hSb⟩, ⟨si, hIa, hIb⟩,
    ⟨gr, hGa, hGb⟩, ⟨gi, hJa, hJb⟩, ⟨lr, hLa, hLb⟩, ⟨li, hKa, hKb⟩⟩
  by_cases hpos : 0 < length - 1 - t
  all_goals simp only [forwardLoop, forwardBody, Kernels.complexBackwardTyped,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, List.map_cons, List.map_nil,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals simp only [List.getElem?_cons_zero, List.getElem?_cons_succ]
  all_goals simp only [historicalLoop, ComplexBackwardSource.body, Erasure.complexBackward,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeKernel.eraseDType, ComputeKernel.fromKernelBody, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil,
    Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
  all_goals simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType, List.drop, List.take,
    eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
  all_goals simp only [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MemAccess.SafeAt, MaskOpt.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub, ComparableDType.gt, Tile.cop,
    bind, Option.bind, Option.map_some, FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.map,
    BlockState.writeMemTyped_real, BlockState.foldl_writeMem_masked_regs, BlockState.foldl_writeMem_prop_masked_regs, true_and, and_true, hTa, hTb, hOa, hOb, String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hIa, hIb, hGa, hGb, hJa, hJb, hLa, hLb, hKa, hKb, Tile.scalar, positive_cast_safe, hpos, decide_true, decide_false, Bool.false_eq_true]
  all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
    hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hLa, hLb,
    BlockState.readMemValue, Region.cast_cast, true_and, and_true]
  all_goals try simp only [Tile.scalar, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
    Nat.mul_assoc, sub_cast_safe, previous_offset_safe, BlockState.readMemAs, memory]
  all_goals try simp (disch := positivity) only [Int.toNat_add, Int.toNat_mul, Int.toNat_natCast, Int.ofNat_eq_natCast, Int.reduceToNat]
  all_goals rfl

#axiomsClean previous_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
