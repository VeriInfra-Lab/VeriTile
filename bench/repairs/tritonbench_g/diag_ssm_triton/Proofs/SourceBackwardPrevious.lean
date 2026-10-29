import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardStages
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Previous (p : BackwardLayout) (length iteration : Nat) (a b : BlockState) : Prop
    extends Indexed p length iteration a b where
  previousState : ∃ v : Tile .real [p.B], a.regs .real [p.B] "s" = some v ∧
    b.regs .real [p.B] "s" = some v

private theorem previous_offset (x batch dim : Nat) :
    ((x : Int) - (batch : Int) * (dim : Int)).toNat = x - batch * dim := by
  rw [← Int.natCast_mul]
  omega

private theorem positive_cast (n : Nat) : ((n : Int) > 0) ↔ 0 < n := by omega

theorem previous_steps (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (h : Indexed p length t a b) :
    ∃ aa bb, stepStmts (((forwardLoop p length).drop 3).take 1) a = some aa ∧
      stepStmts (((historicalLoop p length).drop 3).take 1) b = some bb ∧ Previous p length t aa bb := by
  rcases h with ⟨h, hTa, hTb, ⟨offsets, hOa, hOb⟩, ⟨gy, hYa, hYb⟩⟩
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨state, hSa, hSb⟩, ⟨gradient, hGa, hGb⟩, ⟨transition, hLa, hLb⟩⟩
  by_cases hpos : 0 < length - 1 - t
  ·
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    rotate_left 2
    ·
      simp only [forwardLoop, forwardBody, Kernels.backwardTyped,
        ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
        ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeDType.eraseDType, List.map_cons, List.map_nil,
        Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
        Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
        ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
      simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
      simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, List.drop, List.take]
      simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTa, hOa, hMa, hCa, hBatch, hDim,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, Int.ofNat_lt, positive_cast,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs, memory, Region.cast_cast,
        Tile.scalar, previous_offset, Int.toNat_natCast]
      rfl
    · simp only [historicalLoop, RealReference.diagSsmBackwardLoopBody, List.drop, List.take,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTb, hOb, hMb, hCb,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs]
      rfl
    · refine ⟨⟨⟨rfl, pids, ?_, ?_, ?_, ⟨columns, ?_, ?_⟩, ⟨mask, ?_, ?_⟩,
        ⟨state, ?_, ?_⟩, ⟨gradient, ?_, ?_⟩, ⟨transition, ?_, ?_⟩⟩,
        ?_, ?_, ⟨offsets, ?_, ?_⟩, ⟨gy, ?_, ?_⟩⟩, ?_⟩
      all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hGa, hGb, hLa, hLb,
        hTa, hTb, hOa, hOb, hYa, hYb]
      exact ⟨_, rfl, rfl⟩
  ·
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    rotate_left 2
    ·
      simp only [forwardLoop, forwardBody, Kernels.backwardTyped,
        ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
        ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeDType.eraseDType, List.map_cons, List.map_nil,
        Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
        Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
        ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
      simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
      simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, List.drop, List.take]
      simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTa, hOa, hMa, hCa, hBatch, hDim,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, Int.ofNat_lt, positive_cast,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs, memory, Region.cast_cast,
        Tile.scalar, previous_offset, Int.toNat_natCast]
      rfl
    · simp only [historicalLoop, RealReference.diagSsmBackwardLoopBody, List.drop, List.take,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTb, hOb, hMb, hCb,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs]
      rfl
    · refine ⟨⟨⟨rfl, pids, ?_, ?_, ?_, ⟨columns, ?_, ?_⟩, ⟨mask, ?_, ?_⟩,
        ⟨state, ?_, ?_⟩, ⟨gradient, ?_, ?_⟩, ⟨transition, ?_, ?_⟩⟩,
        ?_, ?_, ⟨offsets, ?_, ?_⟩, ⟨gy, ?_, ?_⟩⟩, ?_⟩
      all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hGa, hGb, hLa, hLb,
        hTa, hTb, hOa, hOb, hYa, hYb]
      exact ⟨_, rfl, rfl⟩

#axiomsClean previous_steps
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
