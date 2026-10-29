import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardStages
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

structure Previous (p : BackwardLayout) (length iteration : Nat) (a b : BlockState) : Prop
    extends Indexed p length iteration a b where
  previousState : ∃ v : Tile .real [p.B], a.regs .real [p.B] "s_real" = some v ∧
    b.regs .real [p.B] "s_real" = some v

  previousStateImag : ∃ v : Tile .real [p.B], a.regs .real [p.B] "s_imag" = some v ∧
    b.regs .real [p.B] "s_imag" = some v

private theorem previous_offset (x batch dim : Nat) :
    ((x : Int) - (batch : Int) * (dim : Int)).toNat = x - batch * dim := by
  rw [← Int.natCast_mul]
  omega

private theorem positive_cast (n : Nat) : ((n : Int) > 0) ↔ 0 < n := by omega

theorem previous_steps (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (h : Indexed p length t a b) :
    ∃ aa bb, stepStmts (((forwardLoop p length).drop 4).take 1) a = some aa ∧
      stepStmts (((historicalLoop p length).drop 4).take 1) b = some bb ∧ Previous p length t aa bb := by
  rcases h with ⟨h, hTa, hTb, ⟨offsets, hOa, hOb⟩, ⟨gy, hYa, hYb⟩, ⟨gyi, hZa, hZb⟩⟩
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨sr, hSa, hSb⟩, ⟨si, hIa, hIb⟩,
    ⟨gr, hGa, hGb⟩, ⟨gi, hJa, hJb⟩, ⟨lr, hLa, hLb⟩, ⟨li, hKa, hKb⟩⟩
  by_cases hpos : 0 < length - 1 - t
  ·
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    rotate_left 2
    ·
      simp only [forwardLoop, forwardBody, Kernels.complexBackwardTyped,
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
      simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, List.drop, List.take, eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
      simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTa, hOa, hMa, hCa, hBatch, hDim,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, Int.ofNat_lt, positive_cast,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs, memory, Region.cast_cast,
        Tile.scalar, previous_offset, Int.ofNat_eq_natCast, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
      rfl
    · simp only [historicalLoop, ComplexBackwardSource.body, Erasure.complexBackward,
        ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeKernel.eraseDType, ComputeKernel.fromKernelBody, ComputeDType.eraseDType,
        Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
        Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
        ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil,
        Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
      simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
        NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
        VeriTile.Triton.eraseDType, FloatDType.toTileDType, List.drop, List.take,
        eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
      simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTb, hOb, hMb, hCb,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs]
      rfl
    · refine ⟨⟨⟨rfl, pids, ?_, ?_, ?_, ⟨columns, ?_, ?_⟩, ⟨mask, ?_, ?_⟩,
        ⟨sr, ?_, ?_⟩, ⟨si, ?_, ?_⟩, ⟨gr, ?_, ?_⟩, ⟨gi, ?_, ?_⟩,
        ⟨lr, ?_, ?_⟩, ⟨li, ?_, ?_⟩⟩,
        ?_, ?_, ⟨offsets, ?_, ?_⟩, ⟨gy, ?_, ?_⟩, ⟨gyi, ?_, ?_⟩⟩, ?_, ?_⟩
      all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hIa, hIb, hGa, hGb, hJa, hJb, hLa, hLb, hKa, hKb,
        hTa, hTb, hOa, hOb, hYa, hYb, hZa, hZb]
      all_goals exact ⟨_, rfl, rfl⟩
  ·
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    rotate_left 2
    ·
      simp only [forwardLoop, forwardBody, Kernels.complexBackwardTyped,
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
      simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, List.drop, List.take, eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
      simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTa, hOa, hMa, hCa, hBatch, hDim,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, Int.ofNat_lt, positive_cast,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs, memory, Region.cast_cast,
        Tile.scalar, previous_offset, Int.ofNat_eq_natCast, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
      rfl
    · simp only [historicalLoop, ComplexBackwardSource.body, Erasure.complexBackward,
        ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeKernel.eraseDType, ComputeKernel.fromKernelBody, ComputeDType.eraseDType,
        Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
        Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
        ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil,
        Except.bind, bind, Bool.false_eq_true, ↓reduceIte]
      simp only [Op.eraseDType.eq_def, MemAccess.eraseDType, MaskOpt.eraseDType,
        NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
        VeriTile.Triton.eraseDType, FloatDType.toTileDType, List.drop, List.take,
        eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
      simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
        Tile.bop, Tile.cop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.gt, bind, Option.bind, hTb, hOb, hMb, hCb,
        String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hpos, decide_true, decide_false, Bool.false_eq_true]
      simp only [BlockState.readMemValue, BlockState.readMemAs]
      rfl
    · refine ⟨⟨⟨rfl, pids, ?_, ?_, ?_, ⟨columns, ?_, ?_⟩, ⟨mask, ?_, ?_⟩,
        ⟨sr, ?_, ?_⟩, ⟨si, ?_, ?_⟩, ⟨gr, ?_, ?_⟩, ⟨gi, ?_, ?_⟩,
        ⟨lr, ?_, ?_⟩, ⟨li, ?_, ?_⟩⟩,
        ?_, ?_, ⟨offsets, ?_, ?_⟩, ⟨gy, ?_, ?_⟩, ⟨gyi, ?_, ?_⟩⟩, ?_, ?_⟩
      all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
        hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hIa, hIb, hGa, hGb, hJa, hJb, hLa, hLb, hKa, hKb,
        hTa, hTb, hOa, hOb, hYa, hYb, hZa, hZb]
      all_goals exact ⟨_, rfl, rfl⟩

#axiomsClean previous_steps
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
