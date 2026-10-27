import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardPrevious
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem write_related (p : BackwardLayout) (length : Nat) (a b : BlockState)
    (h : BackwardRelated p length a b) (r : RegionName) (offset : Nat) (v : ℝ) :
    BackwardRelated p length (a.writeMem r offset v) (b.writeMem r offset v) := by
  refine ⟨?_, h.pids, h.lengthReg, h.batchReg, h.dimReg, h.columns, h.mask, h.carryReal, h.carryImag, h.gradientReal, h.gradientImag, h.transitionReal, h.transitionImag⟩
  simp only [BlockState.writeMem, h.memory]

theorem store_related {ι : Type} (p : BackwardLayout) (length : Nat)
    (a b : BlockState) (h : BackwardRelated p length a b)
    (r : RegionName) (offset : ι → Nat) (v : ι → ℝ) (mask : ι → Bool) (lanes : List ι) :
    BackwardRelated p length
      (lanes.foldl (fun u i => if mask i then u.writeMem r (offset i) (v i) else u) a)
      (lanes.foldl (fun u i => if mask i then u.writeMem r (offset i) (v i) else u) b) := by
  induction lanes generalizing a b with
  | nil => exact h
  | cons i is ih =>
    simp only [List.foldl_cons]
    cases hm : mask i <;> simp only [hm, Bool.false_eq_true, ↓reduceIte]
    · exact ih a b h
    · exact ih _ _ (write_related p length a b h r (offset i) (v i))

theorem suffix_steps (p : BackwardLayout) (length t : Nat) (a b : BlockState)
    (h : Previous p length t a b) :
    ∃ aa bb, stepStmts ((forwardLoop p length).drop 5) a = some aa ∧
      stepStmts ((historicalLoop p length).drop 5) b = some bb ∧ BackwardRelated p length aa bb := by
  rcases h with ⟨⟨h, hTa, hTb, ⟨offsets, hOa, hOb⟩, ⟨gy, hYa, hYb⟩, ⟨gyi, hZa, hZb⟩⟩, ⟨prev, hPa, hPb⟩, ⟨previ, hQa, hQb⟩⟩
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨sr, hSa, hSb⟩, ⟨si, hIa, hIb⟩,
    ⟨gr, hGa, hGb⟩, ⟨gi, hJa, hJb⟩, ⟨lr, hLa, hLb⟩, ⟨li, hKa, hKb⟩⟩
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
    simp only [List.getElem?_cons_zero, List.getElem?_cons_succ, List.drop, eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
      Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
      FloatDType.toTileDType, FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.map,
      BlockState.writeMemTyped_real, BlockState.foldl_writeMem_masked_regs, BlockState.foldl_writeMem_prop_masked_regs, bind, Option.bind, Option.map_some,
      String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hOa, hOb, hMa, hMb, hSa, hSb, hIa, hIb, hGa, hGb, hJa, hJb, hLa, hLb, hKa, hKb, hYa, hYb, hZa, hZb, hPa, hPb, hQa, hQb]
    try simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hOa, hMa, hSa, hGa, hLa, hYa, hPa, BlockState.writeMemTyped]
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
    simp only [
      stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.setReg, Tile.scalar,
      Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, NumericDType.sub,
      FloatDType.toTileDType, FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.map,
      BlockState.writeMemTyped_real, BlockState.foldl_writeMem_masked_regs, BlockState.foldl_writeMem_prop_masked_regs, bind, Option.bind, Option.map_some,
      String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hOa, hOb, hMa, hMb, hSa, hSb, hIa, hIb, hGa, hGb, hJa, hJb, hLa, hLb, hKa, hKb, hYa, hYb, hZa, hZb, hPa, hPb, hQa, hQb]
    try simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hOb, hMb, hSb, hGb, hLb, hYb, hPb, BlockState.writeMemTyped]
    rfl
  · try simp only [Tile.scalar, Int.toNat_natCast, BlockState.writeMemAs_real]
    simp (config := {maxSteps := 2000000}) (disch := positivity) only [Int.toNat_add, Int.toNat_natCast, Int.reduceToNat]
    apply store_related
    apply store_related
    refine ⟨memory, pids, ?_, ?_, ?_, ⟨columns, ?_, ?_⟩, ⟨mask, ?_, ?_⟩,
      ?_, ?_, ?_, ?_, ⟨lr, ?_, ?_⟩, ⟨li, ?_, ?_⟩⟩
    all_goals simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
      hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hLa, hLb, hKa, hKb]
    all_goals exact ⟨_, rfl, rfl⟩

#axiomsClean write_related
#axiomsClean store_related
#axiomsClean suffix_steps
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
