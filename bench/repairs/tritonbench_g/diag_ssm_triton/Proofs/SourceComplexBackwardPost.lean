import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexBackwardLoop
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexStepRelation
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem post_memory (p : BackwardLayout) (length : Nat) (a b : BlockState)
    (h : BackwardRelated p length a b) :
    ∃ aa bb, stepStmts (backwardPost p length) a = some aa ∧
      stepStmts (ComplexBackwardSource.postlude p.gs p.gl p.B) b = some bb ∧
      aa.mem = bb.mem := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨sr, hSa, hSb⟩, ⟨si, hIa, hIb⟩,
    ⟨gr, hGa, hGb⟩, ⟨gi, hJa, hJb⟩, ⟨lr, hLa, hLb⟩, ⟨li, hKa, hKb⟩⟩
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  rotate_left 2
  · simp only [backwardPost, forwardBody, Kernels.complexBackwardTyped,
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
    simp only [List.drop, eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_masked_regs, BlockState.foldl_writeMem_prop_masked_regs, Region.cast_cast, BlockState.setReg, Tile.scalar,
      Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, bind, Option.bind, Option.map_some,
      String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hCa, hMa, hSa, hIa, hGa, hJa,
      FloatDType.toTileDType, FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.map]
    rfl
  · simp only [ComplexBackwardSource.postlude, Erasure.complexBackward,
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
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def,
      BlockState.writeMemTyped_real, BlockState.foldl_writeMem_masked_regs, BlockState.foldl_writeMem_prop_masked_regs, Region.cast_cast,
      bind, Option.bind, Option.map_some, hCb, hMb, hSb, hIb, hGb, hJb,
      FloatDType.toTileDType, FloatDType.ofWithBot, FloatDType.toWithBot, WithBot.map,
      Tile.scalar, Tile.bop, NumericDType.add, NumericDType.mul]
    rfl
  · simp only [Int.ofNat_eq_natCast, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, BlockState.writeMemAs_real]
    simp (disch := positivity) only [Int.toNat_add, Int.toNat_mul, Int.toNat_natCast, Int.reduceToNat]
    apply SourceComplex.store_memory
    apply SourceComplex.store_memory
    apply SourceComplex.store_memory
    apply SourceComplex.store_memory
    exact memory

#axiomsClean post_memory
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplexBackward
