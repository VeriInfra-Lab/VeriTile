import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardLoop
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexStepRelation
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem post_memory (p : BackwardLayout) (length : Nat) (a b : BlockState)
    (h : BackwardRelated p length a b) :
    ∃ aa bb, stepStmts (backwardPost p length) a = some aa ∧
      stepStmts (RealReference.diagSsmBackwardPostLoop p.gs p.gl p.B) b = some bb ∧
      aa.mem = bb.mem := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨state, hSa, hSb⟩, ⟨gradient, hGa, hGb⟩, ⟨transition, hLa, hLb⟩⟩
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  rotate_left 2
  · simp only [backwardPost, forwardBody, Kernels.backwardTyped,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      ComputeDType.eraseDType, List.map_cons, List.map_nil,
      Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
    simp only [List.drop, eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, BlockState.writeMemTyped_real,
      BlockState.foldl_writeMem_masked_regs, BlockState.setReg, Tile.scalar,
      Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, bind, Option.bind, Option.map_some,
      String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, hCa, hMa, hSa, hGa]
    rfl
  · simp only [RealReference.diagSsmBackwardPostLoop, stepStmts, stepStmt, evalOp,
      BlockState.writeMemTyped_real, BlockState.foldl_writeMem_masked_regs,
      bind, Option.bind, Option.map_some, hCb, hMb, hSb, hGb]
    rfl
  · simp only [Int.toNat_natCast, BlockState.writeMemAs_real]
    apply SourceComplex.store_memory
    apply SourceComplex.store_memory
    exact memory

#axiomsClean post_memory
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
