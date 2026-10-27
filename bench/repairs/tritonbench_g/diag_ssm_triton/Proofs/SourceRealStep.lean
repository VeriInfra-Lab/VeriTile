import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceRealLoop
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem actualForwardStep (p : ForwardLayout) (length t : Nat) (a b : BlockState)
    (h : ForwardRelated p length a b) :
    ∃ v : BlockState, stepStmts (forwardLoop p length)
      (a.setReg "t" .int [] (Tile.scalar (t : Int))) = some v := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨state, hSa, hSb⟩, ⟨transition, hLa, hLb⟩⟩
  refine ⟨?_, ?_⟩
  swap
  simp only [forwardLoop, forwardBody, Kernels.forwardTyped,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, List.map_cons, List.map_nil,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [List.getElem?_cons_zero, List.getElem?_cons_succ]
  simp only [stepStmts, stepStmt, evalOp, BlockState.setReg, Tile.scalar,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, bind, Option.bind]
  simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
    hLength, hBatch, hDim, hCa, hMa, hSa, hLa, BlockState.readMemValue, Region.cast_cast]
  try rfl

#axiomsClean actualForwardStep
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal
