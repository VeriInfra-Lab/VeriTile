import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceRealStepRelation
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

theorem forward_step_safe (p : ForwardLayout) (length t : Nat) (a b : BlockState)
    (h : ForwardRelated p length a b) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (forwardLoop p length)
      (a.setReg "t" .int [] (Tile.scalar (t : Int))) ↔
    Stmt.TraceSafeList bounds (RealReference.diagSsmForwardLoopBody p.x p.y p.batch p.dim p.B)
      (b.setReg "t" .nat [] (Tile.scalar t)) := by
  rcases h with ⟨memory, pids, hLength, hBatch, hDim, ⟨columns, hCa, hCb⟩,
    ⟨mask, hMa, hMb⟩, ⟨state, hSa, hSb⟩, ⟨transition, hLa, hLb⟩⟩
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
  simp only [RealReference.diagSsmForwardLoopBody,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MemAccess.SafeAt, MaskOpt.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, stepStmt, evalOp,
    BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul,
    bind, Option.bind, true_and, and_true]
  simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self,
    hLength, hBatch, hDim, hCa, hCb, hMa, hMb, hSa, hSb, hLa, hLb,
    BlockState.readMemValue, Region.cast_cast, true_and, and_true]
  simp only [Tile.scalar, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
    Nat.mul_assoc, BlockState.readMemAs, memory]
  try rfl

#axiomsClean forward_step_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceReal
