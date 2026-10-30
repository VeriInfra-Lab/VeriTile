import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceBackwardCorrespondence
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

private theorem nonnegative (n : Nat) : ¬ (n : Int) < 0 := by omega

theorem forward_prefix_safe (p : BackwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (forwardPrefix p length) u ↔
    Stmt.TraceSafeList bounds (RealReference.diagSsmBackwardPreLoop p.l p.batch p.dim p.B) u := by
  simp only [forwardPrefix, forwardBody, Kernels.backwardTyped, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [List.take]
  simp only [RealReference.diagSsmBackwardPreLoop,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MemAccess.SafeAt, MaskOpt.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, stepStmt, evalOp,
    BlockState.setReg, Tile.scalar, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul,
    NumericDType.sub, ComparableDType.lt, IntegralDType.mod, IntegralDType.floorDiv,
    bind, Option.bind, true_and, and_true]
  simp only [String.reduceEq, ↓reduceDIte, ↓reduceIte, castTile_self, Region.cast_cast,
    true_and, and_true]
  simp only [Tile.cop, Tile.select, Broadcast.leftIndex, Broadcast.rightIndex, IntegerExpr.sameShape,
    ComparableDType.lt, Tile.scalar, Int.ofNat_eq_natCast, Int.add_zero, Int.zero_add,
    ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
    Nat.mul_assoc, nonnegative, Int.ofNat_lt, decide_false, Bool.false_eq_true, ↓reduceIte, source_mod]
  try rfl


#axiomsClean forward_prefix_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceBackward
