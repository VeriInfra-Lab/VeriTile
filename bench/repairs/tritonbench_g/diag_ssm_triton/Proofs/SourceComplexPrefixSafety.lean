import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceComplexCorrespondence
import VeriTile.Triton.Memory.TraceRewrite
namespace VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

private theorem nonnegative (n : Nat) : ¬ (n : Int) < 0 := by omega

theorem forward_prefix_safe (p : ForwardLayout) (length : Nat) (u : BlockState) (bounds : RegionBounds) :
    Stmt.TraceSafeList bounds (forwardPrefix p length) u ↔
    Stmt.TraceSafeList bounds (ComplexForwardSource.prelude p.s p.l p.batch p.dim p.B) u := by
  simp only [forwardPrefix, forwardBody, Kernels.complexForwardTyped, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    List.map_cons, List.map_nil, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
  simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  simp only [List.take]
  simp only [ComplexForwardSource.prelude, Erasure.complexForward,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Except.bind, bind, Bool.false_eq_true, ↓reduceIte, List.take]
  simp only [
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
  simp only [source_mod_int, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  try simp (config := {maxSteps := 2000000}) (disch := positivity) only
    [Int.toNat_add, Int.toNat_mul, Int.toNat_natCast, Int.ofNat_eq_natCast, Int.reduceToNat]
  try rfl


#axiomsClean forward_prefix_safe
end
end VeriTile.Bench.Repairs.DiagSsmTriton.SourceComplex
