import bench.tritonbench_g.reversed_cumsum.Proofs.SourceAudit
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Erasure for real correctness preserves the complete loop, triangular dot, carry, branch and stores. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealSource
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def source (swapped : Bool) := if swapped then TypedSource.compareRight else TypedSource.original

section
variable (swapped : Bool) (dtype out_dtype : FloatDType) (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)

theorem projects : (source swapped dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS).toAlgorithm? =
    .ok (source swapped dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS).toAlgKernel := by
  cases swapped <;> rfl

theorem storage_erasure :
    (source swapped dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS).eraseDType =
    (source swapped .real .real s o s_s_h s_s_t s_s_d T S BT BS).eraseDType := by
  cases swapped
  all_goals simp only [source, Bool.false_eq_true, ↓reduceIte, TypedSource.original,
    TypedSource.compareRight, ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel,
    ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    Except.bind, bind, List.map_cons, List.map_nil, eraseDType_float]
  all_goals simp only [VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, eraseDType_float]
  all_goals cases dtype <;> cases out_dtype
  all_goals exact ⟨rfl, rfl, HEq.rfl, HEq.rfl⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealSource
